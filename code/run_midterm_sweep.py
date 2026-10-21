#!/usr/bin/env python3
"""Run and persist the midterm measurement matrix one completed row at a time."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from workload import DEFAULT_MODEL, MAX_NEW_TOKENS, load_workload, matrix_prompts

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_BASE_URLS = {
    "vllm_paged": "http://127.0.0.1:8000",
    "llama_cpp": "http://127.0.0.1:8081",
}


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _new_output(args: argparse.Namespace, wl: dict[str, Any]) -> dict[str, Any]:
    return {
        "system_id": args.system,
        "model": args.model or wl.get("model", DEFAULT_MODEL),
        "device": args.device or "auto",
        "started": _now(),
        "finished": None,
        "workload": {
            "source": str(args.workload) if args.workload else "code/workload/prompts.json",
            "max_new_tokens": int(wl.get("max_new_tokens", MAX_NEW_TOKENS)),
            "concurrency_grid": list(wl["concurrency_grid"]),
            "reps_per_cell": int(wl["reps_per_cell"]),
            "warmup_runs_per_cell": args.warmups,
        },
        "runs": [],
    }


def _load_or_create_output(
    path: Path,
    args: argparse.Namespace,
    wl: dict[str, Any],
) -> dict[str, Any]:
    if args.overwrite or not path.exists():
        return _new_output(args, wl)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or not isinstance(doc.get("runs"), list):
        raise ValueError(f"{path} is not a baseline sweep output")
    if doc.get("system_id") != args.system:
        raise ValueError(
            f"{path} belongs to {doc.get('system_id')!r}, not {args.system!r}"
        )
    expected_model = args.model or wl.get("model", DEFAULT_MODEL)
    if doc.get("model") != expected_model:
        raise ValueError(
            f"{path} uses model {doc.get('model')!r}; requested {expected_model!r}"
        )
    doc["finished"] = None
    return doc


def _write_atomic(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _completed_keys(runs: list[dict[str, Any]]) -> set[tuple[str, int, int]]:
    keys: set[tuple[str, int, int]] = set()
    for row in runs:
        prompt_id = row.get("prompt_id")
        if isinstance(prompt_id, str):
            keys.add((prompt_id, int(row["concurrency"]), int(row["rep"])))
    return keys


def _print_plan(args: argparse.Namespace, wl: dict[str, Any]) -> None:
    matrix = matrix_prompts(wl)
    count = 0
    for prompt in matrix:
        for concurrency in wl["concurrency_grid"]:
            for rep in range(int(wl["reps_per_cell"])):
                count += 1
                print(
                    f"{args.system} {prompt['id']} "
                    f"conc={concurrency} rep={rep}"
                )
    print(
        f"{count} matrix runs "
        f"({len(matrix)} prompts × {len(wl['concurrency_grid'])} concurrency "
        f"× {wl['reps_per_cell']} reps)"
    )


def _run_minimal(args: argparse.Namespace, wl: dict[str, Any]) -> None:
    # Keep list/dry-run usable before torch/transformers are installed.
    from minimal_full_server import MinimalServer, build_run_row

    output_path = args.output.resolve()
    doc = _load_or_create_output(output_path, args, wl)
    completed = _completed_keys(doc["runs"])
    matrix = matrix_prompts(wl)
    max_new_tokens = int(wl.get("max_new_tokens", MAX_NEW_TOKENS))
    reps_per_cell = int(wl["reps_per_cell"])
    model = args.model or wl.get("model", DEFAULT_MODEL)

    print(f"output={output_path}")
    print(f"model={model} device={args.device or 'auto'}")
    with MinimalServer(model, args.device) as server:
        print(f"loaded in {server.load_ms:.0f} ms on {server.device}")
        doc["device"] = server.device
        _write_atomic(output_path, doc)

        for prompt in matrix:
            prompt_id = str(prompt["id"])
            for concurrency_value in wl["concurrency_grid"]:
                concurrency = int(concurrency_value)
                missing_reps = [
                    rep
                    for rep in range(reps_per_cell)
                    if (prompt_id, concurrency, rep) not in completed
                ]
                if not missing_reps:
                    print(f"skip {prompt_id} N={concurrency}: complete")
                    continue

                for warmup in range(args.warmups):
                    print(
                        f"warmup {warmup + 1}/{args.warmups}: "
                        f"{prompt_id} N={concurrency}"
                    )
                    server.benchmark_burst(
                        prompt["text"],
                        concurrency,
                        max_new_tokens,
                    )
                server.synchronize()

                for rep in missing_reps:
                    print(f"measure {prompt_id} N={concurrency} rep={rep}")
                    clients = server.benchmark_burst(
                        prompt["text"],
                        concurrency,
                        max_new_tokens,
                    )
                    row = build_run_row(
                        clients,
                        system_id=args.system,
                        concurrency=concurrency,
                        rep=rep,
                    )
                    row["prompt_id"] = prompt_id
                    if prompt.get("target_tokens") is not None:
                        row["target_prompt_tokens"] = int(prompt["target_tokens"])
                    doc["runs"].append(row)
                    completed.add((prompt_id, concurrency, rep))
                    _write_atomic(output_path, doc)

    doc["finished"] = _now()
    _write_atomic(output_path, doc)
    print(f"complete: {len(doc['runs'])} rows in {output_path}")


def _run_openai_engine(args: argparse.Namespace, wl: dict[str, Any]) -> None:
    from minimal_full_server import build_run_row
    from openai_engine_adapter import OpenAIEngineAdapter

    output_path = args.output.resolve()
    doc = _load_or_create_output(output_path, args, wl)
    completed = _completed_keys(doc["runs"])
    matrix = matrix_prompts(wl)
    max_new_tokens = int(wl.get("max_new_tokens", MAX_NEW_TOKENS))
    reps_per_cell = int(wl["reps_per_cell"])
    config_model = args.model or wl.get("model", DEFAULT_MODEL)
    base_url = args.base_url or DEFAULT_BASE_URLS[args.system]
    served_model = args.served_model or (
        "tinyllama" if args.system == "llama_cpp" else config_model
    )
    adapter = OpenAIEngineAdapter(
        base_url=base_url,
        served_model=served_model,
        config_model=config_model,
        timeout_s=args.request_timeout,
    )
    adapter.check_health()

    print(f"output={output_path}")
    print(f"system={args.system} endpoint={base_url} served_model={served_model}")
    doc["device"] = f"cuda via {args.system} server"
    doc["endpoint"] = base_url
    doc["served_model"] = served_model
    doc["kv_method"] = "dense per-request FP16 estimate from TinyLlama config"
    _write_atomic(output_path, doc)

    for prompt in matrix:
        prompt_id = str(prompt["id"])
        expected_prompt_tokens = int(prompt["prompt_tokens"])
        for concurrency_value in wl["concurrency_grid"]:
            concurrency = int(concurrency_value)
            missing_reps = [
                rep
                for rep in range(reps_per_cell)
                if (prompt_id, concurrency, rep) not in completed
            ]
            if not missing_reps:
                print(f"skip {prompt_id} N={concurrency}: complete")
                continue

            for warmup in range(args.warmups):
                print(
                    f"warmup {warmup + 1}/{args.warmups}: "
                    f"{prompt_id} N={concurrency}"
                )
                adapter.benchmark_burst(
                    prompt["text"],
                    concurrency,
                    max_new_tokens,
                    expected_prompt_tokens,
                )

            for rep in missing_reps:
                print(f"measure {prompt_id} N={concurrency} rep={rep}")
                clients = adapter.benchmark_burst(
                    prompt["text"],
                    concurrency,
                    max_new_tokens,
                    expected_prompt_tokens,
                )
                row = build_run_row(
                    clients,
                    system_id=args.system,
                    concurrency=concurrency,
                    rep=rep,
                )
                row["prompt_id"] = prompt_id
                if prompt.get("target_tokens") is not None:
                    row["target_prompt_tokens"] = int(prompt["target_tokens"])
                doc["runs"].append(row)
                completed.add((prompt_id, concurrency, rep))
                _write_atomic(output_path, doc)

    doc["finished"] = _now()
    _write_atomic(output_path, doc)
    print(f"complete: {len(doc['runs'])} rows in {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Midterm benchmark sweep")
    parser.add_argument("--workload", type=Path, default=None)
    parser.add_argument("--list-prompts", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--system",
        default="minimal_b1",
        choices=["minimal_b1", "vllm_paged", "llama_cpp"],
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=None, help="cuda | cpu | mps")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--served-model", default=None)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    if args.output is None:
        args.output = RESULTS_DIR / f"{args.system}-runs.json"

    wl = load_workload(args.workload)
    if args.list_prompts:
        for prompt in wl["prompts"]:
            print(
                f"{prompt['id']}: {prompt['prompt_tokens']} tok — "
                f"{prompt['text'][:100]}..."
            )
        return
    if args.dry_run:
        _print_plan(args, wl)
        return
    if args.system == "minimal_b1":
        _run_minimal(args, wl)
    else:
        _run_openai_engine(args, wl)


if __name__ == "__main__":
    main()
