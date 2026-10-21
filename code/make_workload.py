#!/usr/bin/env python3
"""Regenerate code/workload/prompts.json (synthetic 128/512/2048 + optional --wildchat)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from workload import (
    CONCURRENCY_GRID,
    DEFAULT_MODEL,
    MAX_NEW_TOKENS,
    PROMPT_TARGETS,
    REPS_PER_CELL,
    WORKLOAD_PATH,
)

SEED = (
    "Explain how TCP uses sequence numbers, acknowledgments, congestion control, "
    "and flow control when transferring data across a wide-area network. "
    "Include slow start, congestion avoidance, fast retransmit, and the role of RTT. "
)


def templated_len(tokenizer, text: str) -> int:
    messages = [{"role": "user", "content": text}]
    ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    )
    return int(ids.shape[-1])


def trim_to(tokenizer, body: str, build: int) -> tuple[str, int]:
    lo, hi = 0, len(body)
    best, best_n, best_d = body, templated_len(tokenizer, body), 10**9
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = body[:mid].rstrip()
        if not cand:
            lo = mid + 1
            continue
        n = templated_len(tokenizer, cand)
        d = abs(n - build)
        if d < best_d or (d == best_d and n >= build):
            best, best_n, best_d = cand, n, d
        if n < build:
            lo = mid + 1
        else:
            hi = mid - 1
    return best, best_n


def fit_synthetic(tokenizer, label: int, build: int) -> dict[str, Any]:
    body = SEED
    while templated_len(tokenizer, body) < build:
        body += " " + SEED
    body, n = trim_to(tokenizer, body, build)
    note = f"~{build} templated tokens"
    if build != label:
        note += f" (label {label}, capped for 2048 context + decode)"
    return {
        "id": f"syn_{label}",
        "kind": "synthetic",
        "target_tokens": label,
        "prompt_tokens": n,
        "text": body,
        "notes": note,
    }


def wildchat_rows(tokenizer, dataset: str, indices: list[int]) -> list[dict[str, Any]]:
    from datasets import load_dataset

    stream = load_dataset(dataset, split="train", streaming=True)
    out: list[dict[str, Any]] = []
    for idx in indices:
        row = next(iter(stream.skip(idx).take(1)))
        text = None
        if "conversation" in row:
            for turn in row["conversation"]:
                if turn.get("role") == "user":
                    text = turn.get("content") or turn.get("text")
                    break
        text = text or row.get("prompt") or row.get("text")
        if not text:
            raise ValueError(f"row {idx}: no user text")
        text = str(text).strip()
        out.append(
            {
                "id": f"wildchat_{idx}",
                "kind": "wildchat",
                "prompt_tokens": templated_len(tokenizer, text),
                "text": text,
                "source_index": idx,
                "notes": f"{dataset}[{idx}]",
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", type=Path, default=WORKLOAD_PATH)
    ap.add_argument("--wildchat", action="store_true")
    ap.add_argument("--wildchat-dataset", default="allenai/WildChat")
    ap.add_argument("--wildchat-indices", type=int, nargs="+", default=[42, 2048])
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    slack = MAX_NEW_TOKENS + 16
    prompts = []
    for label in PROMPT_TARGETS:
        build = label if label <= 512 else min(label, 2048 - slack)
        p = fit_synthetic(tokenizer, label, build)
        prompts.append(p)
        print(f"{p['id']}: {p['prompt_tokens']} tokens")

    if args.wildchat:
        for p in wildchat_rows(tokenizer, args.wildchat_dataset, args.wildchat_indices):
            prompts.append(p)
            print(f"{p['id']}: {p['prompt_tokens']} tokens")

    doc = {
        "model": args.model,
        "max_new_tokens": MAX_NEW_TOKENS,
        "concurrency_grid": CONCURRENCY_GRID,
        "reps_per_cell": REPS_PER_CELL,
        "prompts": prompts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
