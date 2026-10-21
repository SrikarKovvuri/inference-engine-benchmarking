#!/usr/bin/env python3
"""Combine per-engine sweep JSON into report/src/data/srk2211-midterm-results.json."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"
OUT = REPO / "report" / "src" / "data" / "srk2211-midterm-results.json"
ORDER = {"minimal_b1": 0, "vllm_paged": 1, "llama_cpp": 2}
PROMPT_ORDER = {"syn_128": 0, "syn_512": 1, "syn_2048": 2}


def load_sweep(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main() -> None:
    sweeps = [
        load_sweep("minimal_b1-runs.json"),
        load_sweep("vllm_paged-runs.json"),
        load_sweep("llama_cpp-runs.json"),
    ]
    runs: list[dict] = []
    for sweep in sweeps:
        runs.extend(sweep["runs"])
    runs.sort(
        key=lambda r: (
            ORDER[r["system_id"]],
            PROMPT_ORDER.get(r.get("prompt_id", ""), 99),
            int(r["concurrency"]),
            int(r["rep"]),
        )
    )

    started = min(s["started"] for s in sweeps)
    finished = max(s["finished"] for s in sweeps)

    def cell(sid: str, prompt_id: str, concurrency: int) -> list[dict]:
        return [
            r
            for r in runs
            if r["system_id"] == sid
            and r.get("prompt_id") == prompt_id
            and r["concurrency"] == concurrency
        ]

    def ttft_samples(sid: str, prompt_id: str, concurrency: int) -> list[float]:
        out: list[float] = []
        for row in cell(sid, prompt_id, concurrency):
            samples = row.get("ttft_ms_samples")
            if isinstance(samples, list) and samples:
                out.extend(float(v) for v in samples)
            else:
                out.append(float(row["ttft_ms"]))
        return out

    def med_ttft(sid: str, prompt_id: str, concurrency: int) -> float:
        return float(median(r["ttft_ms"] for r in cell(sid, prompt_id, concurrency)))

    def med_tbt(sid: str, prompt_id: str, concurrency: int) -> float:
        return float(median(r["tbt_ms_p50"] for r in cell(sid, prompt_id, concurrency)))

    base_n1 = med_ttft("minimal_b1", "syn_512", 1)
    base_n4 = med_ttft("minimal_b1", "syn_512", 4)
    vllm_n1 = med_ttft("vllm_paged", "syn_512", 1)
    vllm_n4 = med_ttft("vllm_paged", "syn_512", 4)
    llama_n1 = med_ttft("llama_cpp", "syn_512", 1)
    llama_n4 = med_ttft("llama_cpp", "syn_512", 4)
    base_tail = max(ttft_samples("minimal_b1", "syn_512", 4))
    vllm_tail = max(ttft_samples("vllm_paged", "syn_512", 4))
    llama_tail = max(ttft_samples("llama_cpp", "syn_512", 4))
    base_tbt = med_tbt("minimal_b1", "syn_512", 1)
    vllm_tbt = med_tbt("vllm_paged", "syn_512", 1)
    llama_tbt = med_tbt("llama_cpp", "syn_512", 1)
    base_long = med_ttft("minimal_b1", "syn_2048", 1)
    vllm_long = med_ttft("vllm_paged", "syn_2048", 1)
    llama_long = med_ttft("llama_cpp", "syn_2048", 1)

    doc = {
        "meta": {
            "uni": "srk2211",
            "name": "Srikar Kovvuri",
            "github": "SrikarKovvuri",
            "date": "2026-10-20",
            "title": "Midterm: From minimal_full_server to Production Engines",
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "device": "NVIDIA GeForce RTX 4060 Laptop GPU · 8 GB · CUDA 12.4 (WSL2 + Windows)",
            "repo_url": "https://github.com/networks-for-ai-coms6998-hw/midterm-srk2211",
        },
        "source": {
            "harness": (
                "code/run_midterm_sweep.py with code/minimal_full_server.py and "
                "code/openai_engine_adapter.py; validated by code/export_midterm_results.py. "
                "Baseline and vLLM ran in WSL2 Ubuntu (Python 3.12, vllm==0.8.5.post1, "
                "torch 2.6.0+cu124). llama.cpp used the official Windows CUDA 12.4 "
                "llama-server b11765 against TheBloke TinyLlama Q8_0 GGUF. "
                "vLLM: `vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dtype half "
                "--max-model-len 2048 --gpu-memory-utilization 0.75`. "
                "llama.cpp: `llama-server -m tinyllama-1.1b-chat-v1.0.Q8_0.gguf "
                "-ngl 99 -c 8192 --parallel 4` (8192 is a shared KV pool so 4 clients "
                "each keep a 2048-token slot)."
            ),
            "json_file": "report/src/data/srk2211-midterm-results.json",
            "started": started,
            "finished": finished,
            "workload": {
                "prompt_token_targets": [128, 512, 1920],
                "max_new_tokens": 64,
                "concurrency_grid": [1, 4],
                "reps_per_cell": 3,
                "warmup_runs": 1,
                "notes": (
                    "Synthetic TCP/networking prompts from code/workload/prompts.json, "
                    "sized with TinyLlama chat template (add_generation_prompt=true). "
                    "The long cell is labeled syn_2048 but capped at 1920 templated "
                    "tokens so prompt + 64 decode fits TinyLlama's 2048-token context. "
                    "HF engines (baseline, vLLM) measured 128 / 512 / 1920 tokens. "
                    "llama.cpp counted 131 / 522 / 1956 on the same texts. "
                    "WildChat train indices 42 and 2048 were generated for sanity "
                    "checks but are not part of the 18-row matrix. "
                    "KV bytes are the dense FP16 GQA formula from HW1 "
                    "(2 * 2B * 22 layers * 4 KV heads * seq_len * 64). "
                    "llama.cpp used Q8_0 GGUF; that quantization is a confounder "
                    "versus the FP16 Hugging Face weights used by baseline and vLLM. "
                    "llama.cpp long-prompt TTFT stayed ~34-60 ms after a same-prompt "
                    "warmup, consistent with prefix reuse in addition to the C++ runtime."
                ),
            },
        },
        "systems": [
            {
                "id": "minimal_b1",
                "label": "minimal_full_server (batch=1)",
                "family": "baseline",
                "description": (
                    "Week 2 five-step server instrumented in code/minimal_full_server.py: "
                    "one worker, FIFO queue, contiguous Hugging Face DynamicCache, "
                    "no iteration-level batching. Concurrent clients start together; "
                    "queue wait is included in TTFT."
                ),
                "novelty": (
                    "Contiguous DynamicCache and no iteration-level batching — "
                    "N>1 TTFT becomes a staircase because later clients wait for "
                    "each earlier request's full prefill + decode."
                ),
            },
            {
                "id": "vllm_paged",
                "label": "vLLM PagedAttention",
                "family": "vllm",
                "description": (
                    "vllm 0.8.5.post1 serving TinyLlama FP16 over the OpenAI-compatible "
                    "HTTP API. Concurrent streaming clients from "
                    "code/openai_engine_adapter.py."
                ),
                "novelty": (
                    "PagedAttention plus continuous batching: at N=4, decode TBT stays "
                    "near the N=1 value while TTFT rises from shared prefill, not from "
                    "serializing whole requests."
                ),
                "repo": "https://github.com/vllm-project/vllm",
            },
            {
                "id": "llama_cpp",
                "label": "llama.cpp llama-server (Q8_0)",
                "family": "llama.cpp",
                "description": (
                    "Official Windows CUDA 12.4 llama-server b11765, TinyLlama 1.1B "
                    "Chat Q8_0 GGUF, 4 parallel slots, 8192-token shared context pool. "
                    "Same OpenAI-compatible streaming client as vLLM."
                ),
                "novelty": (
                    "C++/GGUF runtime with slot-based continuous batching; KV is "
                    "preallocated per slot. Q8_0 quantization and possible prefix "
                    "reuse after warmup both affect the numbers and are disclosed."
                ),
                "repo": "https://github.com/ggml-org/llama.cpp",
            },
        ],
        "runs": runs,
        "takeaways": [
            (
                f"At concurrency 1, 512-token TTFT is {base_n1:.0f} ms (baseline), "
                f"{vllm_n1:.0f} ms (vLLM), and {llama_n1:.0f} ms (llama.cpp Q8). "
                f"TBT p50 is {base_tbt:.1f} / {vllm_tbt:.1f} / {llama_tbt:.1f} ms. "
                "llama.cpp's gap includes Q8 quantization, not only scheduling."
            ),
            (
                f"The baseline novelty is queueing, not decode speed: at N=4 and 512 "
                f"tokens, median TTFT jumps from {base_n1:.0f} ms to {base_n4:.0f} ms "
                f"and the slowest client hits {base_tail:.0f} ms, while per-token TBT "
                "stays ~50 ms. Four clients share one worker, so TTFT is a staircase."
            ),
            (
                f"vLLM continuous batching cuts that tail: N=4 512-token median TTFT "
                f"is {vllm_n4:.0f} ms (max sample {vllm_tail:.0f} ms) versus the "
                f"baseline's {base_n4:.0f} / {base_tail:.0f} ms. TBT stays near the "
                "single-user value because decode iterations are shared."
            ),
            (
                f"llama.cpp slot batching keeps N=4 512-token TTFT at {llama_n4:.0f} ms "
                f"(max sample {llama_tail:.0f} ms). Long-prompt N=1 TTFT is "
                f"{llama_long:.0f} ms after warmup versus {base_long:.0f} / "
                f"{vllm_long:.0f} ms for the FP16 engines, treating that as Q8 + "
                "possible prefix reuse, not a pure scheduler win."
            ),
            (
                "Dense KV bytes scale linearly with seq_len and match across the two "
                "FP16 engines (4.3 / 13.0 / 44.7 MB at 192 / 576 / 1984 tokens). "
                "llama.cpp is a few percent higher only because its tokenizer counted "
                "a few extra prompt tokens on the same text."
            ),
        ],
        "summary": (
            f"On an RTX 4060 Laptop (8 GB) with TinyLlama 1.1B, the Week 2 "
            f"minimal server, vLLM 0.8.5, and llama.cpp Q8 were compared on the "
            f"same 128 / 512 / 1920-token prompts at concurrency 1 and 4 (3 reps "
            f"after 1 warmup). Baseline TTFT explodes at N=4 because it finishes "
            f"one full request before starting the next ({base_n4:.0f} ms median "
            f"at 512 tokens, {base_tail:.0f} ms worst client). vLLM's continuous "
            f"batching and PagedAttention keep the same N=4 cell at {vllm_n4:.0f} ms "
            f"median / {vllm_tail:.0f} ms max while TBT stays near the N=1 line. "
            f"llama.cpp is the fastest wall-clock engine ({llama_n4:.0f} ms median "
            f"TTFT at the same cell), but it ran a Q8 GGUF after a same-prompt "
            f"warmup, so that comparison is not FP16-apples-to-apples. KV bytes "
            f"follow the dense GQA formula on every engine."
        ),
        "plot": {
            "title": "Median TTFT vs prompt length (concurrency = 1)",
            "metric": "ttft_ms",
            "x_field": "prompt_tokens",
            "series_field": "system_id",
            "aggregate": "median",
            "filter": {"concurrency": 1},
            "y_label": "TTFT (ms)",
            "x_label": "Prompt tokens",
        },
        "boxplot": {
            "title": "TTFT distribution per client (512-token target prompt)",
            "subtitle": (
                "Each box pools one sample per concurrent client across 3 repetitions. "
                "Filter is prompt_id=syn_512 so llama.cpp's 522-token count is included "
                "with the HF 512-token rows."
            ),
            "metric": "ttft_ms",
            "samples_field": "ttft_ms_samples",
            "group_field": "system_id",
            "default_concurrency": 4,
            "filter": {"prompt_id": "syn_512"},
            "y_label": "TTFT (ms)",
            "metrics": [
                {"key": "ttft_ms", "label": "TTFT", "unit": "ms"},
                {"key": "tbt_ms_p50", "label": "TBT", "unit": "ms"},
                {"key": "kv_bytes", "label": "KV bytes", "unit": ""},
            ],
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
