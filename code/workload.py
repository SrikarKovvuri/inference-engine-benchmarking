"""Load shared midterm prompts (code/workload/prompts.json)."""

from __future__ import annotations

import json
from pathlib import Path

WORKLOAD_PATH = Path(__file__).resolve().parent / "workload" / "prompts.json"

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW_TOKENS = 64
CONCURRENCY_GRID = [1, 4]
REPS_PER_CELL = 3
PROMPT_TARGETS = [128, 512, 2048]


def load_workload(path: Path | str | None = None) -> dict:
    p = Path(path) if path else WORKLOAD_PATH
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def matrix_prompts(wl: dict) -> list[dict]:
    return [p for p in wl["prompts"] if p.get("kind", "synthetic") == "synthetic"]
