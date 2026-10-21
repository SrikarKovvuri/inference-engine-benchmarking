#!/usr/bin/env python3
"""OpenAI-compatible streaming adapter for vLLM and llama.cpp."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from statistics import mean, median
from typing import Any

import requests
from transformers import AutoConfig

from export_midterm_results import kv_bytes as dense_kv_bytes
from minimal_full_server import ClientMetrics


class OpenAIEngineAdapter:
    """Measure an OpenAI-compatible chat server with barrier-style bursts."""

    def __init__(
        self,
        *,
        base_url: str,
        served_model: str,
        config_model: str,
        timeout_s: float = 300.0,
        kv_itemsize: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.served_model = served_model
        self.timeout_s = timeout_s
        self.kv_itemsize = kv_itemsize
        self.config = AutoConfig.from_pretrained(config_model)

    def check_health(self) -> None:
        """Fail early when the selected engine server is not reachable."""
        response = requests.get(f"{self.base_url}/health", timeout=10)
        response.raise_for_status()

    def _kv_bytes(self, seq_len: int) -> int:
        n_layers = int(self.config.num_hidden_layers)
        n_attention_heads = int(self.config.num_attention_heads)
        n_kv_heads = int(
            getattr(self.config, "num_key_value_heads", n_attention_heads)
        )
        head_dim = int(
            getattr(
                self.config,
                "head_dim",
                self.config.hidden_size // n_attention_heads,
            )
        )
        return dense_kv_bytes(
            n_layers=n_layers,
            n_kv_heads=n_kv_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            batch=1,
            itemsize=self.kv_itemsize,
        )

    def _request(
        self,
        prompt: str,
        max_new_tokens: int,
        expected_prompt_tokens: int,
        barrier: threading.Barrier,
    ) -> ClientMetrics:
        barrier.wait()
        request_start = time.perf_counter()
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.served_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_new_tokens,
                "temperature": 0,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            headers={"Accept": "text/event-stream"},
            stream=True,
            timeout=self.timeout_s,
        )
        if not response.ok:
            raise RuntimeError(
                f"{self.base_url} returned HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

        token_times: list[float] = []
        pieces: list[str] = []
        usage: dict[str, Any] = {}
        for raw_line in response.iter_lines(chunk_size=1, decode_unicode=True):
            if not raw_line or not raw_line.startswith("data:"):
                continue
            payload = raw_line[5:].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            if event.get("error"):
                raise RuntimeError(f"engine stream error: {event['error']}")
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if isinstance(content, str) and content:
                    token_times.append(time.perf_counter())
                    pieces.append(content)

        if not token_times:
            raise RuntimeError("engine stream completed without a generated token")

        prompt_tokens = int(usage.get("prompt_tokens", expected_prompt_tokens))
        new_tokens = int(usage.get("completion_tokens", len(token_times)))
        seq_len = prompt_tokens + new_tokens
        steps_s = [
            later - earlier
            for earlier, later in zip(token_times, token_times[1:])
        ]
        steps_ms = [duration * 1000.0 for duration in steps_s]

        return ClientMetrics(
            prompt_tokens=prompt_tokens,
            new_tokens=new_tokens,
            ttft_ms=(token_times[0] - request_start) * 1000.0,
            decode_step_durations_s=steps_s,
            tbt_ms_p50=float(median(steps_ms)) if steps_ms else 0.0,
            tbt_ms_mean=float(mean(steps_ms)) if steps_ms else 0.0,
            kv_bytes=self._kv_bytes(seq_len),
            kv_seq_len=seq_len,
            materialized_kv_seq_len=max(prompt_tokens, seq_len - 1),
            # The HTTP API returns text rather than token IDs. This field is
            # private baseline output state and is not used in report rows.
            generated_token_ids=[0] * new_tokens,
        )

    def benchmark_burst(
        self,
        prompt: str,
        concurrency: int,
        max_new_tokens: int,
        expected_prompt_tokens: int,
    ) -> list[ClientMetrics]:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        barrier = threading.Barrier(concurrency + 1)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            pending = [
                pool.submit(
                    self._request,
                    prompt,
                    max_new_tokens,
                    expected_prompt_tokens,
                    barrier,
                )
                for _ in range(concurrency)
            ]
            barrier.wait()
            return [future.result() for future in pending]
