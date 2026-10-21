#!/usr/bin/env python3
"""Single-request TinyLlama baseline with midterm benchmark instrumentation."""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from statistics import mean, median
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from export_midterm_results import kv_bytes as dense_kv_bytes
from export_midterm_results import tbt_ms_per_step

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_MAX_NEW_TOKENS = 64


def kv_seq_len(kv_cache: Any) -> int:
    """Return the sequence materialized in a Hugging Face KV cache."""
    if hasattr(kv_cache, "get_seq_length"):
        return int(kv_cache.get_seq_length())
    return int(kv_cache[0][0].shape[2])


@dataclass
class ClientMetrics:
    """Measurements and private output state for one client request."""

    prompt_tokens: int
    new_tokens: int
    ttft_ms: float
    decode_step_durations_s: list[float]
    tbt_ms_p50: float
    tbt_ms_mean: float
    kv_bytes: int
    kv_seq_len: int
    materialized_kv_seq_len: int
    generated_token_ids: list[int]


@dataclass
class _QueuedRequest:
    prompt: str
    arrival_time: float
    max_new_tokens: int
    result: Future[ClientMetrics]


class MinimalServer:
    """
    One-model, one-worker baseline.

    Concurrent clients enter a FIFO queue, but the worker finishes a complete
    request before serving the next one. This deliberately does not implement
    continuous or iteration-level batching.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_type = torch.device(self.device).type
        dtype = torch.float32 if self.device_type == "cpu" else torch.float16

        load_start = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        self.model.to(self.device)
        self.model.eval()
        self.load_ms = (time.perf_counter() - load_start) * 1000.0

        self._requests: queue.Queue[_QueuedRequest | None] = queue.Queue()
        self._worker = threading.Thread(
            target=self._serve,
            name="minimal-server-worker",
            daemon=True,
        )
        self._closed = False
        self._worker.start()

    def synchronize(self) -> None:
        """Wait for queued accelerator work; CPU execution is already synchronous."""
        if self.device_type == "cuda":
            torch.cuda.synchronize(torch.device(self.device))
        elif self.device_type == "mps":
            torch.mps.synchronize()

    def _dense_kv_bytes(self, seq_len: int) -> int:
        config = self.model.config
        n_layers = int(config.num_hidden_layers)
        n_attention_heads = int(config.num_attention_heads)
        n_kv_heads = int(getattr(config, "num_key_value_heads", n_attention_heads))
        head_dim = int(getattr(config, "head_dim", config.hidden_size // n_attention_heads))
        itemsize = next(self.model.parameters()).element_size()
        return dense_kv_bytes(
            n_layers=n_layers,
            n_kv_heads=n_kv_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            batch=1,
            itemsize=itemsize,
        )

    def _context_limit(self) -> int | None:
        value = getattr(self.model.config, "max_position_embeddings", None)
        return int(value) if value is not None else None

    def _benchmark_request(
        self,
        prompt: str,
        arrival_time: float,
        max_new_tokens: int,
    ) -> ClientMetrics:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")

        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.device)
        prompt_tokens = int(input_ids.shape[1])
        logical_kv_seq_len = prompt_tokens + max_new_tokens
        context_limit = self._context_limit()
        if context_limit is not None and logical_kv_seq_len > context_limit:
            raise ValueError(
                f"prompt ({prompt_tokens}) + output ({max_new_tokens}) exceeds "
                f"model context limit ({context_limit})"
            )

        decode_step_durations_s: list[float] = []
        generated_token_ids: list[int] = []

        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, use_cache=True, return_dict=True)
            kv_cache = outputs.past_key_values
            next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            self.synchronize()
            first_token_ready = time.perf_counter()
            generated_token_ids.append(int(next_token_id.item()))
            previous_token_ready = first_token_ready

            # Prefill produces token 1; each subsequent full forward pass produces
            # one more token. EOS is intentionally ignored for a fixed-size workload.
            for _ in range(max_new_tokens - 1):
                outputs = self.model(
                    input_ids=next_token_id,
                    past_key_values=kv_cache,
                    use_cache=True,
                    return_dict=True,
                )
                kv_cache = outputs.past_key_values
                next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                self.synchronize()
                token_ready = time.perf_counter()
                decode_step_durations_s.append(token_ready - previous_token_ready)
                previous_token_ready = token_ready
                generated_token_ids.append(int(next_token_id.item()))

        tbt = tbt_ms_per_step(decode_step_durations_s)
        return ClientMetrics(
            prompt_tokens=prompt_tokens,
            new_tokens=len(generated_token_ids),
            ttft_ms=(first_token_ready - arrival_time) * 1000.0,
            decode_step_durations_s=decode_step_durations_s,
            tbt_ms_p50=tbt["tbt_ms_p50"],
            tbt_ms_mean=tbt["tbt_ms_mean"],
            kv_bytes=self._dense_kv_bytes(logical_kv_seq_len),
            kv_seq_len=logical_kv_seq_len,
            # The final sampled token has not yet been fed back into the model,
            # so the materialized DynamicCache is one token behind the logical
            # sequence represented by prompt + generated output.
            materialized_kv_seq_len=kv_seq_len(kv_cache),
            generated_token_ids=generated_token_ids,
        )

    def _serve(self) -> None:
        while True:
            request = self._requests.get()
            try:
                if request is None:
                    return
                try:
                    metrics = self._benchmark_request(
                        request.prompt,
                        request.arrival_time,
                        request.max_new_tokens,
                    )
                except BaseException as exc:
                    request.result.set_exception(exc)
                else:
                    request.result.set_result(metrics)
            finally:
                self._requests.task_done()

    def submit(
        self,
        prompt: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        *,
        arrival_time: float | None = None,
    ) -> ClientMetrics:
        if self._closed:
            raise RuntimeError("server is closed")
        result: Future[ClientMetrics] = Future()
        self._requests.put(
            _QueuedRequest(
                prompt=prompt,
                arrival_time=arrival_time if arrival_time is not None else time.perf_counter(),
                max_new_tokens=max_new_tokens,
                result=result,
            )
        )
        return result.result()

    def benchmark_burst(
        self,
        prompt: str,
        concurrency: int,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> list[ClientMetrics]:
        """Release N clients together and return one result per client."""
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self.synchronize()
        barrier = threading.Barrier(concurrency + 1)

        def client() -> ClientMetrics:
            barrier.wait()
            arrival_time = time.perf_counter()
            return self.submit(
                prompt,
                max_new_tokens=max_new_tokens,
                arrival_time=arrival_time,
            )

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            pending = [pool.submit(client) for _ in range(concurrency)]
            barrier.wait()
            return [future.result() for future in pending]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._requests.put(None)
        self._requests.join()
        self._worker.join()

    def __enter__(self) -> MinimalServer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def build_run_row(
    clients: list[ClientMetrics],
    *,
    system_id: str,
    concurrency: int,
    rep: int,
) -> dict[str, Any]:
    """Aggregate a burst into the JSON row required by the report."""
    if len(clients) != concurrency:
        raise ValueError(f"expected {concurrency} clients, got {len(clients)}")
    prompt_counts = {client.prompt_tokens for client in clients}
    new_token_counts = {client.new_tokens for client in clients}
    kv_lengths = {client.kv_seq_len for client in clients}
    if len(prompt_counts) != 1 or len(new_token_counts) != 1 or len(kv_lengths) != 1:
        raise ValueError("clients in one burst must use the same workload")

    ttft_samples = [client.ttft_ms for client in clients]
    tbt_samples = [client.tbt_ms_p50 for client in clients]
    kv_samples = [client.kv_bytes for client in clients]
    all_steps_s = [
        duration
        for client in clients
        for duration in client.decode_step_durations_s
    ]
    all_tbt_ms = [duration * 1000.0 for duration in all_steps_s]

    row: dict[str, Any] = {
        "system_id": system_id,
        "prompt_tokens": prompt_counts.pop(),
        "concurrency": concurrency,
        "rep": rep,
        "ttft_ms": round(float(median(ttft_samples)), 3),
        "tbt_ms_p50": round(float(median(all_tbt_ms)), 3) if all_tbt_ms else 0.0,
        "tbt_ms_mean": round(float(mean(all_tbt_ms)), 3) if all_tbt_ms else 0.0,
        "kv_bytes": int(median(kv_samples)),
        "kv_seq_len": kv_lengths.pop(),
        "new_tokens": new_token_counts.pop(),
    }
    if concurrency > 1:
        row.update(
            {
                "ttft_ms_samples": [round(value, 3) for value in ttft_samples],
                "tbt_ms_samples": [round(value, 3) for value in tbt_samples],
                "kv_bytes_samples": kv_samples,
            }
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the minimal TinyLlama server")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None, help="cuda | cpu | mps")
    parser.add_argument("--prompt", default=None, help="Input prompt; otherwise read stdin")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--json", action="store_true", help="Print one report-compatible row")
    args = parser.parse_args()

    prompt = args.prompt if args.prompt is not None else input("Prompt: ").strip()
    if not prompt:
        print("Empty prompt.", file=sys.stderr)
        sys.exit(1)

    with MinimalServer(args.model, args.device) as server:
        clients = server.benchmark_burst(prompt, args.concurrency, args.max_tokens)
        row = build_run_row(
            clients,
            system_id="minimal_b1",
            concurrency=args.concurrency,
            rep=0,
        )
        if args.json:
            print(json.dumps(row, indent=2))
            return

        print(f"model={server.model_id}")
        print(f"device={server.device}")
        print(f"load_ms={server.load_ms:.1f}")
        print(json.dumps(row, indent=2))
        print("output:", server.tokenizer.decode(clients[0].generated_token_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
