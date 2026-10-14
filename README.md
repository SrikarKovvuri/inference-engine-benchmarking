# Midterm: From `minimal_full_server.py` to Production Engines

## Overview

Week 2’s `minimal_full_server.py` is a **single-request, colocated** inference loop:
load → tokenize → prefill + KV cache → autoregressive decode → exit. There is no
continuous batching, no paged KV blocks, no prefill/decode (PD) split, and no
cluster-wide KV pool.

By midterm you have studied systems that **keep the same math** but change
**scheduling, memory layout, and placement**: Orca (iteration-level scheduling),
vLLM (PagedAttention + continuous batching), DistServe / PD disaggregation,
Mooncake (hash-addressed KV pool and prefix reuse), etc.

**Your job:** run a controlled comparison that **bridges** the minimal server
mental model to **two** production-style stacks you can actually execute — **three
systems in total** (baseline + 2). Document hardware (GCP or local CPU/GPU).

## At a glance: run vs build

This midterm is **mostly measurement + writeup**, not implementing a new inference
engine from scratch. The React report and charts are **already built** in this repo.

| | What |
|---|------|
| **You run** | **At least three systems:** baseline + **two** menu picks (minimum). Run the prompt × concurrency sweep; record TTFT, TBT, and KV bytes. **More engines are welcome** — add each to `systems[]`, `runs[]`, and your plots. |
| **You build** | (1) A **measurement harness** or notebook/scripts under `code/` that produces numbers (or documented manual procedure). (2) **`report/src/data/<UNI>-midterm-results.json`**. (3) **`report/src/results.js`** (one import line). (4) **Discussion** text in `report/src/App.jsx`. (5) **`takeaways` / `summary` / `systems[]`** — each system needs a **`novelty`** string you tie to your plots. |
| **We provide** | Vite report (`report/`), `MetricsPlot`, `DistributionBoxplot`, sample JSON, `code/export_midterm_results.py` helpers, GitHub Pages workflow. |

**There is no `minimal_full_server.py` in this repo.** Copy it from the course
Week 2 tree (or your HW1 tree) into your `code/` folder, or call it by path — but
your JSON `source.harness` must say exactly what you ran.

### At least three systems (baseline + 2, more OK)

**Minimum:** **3** entries in `systems[]` — **`minimal_full_server` baseline** plus **2**
menu picks. Benchmark **additional** stacks if you want; put **every reported engine** in
`systems[]`, `runs[]`, and the line/box plots (hideable sections can group extras). You do
**not** need every lecture topic for the minimum.

**Lecture menu** (map each pick to `systems[].novelty`):

| Topic | GitHub / source | Typical `system_id` |
|-------|-----------------|---------------------|
| **Baseline** | Course `week-02/code/minimal_full_server.py` (copy to `code/`) | `minimal_b1` |
| **vLLM** (PagedAttention + batching) | [vllm-project/vllm](https://github.com/vllm-project/vllm) | `vllm_paged` |
| **llama.cpp** (`llama-server`) | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | `llama_cpp` |
| **Orca** (iteration scheduling — no separate binary) | Same as vLLM or llama.cpp at **N&gt;1** | `orca_map` if listed separately from vLLM |
| **DistServe** (PD disaggregation) | [LLMServe/DistServe](https://github.com/LLMServe/DistServe) | `pd_split` |
| **Mooncake** (cluster KV / prefix reuse) | [kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake) | `mooncake` |

Install from the linked repos; document exact commands in `source.harness`. Only the
**baseline** file ships in the course tree — everything else is upstream.

**Valid examples (3 `system_id`s in JSON):**

- baseline + vLLM + llama.cpp  
- baseline + vLLM + Mooncake  
- baseline + vLLM + DistServe  

**Invalid:** only baseline, or baseline + one engine. **Encouraged (not required):** a fourth
or fifth system — fully documented in JSON and figures like the first three.

If you cannot run a menu item (hardware, time), swap in **another menu item** you *did*
run and say so in `systems[].description`.

### Example `systems[]` rows

For **each** `system_id` in JSON, fill **`novelty`** and tie it to your plots in Discussion.

| `system_id` (example) | GitHub repo (set `systems[].repo`) | What you run | **Key novelty** (→ `novelty`) |
|-------------------------|-------------------------------------|--------------|--------------------------------|
| `minimal_b1` | *(none — course Week 2 file)* | `python code/minimal_full_server.py …` | Contiguous `DynamicCache`; no iteration batching |
| `vllm_paged` | [vllm-project/vllm](https://github.com/vllm-project/vllm) | `vllm serve …` + concurrent clients | PagedAttention; continuous batching at N&gt;1 |
| `llama_cpp` | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | `llama-server -m …` (GGUF) | C++/GGUF runtime vs HF minimal server |
| `orca_map` | [vllm-project/vllm](https://github.com/vllm-project/vllm) | vLLM at N&gt;1 | Iteration-level batching (Orca) |
| `mooncake` | [kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake) | Mooncake store / serving path per repo docs | Cluster KV pool; prefix reuse (cold vs warm TTFT) |
| `pd_split` | [LLMServe/DistServe](https://github.com/LLMServe/DistServe) | DistServe or documented PD setup | Prefill/decode disaggregation |

Use the same prompts and `max_new_tokens` where model support allows (document model id
and quantizations in `meta.model` / `systems[].description`).

### End-to-end workflow

1. **Setup:** `python3 -m venv .venv && source .venv/bin/activate` — install
   `torch`, `transformers`, and whatever your engines need (`vllm`, etc.).
2. **Copy baseline** into `code/minimal_full_server.py` (from Week 2).
3. **Scaffold JSON:** `python3 code/export_midterm_results.py --scaffold <UNI>`
4. **Point report at your file:** edit `report/src/results.js` →
   `import DATA from './data/<UNI>-midterm-results.json'`
5. **Run sweep** (see matrix below) — append one object per repetition to `runs[]`.
6. **Fill** `meta`, `systems[]`, `takeaways`, `summary`, `plot`, `boxplot`.
7. **Validate:** `python3 code/export_midterm_results.py report/src/data/<UNI>-midterm-results.json`
8. **Preview site:** `npm run setup && npm run dev` — edit Discussion in `App.jsx`.
9. **Push `main`** — confirm GitHub Pages workflow is green.

### Measurement matrix (minimum)

For **each** `system_id` you claim in JSON:

| Dimension | Minimum |
|-----------|---------|
| Prompt lengths (tokens after chat template) | 3 (e.g. 128, 512, 2048) |
| Concurrency | 2 (e.g. 1 and 4) |
| Repetitions per (system, prompt, concurrency) | 3 after warmup |
| Metrics per run | `ttft_ms`, `tbt_ms_p50`, `tbt_ms_mean`, `kv_bytes`, `kv_seq_len`, `new_tokens` |
| Extra when concurrency &gt; 1 | `ttft_ms_samples`, `tbt_ms_samples` (length = N) |

That is on the order of **3 × 2 × 3 = 18 rows per system** (more if you add reps or prompts).

## Prompts (WildChat-1M + HW1-style synthetic)

Use **real** and **controlled** prompts so systems are comparable:

1. **[WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M)** — sample user turns
   from this dataset for realistic chat lengths and phrasing. Record which split/rows
   or seed you used in `source.workload.notes`. (Smaller sibling:
   [allenai/WildChat](https://huggingface.co/datasets/allenai/WildChat).)
2. **HW1-style synthetic length targets** — same idea as Homework 1’s evaluation sweep:
   build or reuse **fixed prompts** trimmed/padded to **≥ 3 token lengths** after your
   model’s chat template (e.g. 128 / 512 / 2048). Reuse prompts from your HW1 runs; 
   otherwise document the generator (seed text, repeat/truncate
   policy) in `source.harness`.

**Mix:** e.g. synthetic prompts for the full prompt-length grid (apples-to-apples across
engines) plus a subset of WildChat-1M turns for one column of sanity checks. Every row in
`runs[]` must record the measured **`prompt_tokens`** (after templating).

## Required metrics

| Metric | Meaning | How to measure |
|--------|---------|----------------|
| **TTFT** | Time to first token | Wall time from **request start** (prompt accepted) to **first generated token** (include queue/scheduling/KV fetch if present). Use `ttft_ms()` in `code/export_midterm_results.py` or equivalent. |
| **TBT** | Time between tokens | Per **decode step** after prefill; report **p50** and **mean** (`tbt_ms_p50`, `tbt_ms_mean` via `tbt_ms_per_step()`). Under concurrency, report **per-user** TBT, not only system throughput. |
| **KV bytes** | KV cache footprint | Use `kv_bytes()` from HW1 / `less_minimal_full_server.py` when the cache is dense on GPU, **or** engine-reported allocated KV bytes at end of run. State which you used. |

### Concurrency: TTFT and TBT are distributions

At **concurrency = 1**, a single TTFT and per-step TBT trace per run is enough.
When **N concurrent clients** share a GPU (Orca/vLLM batching, PD pools, Mooncake
Conductor admits), each user gets its **own** TTFT and decode experience.
Per-client samples let you report medians and tails (p95/max) separately — the
fairness view from lecture TTFT boxplots.

For every run with `concurrency > 1` you must record **per-client samples**:

| JSON field | Contents |
|------------|----------|
| `ttft_ms_samples` | Length **N** (one TTFT per client in the burst) |
| `tbt_ms_samples` | Length **N** (one TBT summary per client — e.g. median ms between decode tokens for that request) |
| `kv_bytes_samples` | Length **N** (optional in JSON; use for boxplot **KV bytes** — per-client allocated KV or dense `kv_bytes()` at end of prefill/decode for that client) |

Also keep scalar `ttft_ms`, `tbt_ms_p50`, `tbt_ms_mean` on each row as the
**aggregate** across clients (median/mean of the samples is fine). In your writeup,
call out **p95** or **max** when tails differ across engines, not only medians.

Optional context (not a substitute for the three metrics): output quality spot-check,
prefix hit rate on Mooncake runs, prefill vs decode phase times.

## Setup

**Python 3.10+** for measurements and JSON export. **Node 20.19+ or 22.12+** for the report site (Vite 8 requirement).

```bash
# Measurements (baseline server — copy from course week-2 repo or your HW tree)
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch transformers  # plus vLLM / server deps for your chosen engines

# Report (from this repo root — either path works)
npm run setup && npm run dev
# or: cd report && npm install && npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`).

`npm run build` at the **repository root** delegates to `report/` (same as CI).
Install dependencies once with `npm run setup` or `cd report && npm ci`.

## Task (detailed)

1. **Run** the measurement matrix for **each** system listed in `systems[]` (minimum 3).
2. **Record** every repetition as one object in `runs[]` (hand-edited JSON is OK if your
   harness prints rows you paste in).
3. **Write** takeaways, summary, and Discussion — **for each `system_id`**, tie
   results to that system’s **`novelty`** (scheduling, paging, prefix reuse, PD, etc.)
   and to **tail latency** under N&gt;1.
4. **Configure both required figures:** `plot` (line chart) and `boxplot`
   (distribution; fixed prompt via `boxplot.filter`, concurrency via report UI).

Recommended **`code/` layout** (names flexible; list real paths in `source.harness`):

```text
code/
  minimal_full_server.py      # copied from Week 2; add timing if needed
  run_midterm_sweep.py        # your driver: loops systems × prompts × N × reps
  export_midterm_results.py   # provided — validate JSON; import helpers from here
```

Your driver should call the same helper names (`ttft_ms`, `tbt_ms_per_step`, `kv_bytes`)
or document equivalents in `source.workload.notes`.

## Results JSON (required)

Pattern matches inference-bench catalog pages: data lives under
`report/src/data/*.json` and is **imported at Vite build time** (bundled into JS).
GitHub Pages does **not** fetch JSON at runtime — **commit** your file.

| File | Purpose |
|------|---------|
| `report/src/data/midterm-results.sample.json` | **Template only** — do not submit with your UNI |
| `report/src/data/<UNI>-midterm-results.json` | **Your measurements** (replace `abc1234` with Columbia UNI) |

Wire your file in **one place**:

```js
// report/src/results.js
import DATA from './data/abc1234-midterm-results.json'
export default DATA
```

### Top-level schema

- `meta` — `uni`, `name`, `github`, `date`, `title`, `model`, `device`
- `source` — harness name, `json_file` path, timestamps, `workload` grid
- `systems[]` — `id`, `label`, `family`, `description`, **`novelty`**, optional **`repo`**
  (HTTPS GitHub URL for upstream code you ran — omit for baseline course file only;
  shown as a link on the report page)
- `runs[]` — one object per measured repetition (see required keys below)
- `takeaways` — **≥ 3** short bullets (shown at top of page)
- `summary` — one paragraph (shown under takeaways)
- `plot` — line chart (`title`, `metric`, `x_field`, `series_field`, `filter`, axis labels)
- `boxplot` — **required** (`title`, `metric`, `filter` e.g. `prompt_tokens`, optional
  `default_concurrency`, `group_field`, optional `subtitle`, `samples_field`, `metrics[]`).
  **Concurrency** is selected on the report page (from `source.workload.concurrency_grid`).

Required keys on each `runs[]` row:

`system_id`, `prompt_tokens`, `concurrency`, `rep`, `ttft_ms`, `tbt_ms_p50`,
`tbt_ms_mean`, `kv_bytes`, `kv_seq_len`, `new_tokens`

Required when `concurrency > 1` on that row:

`ttft_ms_samples`, `tbt_ms_samples` (each a JSON array of numbers, length = concurrency).
Recommended for boxplot KV view: `kv_bytes_samples` (same length **N**).

Optional: `prefix_reuse` (`"cold"` | `"warm"`), `notes`.

Example concurrent row:

```json
{
  "system_id": "vllm_paged",
  "prompt_tokens": 512,
  "concurrency": 4,
  "rep": 0,
  "ttft_ms": 301.5,
  "ttft_ms_samples": [182, 198, 405, 892],
  "tbt_ms_p50": 17.15,
  "tbt_ms_mean": 23.93,
  "tbt_ms_samples": [13.8, 15.1, 19.2, 47.6],
  "kv_bytes": 12976128,
  "kv_seq_len": 576,
  "new_tokens": 64
}
```

Example `boxplot` block (see `midterm-results.sample.json`):

```json
"boxplot": {
  "title": "TTFT distribution per client (512-token prompt)",
  "metric": "ttft_ms",
  "samples_field": "ttft_ms_samples",
  "group_field": "system_id",
  "default_concurrency": 4,
  "filter": { "prompt_tokens": 512 },
  "y_label": "TTFT (ms)"
}
```

Validate before you push (paths resolve from the repo root, so this works from any folder).
Schema problems fail validation; missing cells in the measurement matrix only print a warning:

```bash
python3 code/export_midterm_results.py report/src/data/abc1234-midterm-results.json
```

Scaffold an empty runs file:

```bash
python3 code/export_midterm_results.py --scaffold abc1234
```

### Python helpers (function names to use in code or cite in methods)

In `code/export_midterm_results.py`:

- `kv_seq_len(kv_cache)`
- `kv_bytes(n_layers, n_kv_heads, seq_len, head_dim, batch=1, itemsize=2)`
- `ttft_ms(t_request_start, t_first_token)`
- `tbt_ms_per_step(decode_step_durations_s)` → `tbt_ms_p50`, `tbt_ms_mean`
- `summarize_client_samples(samples_ms)` → `p50`, `p95`, `max`, `mean` for JSON QA
- `percentile_ms(samples_ms, p)` — tail checks in your harness

You may instrument `minimal_full_server.py` directly (prefill boundary → first token,
per-step decode timers). For vLLM, use server/client timestamps or engine metrics;
document the hook in **Measurement setup**. For **N concurrent clients**, launch a
barrier-style burst (all start together), record **one TTFT per client**, and store
them in `ttft_ms_samples` before you collapse to `ttft_ms`.

## Report template (React)

Pre-built layout in `report/src/App.jsx`:

1. **Takeaways** — from JSON `takeaways[]` (not hideable)
2. **Summary** — from JSON `summary`
3. **Results** — hideable; **two required figures:** (1) line chart `MetricsPlot`
   ← JSON `plot`; (2) boxplot `DistributionBoxplot` ← JSON `boxplot`
4. **Systems compared** — hideable; from JSON `systems[]`
5. **Measurement setup** — hideable; from JSON `source`
6. **Discussion** — hideable; **you** replace placeholder text in `App.jsx` (or extend JSON + map it)
7. **Raw run count** — hideable sanity check

Reusable UI: `report/src/Hideable.jsx` (collapsible sections).

**Do not** remove hideable sections — TAs use them for rubric spot-checks. You may
add subsections inside **Discussion**.

Local preview:

```bash
npm run setup && npm run dev
```

Production build (same as CI):

```bash
npm run setup && npm run build && npm run preview
```

## GitHub Pages

This repo includes `.github/workflows/pages.yml`.

1. Commit `report/src/data/<UNI>-midterm-results.json`, updated `report/src/results.js`, and `report/src/App.jsx` (discussion).
2. Push to `main`.
3. Wait for the **Deploy React App to GitHub Pages** workflow.
4. **Settings → Pages → Build and deployment → Source: GitHub Actions**.
5. Open the published URL (collaborators only on a private repo).

Put the Pages URL in `meta.repo_url` or in your Discussion section.

## Deliverable checklist

Submit on `main` before the due date posted on Ed:

- [ ] **Ran** at least **3** systems (baseline + 2); more included in JSON/plots if benchmarked
- [ ] **`code/`** contains baseline + driver or README describing exact commands you used
- [ ] `report/src/data/<UNI>-midterm-results.json` validates with `export_midterm_results.py`
- [ ] `report/src/results.js` imports **your** JSON (not the sample)
- [ ] `meta.uni`, name, GitHub handle, date filled in JSON
- [ ] **≥ 3** rows in `systems[]` (baseline + 2 minimum); full sweep ≫ 6 rows in `runs[]`
- [ ] `takeaways` (≥ 3) + `summary` written for **your** results
- [ ] **`plot`** line chart configured (TTFT, TBT, or KV bytes — not tok/s alone)
- [ ] **`boxplot`** configured + `ttft_ms_samples` / `tbt_ms_samples` on concurrent runs
- [ ] Every `systems[]` entry has **`novelty`**; upstream systems include **`repo`** URL
- [ ] Discussion references each system’s novelty
- [ ] Discussion mentions **tail** latency (p95/max or box whiskers), not only medians
- [ ] Discussion ties numbers to **that system’s novelty**, not generic tok/s
- [ ] GitHub Pages builds green; URL works
- [ ] Scripts or notes under `code/` showing how you produced `runs[]`

## Reporting logistics (unchanged)

Focus on content; the Vite app and Pages workflow are provided. Edit
`report/src/App.jsx`, add components under `report/src/`, push to `main`.

Good luck!
