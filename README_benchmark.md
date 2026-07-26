# CapAI Phase 2 — Benchmark Runner

## Files

| File | Purpose |
|---|---|
| `benchmark_tasks.yaml` | 170 tasks across 11 categories |
| `run_benchmark.py` | Hits CapAI server, saves results to CSV |
| `analyze_results.py` | Reads CSV, prints all paper statistics |
| `requirements_benchmark.txt` | Python dependencies |

---

## Setup (one time)

```bash
pip install requests pyyaml
```

---

## Step 1 — Verify the server is up

```bash
curl https://capai-capability-bootstrapping-ai-fu58.onrender.com/health
```

---

## Step 2 — Run the full benchmark

```bash
python run_benchmark.py
```

Results are saved to `results/capai_benchmark_YYYYMMDD_HHMMSS.csv`.
The script prints a live summary as it runs and a full summary at the end.

**To run only /run tasks (faster, ~120 min):**
```bash
python run_benchmark.py --workflow run
```

**To run only /build tasks:**
```bash
python run_benchmark.py --workflow build
```

**To test a single task first:**
```bash
python run_benchmark.py --task-id is_prime
```

**To do a dry run (validate YAML, no HTTP calls):**
```bash
python run_benchmark.py --dry-run
```

**To reduce warm trials for a faster run:**
```bash
python run_benchmark.py --warm-trials 3 --build-trials 3
```

**To skip the baseline condition:**
```bash
python run_benchmark.py --skip-baseline
```

---

## Step 3 — Analyze results

```bash
python analyze_results.py results/capai_benchmark_YYYYMMDD_HHMMSS.csv
```

This prints:
- Table II (latency + correctness with CIs)
- Table III (build outcomes)
- Per-category correctness breakdown
- Per-difficulty breakdown
- Error taxonomy
- Confidence score correlation
- Cache hit rate breakdown
- Provider usage

---

## Estimated run times

| Mode | Tasks | Calls | Est. time |
|---|---|---|---|
| Full (all 170) | 170 | ~2,600 | ~5–6 hours |
| /run only (120 tasks) | 120 | ~2,520 | ~4–5 hours |
| /build only (50 tasks) | 50 | ~250 | ~30–60 min |
| Quick (3 warm, 2 build) | 170 | ~780 | ~90 min |
| Single task test | 1 | ~21 | ~3 min |

The delay between calls (`--delay`, default 3s) is set conservatively to
avoid Groq rate limits. If your Groq quota is high, you can reduce it:

```bash
python run_benchmark.py --delay 1.5
```

---

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `CAPAI_URL` | `https://capai-...onrender.com` | Override server URL |

```bash
# Example: run against local dev server
export CAPAI_URL=http://localhost:8000
python run_benchmark.py --workflow run --warm-trials 2
```

---

## CSV schema

| Column | Type | Description |
|---|---|---|
| task_id | str | Function/class identifier |
| category | str | Category name |
| difficulty | str | Easy / Medium / Hard |
| workflow | str | run / build |
| condition | str | cold / warm / baseline / build_independent |
| trial_index | int | 0-indexed trial |
| latency_ms | float | Wall-clock ms |
| correct | bool | Oracle verified |
| confidence | float | CapAI confidence score (0–100) |
| coverage_pct | float | Statement coverage % |
| repair_iterations | int | Repair loop iterations used |
| cache_status | str | hit / miss / heuristic / N/A |
| provider_used | str | LLM provider name |
| error_type | str | type_error / logic_error / etc. |
| http_status | int | HTTP response code |
| timestamp | ISO-8601 | UTC time of trial |

---

## Troubleshooting

**Server returns 404 on /run or /build:**
The Render free tier spins down after inactivity. Hit `/health` first and
wait 30s for cold start, then re-run.

**Groq 429 rate limit:**
The runner retries automatically with backoff (5s → 15s → 45s).
If you're hitting limits consistently, increase `--delay` to 5 or 10 seconds.

**Task fails with `runner_exception`:**
Check the traceback printed to stdout. Usually a network timeout.
The CSV is flushed after every trial so partial results are always saved.

**`use_cache=False` not supported:**
Some CapAI deployments may not support the `use_cache` flag on /run.
If baseline trials are returning cached results, use `--skip-baseline`
and run the baseline condition manually after a `/reset` call.
