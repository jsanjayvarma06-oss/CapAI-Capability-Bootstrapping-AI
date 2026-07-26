#!/usr/bin/env python3
"""
CapAI Phase 2 — Automated Benchmark Runner
==========================================
Runs all 170 tasks against the live CapAI server and saves results to CSV.

Usage:
    python run_benchmark.py                        # full run (all 170 tasks)
    python run_benchmark.py --workflow run         # only /run tasks
    python run_benchmark.py --workflow build       # only /build tasks
    python run_benchmark.py --category "String and Text Processing"
    python run_benchmark.py --difficulty Hard
    python run_benchmark.py --dry-run              # validate YAML, no HTTP calls
    python run_benchmark.py --warm-trials 5        # override warm trial count
    python run_benchmark.py --skip-baseline        # skip uncached baseline
"""

import argparse
import csv
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml
import requests

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

CAPAI_BASE_URL = os.environ.get(
    "CAPAI_URL",
    "https://capai-capability-bootstrapping-ai-fu58.onrender.com"
).rstrip("/")

BENCHMARK_FILE = Path(__file__).parent / "benchmark_tasks.yaml"
OUTPUT_DIR = Path(__file__).parent / "results"

N_WARM_TRIALS    = 10   # repeated warm-cache calls per task
N_BUILD_TRIALS   = 5    # independent /build trials per task
INTER_TRIAL_DELAY = 3.0 # seconds between calls (avoid rate limits)
REQUEST_TIMEOUT  = 120  # seconds per HTTP request
MAX_RETRIES      = 3
RETRY_BACKOFF    = [5, 15, 45]  # seconds

# ──────────────────────────────────────────────────────────────────────────────
# Output schema
# ──────────────────────────────────────────────────────────────────────────────

FIELDNAMES = [
    "task_id", "category", "difficulty", "workflow", "condition",
    "trial_index", "latency_ms", "correct", "confidence",
    "coverage_pct", "repair_iterations", "cache_status",
    "provider_used", "error_type", "http_status", "timestamp"
]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def approx_equal(a, b, tol=0.01) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def sets_equal(a, b) -> bool:
    try:
        return set(map(str, a)) == set(map(str, b))
    except Exception:
        return False


def anagram_groups_equal(actual, expected) -> bool:
    """Each group must match as a set; the collection of groups must match."""
    try:
        actual_sets   = [frozenset(g) for g in actual]
        expected_sets = [frozenset(g) for g in expected]
        return set(actual_sets) == set(expected_sets)
    except Exception:
        return False


def triplets_equal(actual, expected) -> bool:
    try:
        actual_s   = {frozenset(t) for t in actual}
        expected_s = {frozenset(t) for t in expected}
        return actual_s == expected_s
    except Exception:
        return False


def bytes_equal(actual, expected) -> bool:
    """Accept list of ints or dict with __bytes__ key."""
    def to_list(v):
        if isinstance(v, dict) and "__bytes__" in v:
            return v["__bytes__"]
        if isinstance(v, (list, bytes)):
            return list(v)
        return None
    a, e = to_list(actual), to_list(expected)
    return a is not None and a == e


def verify_output(task: dict, actual) -> bool:
    check = task.get("check", "exact")
    expected = task.get("expected")

    if check == "exact":
        return actual == expected
    elif check == "exact_case_insensitive":
        return str(actual).lower().strip() == str(expected).lower().strip()
    elif check == "approx":
        return approx_equal(actual, expected)
    elif check == "set_equal":
        return sets_equal(actual, expected)
    elif check == "anagram_groups":
        return anagram_groups_equal(actual, expected)
    elif check == "triplet_equal":
        return triplets_equal(actual, expected)
    elif check == "bytes_equal":
        return bytes_equal(actual, expected)
    elif check == "build_success":
        # For /build tasks: success means the API returned a success flag
        if isinstance(actual, dict):
            return actual.get("success", False) or actual.get("status") == "success"
        return bool(actual)
    return False


def classify_error(response_json: dict, correct: bool) -> str:
    if correct:
        return ""
    error = str(response_json.get("error", "")).lower()
    result = str(response_json.get("result", "")).lower()
    combined = error + " " + result
    if "typeerror" in combined or "type error" in combined:
        return "type_error"
    if "importerror" in combined or "modulenotfounderror" in combined:
        return "import_error"
    if "nameerror" in combined or "attributeerror" in combined:
        return "hallucinated_api"
    if "timeout" in combined or "timed out" in combined:
        return "timeout"
    if response_json.get("result") is None and not correct:
        return "logic_error"
    return "edge_case_miss"


# ──────────────────────────────────────────────────────────────────────────────
# Registry management
# ──────────────────────────────────────────────────────────────────────────────

def clear_task_from_registry(task_id: str):
    """Best-effort: try to remove this capability so the next call is cold."""
    try:
        # CapAI has no per-key delete endpoint in the public API.
        # We rely on the /reset endpoint in dry-run mode only.
        # In production runs, cold trials are measured on first-ever call
        # or after a /reset/hard. We skip per-task reset to preserve other caps.
        pass
    except Exception:
        pass


def server_health() -> dict:
    try:
        r = requests.get(f"{CAPAI_BASE_URL}/health", timeout=15)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ──────────────────────────────────────────────────────────────────────────────
# Core HTTP call with retry
# ──────────────────────────────────────────────────────────────────────────────

def call_capai(endpoint: str, payload: dict) -> tuple[dict, float, int]:
    """
    Returns (response_json, latency_ms, http_status).
    Retries on connection errors and 429/503.
    """
    url = f"{CAPAI_BASE_URL}/{endpoint.lstrip('/')}"
    last_exc = None
    last_status = 0

    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.perf_counter()
            resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
            latency_ms = (time.perf_counter() - t0) * 1000
            last_status = resp.status_code

            if resp.status_code in (429, 503) and attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF[attempt]
                log(f"    Rate-limited ({resp.status_code}), retrying in {wait}s …")
                time.sleep(wait)
                continue

            try:
                return resp.json(), latency_ms, resp.status_code
            except Exception:
                return {"error": "non-json response", "raw": resp.text[:200]}, latency_ms, resp.status_code

        except requests.exceptions.Timeout:
            last_exc = "timeout"
            latency_ms = REQUEST_TIMEOUT * 1000
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF[attempt])

        except requests.exceptions.ConnectionError as e:
            last_exc = str(e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF[attempt])

    return {"error": last_exc or "connection_error"}, latency_ms, last_status


# ──────────────────────────────────────────────────────────────────────────────
# Build the request payload
# ──────────────────────────────────────────────────────────────────────────────

def make_run_payload(task: dict) -> dict:
    return {
        "name":        task["id"],
        "description": task["description"],
        "args":        task.get("args", []),
    }


def make_build_payload(task: dict) -> dict:
    return {
        "name":        task["id"],
        "description": task["description"],
        "use_cache":   False,          # each build trial is independent
    }


# ──────────────────────────────────────────────────────────────────────────────
# Single trial runners
# ──────────────────────────────────────────────────────────────────────────────

def run_trial(task: dict, condition: str, trial_index: int) -> dict:
    """Execute one /run trial and return a result record."""
    endpoint = "run"
    payload = make_run_payload(task)

    resp_json, latency_ms, http_status = call_capai(endpoint, payload)

    actual_result = resp_json.get("result")
    correct = verify_output(task, actual_result)

    return {
        "task_id":          task["id"],
        "category":         task["category"],
        "difficulty":       task["difficulty"],
        "workflow":         "run",
        "condition":        condition,
        "trial_index":      trial_index,
        "latency_ms":       round(latency_ms, 2),
        "correct":          correct,
        "confidence":       resp_json.get("confidence", ""),
        "coverage_pct":     resp_json.get("coverage", ""),
        "repair_iterations":resp_json.get("repair_iterations", 0),
        "cache_status":     resp_json.get("cache_status", "miss"),
        "provider_used":    resp_json.get("provider", "unknown"),
        "error_type":       classify_error(resp_json, correct),
        "http_status":      http_status,
        "timestamp":        now_iso(),
    }


def run_build_trial(task: dict, trial_index: int) -> dict:
    """Execute one independent /build trial and return a result record."""
    endpoint = "build"
    payload = make_build_payload(task)

    resp_json, latency_ms, http_status = call_capai(endpoint, payload)

    # For build tasks, 'correct' == the build succeeded
    correct = verify_output(task, resp_json)

    return {
        "task_id":          task["id"],
        "category":         task["category"],
        "difficulty":       task["difficulty"],
        "workflow":         "build",
        "condition":        "build_independent",
        "trial_index":      trial_index,
        "latency_ms":       round(latency_ms, 2),
        "correct":          correct,
        "confidence":       resp_json.get("confidence", ""),
        "coverage_pct":     resp_json.get("coverage", ""),
        "repair_iterations":resp_json.get("repair_iterations", 0),
        "cache_status":     resp_json.get("cache_status", "miss"),
        "provider_used":    resp_json.get("provider", "unknown"),
        "error_type":       classify_error(resp_json, correct) if not correct else "",
        "http_status":      http_status,
        "timestamp":        now_iso(),
    }


def run_baseline_trial(task: dict, trial_index: int) -> dict:
    """
    Baseline: call /run with use_cache=False so CapAI bypasses the registry
    and goes straight to LLM synthesis. This isolates LLM quality from caching.
    """
    endpoint = "run"
    payload = make_run_payload(task)
    payload["use_cache"] = False   # bypass registry

    resp_json, latency_ms, http_status = call_capai(endpoint, payload)

    actual_result = resp_json.get("result")
    correct = verify_output(task, actual_result)

    return {
        "task_id":          task["id"],
        "category":         task["category"],
        "difficulty":       task["difficulty"],
        "workflow":         "run",
        "condition":        "baseline",
        "trial_index":      trial_index,
        "latency_ms":       round(latency_ms, 2),
        "correct":          correct,
        "confidence":       "",
        "coverage_pct":     "",
        "repair_iterations":0,
        "cache_status":     "N/A",
        "provider_used":    resp_json.get("provider", "unknown"),
        "error_type":       classify_error(resp_json, correct),
        "http_status":      http_status,
        "timestamp":        now_iso(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Full task benchmark
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_run_task(task: dict, writer, args) -> list[dict]:
    records = []
    tid = task["id"]

    # ── 1. Cold synthesis (first ever call — no registry entry yet) ──
    log(f"  [{tid}] cold …")
    rec = run_trial(task, condition="cold", trial_index=0)
    records.append(rec)
    writer.writerow(rec)
    time.sleep(INTER_TRIAL_DELAY)

    # ── 2. Warm cache trials ──
    n_warm = args.warm_trials
    for i in range(n_warm):
        log(f"  [{tid}] warm {i+1}/{n_warm} …")
        rec = run_trial(task, condition="warm", trial_index=i)
        records.append(rec)
        writer.writerow(rec)
        time.sleep(INTER_TRIAL_DELAY)

    # ── 3. Uncached baseline trials ──
    if not args.skip_baseline:
        n_base = args.warm_trials
        for i in range(n_base):
            log(f"  [{tid}] baseline {i+1}/{n_base} …")
            rec = run_baseline_trial(task, trial_index=i)
            records.append(rec)
            writer.writerow(rec)
            time.sleep(INTER_TRIAL_DELAY)

    return records


def benchmark_build_task(task: dict, writer, args) -> list[dict]:
    records = []
    tid = task["id"]
    n_trials = args.build_trials

    for i in range(n_trials):
        log(f"  [{tid}] build trial {i+1}/{n_trials} …")
        rec = run_build_trial(task, trial_index=i)
        records.append(rec)
        writer.writerow(rec)
        time.sleep(INTER_TRIAL_DELAY)

    return records


# ──────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(records: list[dict]):
    if not records:
        print("\nNo records collected.")
        return

    run_records   = [r for r in records if r["workflow"] == "run"]
    build_records = [r for r in records if r["workflow"] == "build"]

    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)

    for cond in ["cold", "warm", "baseline"]:
        subset = [r for r in run_records if r["condition"] == cond]
        if not subset:
            continue
        latencies = [r["latency_ms"] for r in subset]
        correct   = [r for r in subset if r["correct"]]
        mean_lat  = sum(latencies) / len(latencies)
        med_lat   = sorted(latencies)[len(latencies)//2]
        corr_pct  = 100 * len(correct) / len(subset)
        print(f"\n/run — {cond.upper()} (n={len(subset)})")
        print(f"  Mean latency : {mean_lat:,.1f} ms")
        print(f"  Median latency: {med_lat:,.1f} ms")
        print(f"  Correctness  : {corr_pct:.1f}%")

    if build_records:
        correct_build = [r for r in build_records if r["correct"]]
        rep_iters = [int(r["repair_iterations"] or 0) for r in build_records]
        covs = [float(r["coverage_pct"]) for r in build_records if r["coverage_pct"] not in ("", None)]
        confs = [float(r["confidence"]) for r in build_records if r["confidence"] not in ("", None)]
        mean_rep = sum(rep_iters)/len(rep_iters) if rep_iters else 0
        mean_cov = sum(covs)/len(covs) if covs else 0
        mean_conf = sum(confs)/len(confs) if confs else 0
        success_pct = 100 * len(correct_build) / len(build_records)
        print(f"\n/build — INDEPENDENT TRIALS (n={len(build_records)})")
        print(f"  Success rate  : {success_pct:.1f}%")
        print(f"  Mean repair iter: {mean_rep:.2f}")
        print(f"  Mean coverage : {mean_cov:.1f}%")
        print(f"  Mean confidence: {mean_conf:.1f}")

    # Cache speedup
    cold_recs = [r for r in run_records if r["condition"] == "cold"]
    warm_recs = [r for r in run_records if r["condition"] == "warm"]
    if cold_recs and warm_recs:
        mean_cold = sum(r["latency_ms"] for r in cold_recs) / len(cold_recs)
        mean_warm = sum(r["latency_ms"] for r in warm_recs) / len(warm_recs)
        speedup = mean_cold / mean_warm if mean_warm > 0 else 0
        print(f"\nCache speedup: {speedup:.2f}x (cold {mean_cold:.0f} ms → warm {mean_warm:.0f} ms)")

    # Error breakdown
    errors = [r["error_type"] for r in records if r["error_type"]]
    if errors:
        from collections import Counter
        print("\nError type breakdown:")
        for etype, cnt in Counter(errors).most_common():
            print(f"  {etype}: {cnt}")

    print("="*60)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CapAI Phase 2 Benchmark Runner")
    p.add_argument("--workflow",    choices=["run", "build"], default=None,
                   help="Run only tasks of this workflow type")
    p.add_argument("--category",    default=None,
                   help="Run only tasks in this category (substring match)")
    p.add_argument("--difficulty",  choices=["Easy", "Medium", "Hard"], default=None,
                   help="Run only tasks of this difficulty")
    p.add_argument("--task-id",     default=None,
                   help="Run only the task with this ID")
    p.add_argument("--dry-run",     action="store_true",
                   help="Validate YAML and print task list without calling the server")
    p.add_argument("--warm-trials", type=int, default=N_WARM_TRIALS,
                   help=f"Number of warm-cache trials per /run task (default {N_WARM_TRIALS})")
    p.add_argument("--build-trials",type=int, default=N_BUILD_TRIALS,
                   help=f"Number of independent /build trials (default {N_BUILD_TRIALS})")
    p.add_argument("--skip-baseline",action="store_true",
                   help="Skip the uncached baseline condition")
    p.add_argument("--output",      default=None,
                   help="Override output CSV file path")
    p.add_argument("--delay",       type=float, default=INTER_TRIAL_DELAY,
                   help=f"Seconds between calls (default {INTER_TRIAL_DELAY})")
    p.add_argument("--skip-tasks",  default=None,
                   help="Comma-separated list of task IDs to skip")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load benchmark ──
    if not BENCHMARK_FILE.exists():
        print(f"ERROR: benchmark file not found: {BENCHMARK_FILE}")
        sys.exit(1)

    with open(BENCHMARK_FILE, "r") as f:
        data = yaml.safe_load(f)

    all_tasks = data["tasks"]

    # ── Filter ──
    tasks = all_tasks
    if args.workflow:
        tasks = [t for t in tasks if t["workflow"] == args.workflow]
    if args.category:
        tasks = [t for t in tasks if args.category.lower() in t["category"].lower()]
    if args.difficulty:
        tasks = [t for t in tasks if t["difficulty"] == args.difficulty]
    if args.task_id:
        tasks = [t for t in tasks if t["id"] == args.task_id]
    if args.skip_tasks:
        skip_set = set(s.strip() for s in args.skip_tasks.split(","))
        tasks = [t for t in tasks if t["id"] not in skip_set]

    if not tasks:
        print("No tasks matched the given filters.")
        sys.exit(1)

    run_tasks   = [t for t in tasks if t["workflow"] == "run"]
    build_tasks = [t for t in tasks if t["workflow"] == "build"]

    # ── Estimate total calls ──
    n_run_calls = len(run_tasks) * (1 + args.warm_trials + (0 if args.skip_baseline else args.warm_trials))
    n_build_calls = len(build_tasks) * args.build_trials
    total_calls = n_run_calls + n_build_calls
    est_minutes = (total_calls * (INTER_TRIAL_DELAY + 5)) / 60

    log(f"CapAI Phase 2 Benchmark")
    log(f"Server : {CAPAI_BASE_URL}")
    log(f"Tasks  : {len(tasks)} ({len(run_tasks)} /run, {len(build_tasks)} /build)")
    log(f"Est. calls: {total_calls}  |  Est. time: ~{est_minutes:.0f} minutes")

    if args.dry_run:
        log("DRY RUN — printing task list only")
        for t in tasks:
            print(f"  [{t['workflow']:5}] [{t['difficulty']:6}] {t['id']}")
        print(f"\nTotal: {len(tasks)} tasks")
        return

    # ── Health check ──
    log("Checking server health …")
    health = server_health()
    if "error" in health:
        log(f"WARNING: health check failed: {health['error']}")
        log("Continuing anyway — server may still be up.")
    else:
        log(f"Server healthy: {json.dumps(health)}")

    # ── Prepare output ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = OUTPUT_DIR / f"capai_benchmark_{ts}.csv"

    log(f"Writing results to: {out_path}")

    all_records = []
    total_done = 0

    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()
        csvfile.flush()

        # ── /run tasks ──
        for i, task in enumerate(run_tasks):
            log(f"\n[{i+1}/{len(run_tasks)}] /run task: {task['id']} [{task['difficulty']}]")
            try:
                recs = benchmark_run_task(task, writer, args)
                all_records.extend(recs)
                csvfile.flush()
                total_done += 1
            except KeyboardInterrupt:
                log("Interrupted by user. Saving progress …")
                break
            except Exception as e:
                log(f"  ERROR on task {task['id']}: {e}")
                traceback.print_exc()
                # Write a failure record and continue
                fail_rec = {
                    "task_id": task["id"], "category": task["category"],
                    "difficulty": task["difficulty"], "workflow": "run",
                    "condition": "error", "trial_index": 0,
                    "latency_ms": 0, "correct": False, "confidence": "",
                    "coverage_pct": "", "repair_iterations": 0,
                    "cache_status": "", "provider_used": "",
                    "error_type": "runner_exception",
                    "http_status": 0, "timestamp": now_iso()
                }
                writer.writerow(fail_rec)
                csvfile.flush()

        # ── /build tasks ──
        for i, task in enumerate(build_tasks):
            log(f"\n[{i+1}/{len(build_tasks)}] /build task: {task['id']} [{task['difficulty']}]")
            try:
                recs = benchmark_build_task(task, writer, args)
                all_records.extend(recs)
                csvfile.flush()
                total_done += 1
            except KeyboardInterrupt:
                log("Interrupted by user. Saving progress …")
                break
            except Exception as e:
                log(f"  ERROR on task {task['id']}: {e}")
                traceback.print_exc()

    log(f"\nDone. {total_done}/{len(tasks)} tasks completed.")
    log(f"Results saved to: {out_path}")

    print_summary(all_records)


if __name__ == "__main__":
    main()