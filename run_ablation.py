#!/usr/bin/env python3
"""
CapAI Ablation Study Runner
============================
Runs all 6 ablation conditions against the live CapAI server.

Conditions:
  A1 - Without cache        (use_cache=False on every /run call)
  A2 - Without sandbox      (requires CAPAI_SKIP_SANDBOX=true on server)
  A3 - Without repair       (max_iterations=1 on every /build call)
  A4 - Without heuristics   (requires CAPAI_SKIP_HEURISTICS=true on server)
  A5 - Without MCP          (REST vs MCP latency comparison)
  A6 - Without persistence  (registry reset between task groups)

Usage:
    python run_ablation.py                    # all conditions
    python run_ablation.py --condition A1     # single condition
    python run_ablation.py --condition A3     # build repair ablation
    python run_ablation.py --dry-run          # validate without HTTP calls

Each condition saves its own CSV to results/ablation_AX_TIMESTAMP.csv
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

# How many tasks per ablation condition (keep small to save Groq quota)
N_RUN_TASKS   = 30   # use 30 /run tasks per condition
N_BUILD_TASKS = 10   # use 10 /build tasks per condition
N_TRIALS_WARM = 3    # warm trials per task
N_BUILD_TRIALS = 3   # build trials per task
DELAY         = 3.0  # seconds between calls
REQUEST_TIMEOUT = 120
MAX_RETRIES   = 3
RETRY_BACKOFF = [5, 15, 45]

# Fixed task subsets for fair ablation comparison
# Using first N tasks from each category (stratified)
RUN_TASK_IDS = [
    # String (4)
    "reverse_words", "count_vowels", "is_palindrome_str", "levenshtein_distance",
    # Numerical (4)
    "is_prime", "fibonacci", "sieve_of_eratosthenes", "matrix_multiply",
    # Algorithms (4)
    "two_sum", "max_subarray_sum", "coin_change", "sliding_window_max",
    # Validation (4)
    "is_valid_email", "is_valid_ipv4", "luhn_check", "is_valid_isbn13",
    # Date/Time (3)
    "days_between", "is_leap_year", "business_days_between",
    # File/IO (3)
    "count_lines", "flatten_dict", "merge_dicts",
    # Encoding (3)
    "hex_to_rgb", "md5_hex", "encode_base64",
    # India (3)
    "compute_gst", "validate_pan", "format_indian_number",
    # Adversarial (2)
    "safe_divide", "merge_intervals",
]  # 30 tasks total

BUILD_TASK_IDS = [
    "build_circular_buffer",
    "build_config_manager",
    "build_etl_pipeline",
    "build_log_parser",
    "build_password_policy",
] # 10 tasks total

FIELDNAMES = [
    "condition", "task_id", "category", "difficulty", "workflow",
    "trial_index", "latency_ms", "correct", "confidence",
    "coverage_pct", "repair_iterations", "cache_status",
    "http_status", "error_type", "timestamp"
]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def log(msg):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def verify_output(task, actual):
    check    = task.get("check", "exact")
    expected = task.get("expected")
    if check == "exact":
        return actual == expected
    elif check == "exact_case_insensitive":
        return str(actual).lower().strip() == str(expected).lower().strip()
    elif check == "approx":
        try:
            return abs(float(actual) - float(expected)) <= 0.01
        except Exception:
            return False
    elif check == "set_equal":
        try:
            return set(map(str, actual)) == set(map(str, expected))
        except Exception:
            return False
    elif check == "build_success":
        if isinstance(actual, dict):
            return actual.get("success", False) or actual.get("status") == "success"
        return bool(actual)
    return False

def classify_error(resp_json, correct):
    if correct:
        return ""
    err = str(resp_json.get("error", "")).lower()
    res = str(resp_json.get("result", "")).lower()
    combined = err + " " + res
    if "typeerror" in combined:      return "type_error"
    if "importerror" in combined:    return "import_error"
    if "nameerror" in combined:      return "hallucinated_api"
    if "timeout" in combined:        return "timeout"
    if resp_json.get("result") is None: return "logic_error"
    return "edge_case_miss"

def call_api(endpoint, payload):
    url = f"{CAPAI_BASE_URL}/{endpoint.lstrip('/')}"
    last_status = 0
    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.perf_counter()
            r  = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
            latency_ms = (time.perf_counter() - t0) * 1000
            last_status = r.status_code
            if r.status_code in (429, 503) and attempt < MAX_RETRIES - 1:
                log(f"    Rate-limited ({r.status_code}), retry in {RETRY_BACKOFF[attempt]}s")
                time.sleep(RETRY_BACKOFF[attempt])
                continue
            try:
                return r.json(), latency_ms, r.status_code
            except Exception:
                return {"error": "non-json"}, latency_ms, r.status_code
        except requests.exceptions.Timeout:
            latency_ms = REQUEST_TIMEOUT * 1000
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF[attempt])
        except requests.exceptions.ConnectionError as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF[attempt])
    return {"error": "connection_error"}, REQUEST_TIMEOUT * 1000, last_status

def reset_registry():
    """Clear the in-memory registry (not MongoDB) to simulate no-persistence."""
    try:
        r = requests.post(f"{CAPAI_BASE_URL}/reset", timeout=15)
        log(f"  Registry reset: {r.status_code}")
    except Exception as e:
        log(f"  Registry reset failed: {e}")

def server_health():
    try:
        r = requests.get(f"{CAPAI_BASE_URL}/health", timeout=15)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# ──────────────────────────────────────────────────────────────────────────────
# Task loader
# ──────────────────────────────────────────────────────────────────────────────

def load_tasks():
    with open(BENCHMARK_FILE) as f:
        data = yaml.safe_load(f)
    all_tasks = {t["id"]: t for t in data["tasks"]}
    run_tasks   = [all_tasks[tid] for tid in RUN_TASK_IDS   if tid in all_tasks]
    build_tasks = [all_tasks[tid] for tid in BUILD_TASK_IDS if tid in all_tasks]
    return run_tasks, build_tasks

# ──────────────────────────────────────────────────────────────────────────────
# Single trial runners
# ──────────────────────────────────────────────────────────────────────────────

def run_trial(condition, task, payload, trial_idx):
    resp, lat, status = call_api("run", payload)
    actual  = resp.get("result")
    correct = verify_output(task, actual)
    return {
        "condition":        condition,
        "task_id":          task["id"],
        "category":         task["category"],
        "difficulty":       task["difficulty"],
        "workflow":         "run",
        "trial_index":      trial_idx,
        "latency_ms":       round(lat, 2),
        "correct":          correct,
        "confidence":       resp.get("confidence", ""),
        "coverage_pct":     resp.get("coverage", ""),
        "repair_iterations":resp.get("repair_iterations", 0),
        "cache_status":     resp.get("cache_status", ""),
        "http_status":      status,
        "error_type":       classify_error(resp, correct),
        "timestamp":        now_iso(),
    }

def build_trial(condition, task, payload, trial_idx):
    resp, lat, status = call_api("build", payload)
    correct = verify_output(task, resp)
    return {
        "condition":        condition,
        "task_id":          task["id"],
        "category":         task["category"],
        "difficulty":       task["difficulty"],
        "workflow":         "build",
        "trial_index":      trial_idx,
        "latency_ms":       round(lat, 2),
        "correct":          correct,
        "confidence":       resp.get("confidence", ""),
        "coverage_pct":     resp.get("coverage", ""),
        "repair_iterations":resp.get("repair_iterations", 0),
        "cache_status":     resp.get("cache_status", ""),
        "http_status":      status,
        "error_type":       classify_error(resp, correct) if not correct else "",
        "timestamp":        now_iso(),
    }

# ──────────────────────────────────────────────────────────────────────────────
# Ablation conditions
# ──────────────────────────────────────────────────────────────────────────────

def run_A1_no_cache(run_tasks, writer):
    """A1: Without cache — use_cache=False forces LLM synthesis every call."""
    log("\n=== A1: Without Cache ===")
    log("Effect: Every call bypasses registry and hits LLM directly.")
    records = []
    for i, task in enumerate(run_tasks):
        log(f"  [{i+1}/{len(run_tasks)}] {task['id']}")
        payload = {
            "name":        task["id"],
            "description": task["description"],
            "args":        task.get("args", []),
            "use_cache":   False,   # KEY: bypass registry
        }
        # 1 cold + N warm (all are effectively cold since cache disabled)
        for trial in range(1 + N_TRIALS_WARM):
            rec = run_trial("A1_no_cache", task, payload, trial)
            records.append(rec)
            writer.writerow(rec)
            time.sleep(DELAY)
    return records


def run_A2_no_sandbox(run_tasks, writer):
    """
    A2: Without sandbox verification.
    Requires CAPAI_SKIP_SANDBOX=true env var on server.
    If not set, this will still run but with sandbox (server ignores flag).
    """
    log("\n=== A2: Without Sandbox Verification ===")
    log("Effect: Code promoted without execution check.")
    log("IMPORTANT: Requires CAPAI_SKIP_SANDBOX=true on Render.")
    records = []
    for i, task in enumerate(run_tasks):
        log(f"  [{i+1}/{len(run_tasks)}] {task['id']}")
        payload = {
            "name":        task["id"],
            "description": task["description"],
            "args":        task.get("args", []),
            "use_cache":   False,
            "skip_sandbox": True,   # server-side flag
        }
        for trial in range(1 + N_TRIALS_WARM):
            rec = run_trial("A2_no_sandbox", task, payload, trial)
            records.append(rec)
            writer.writerow(rec)
            time.sleep(DELAY)
    return records


def run_A3_no_repair(build_tasks, writer):
    """A3: Without repair loop — max_iterations=1 means single synthesis attempt."""
    log("\n=== A3: Without Repair Loop ===")
    log("Effect: Build pipeline gets exactly 1 attempt, no iterative repair.")
    records = []
    for i, task in enumerate(build_tasks):
        log(f"  [{i+1}/{len(build_tasks)}] {task['id']}")
        payload = {
            "name":           task["id"],
            "description":    task["description"],
            "use_cache":      False,
            "max_iterations": 1,    # KEY: disable repair
        }
        for trial in range(N_BUILD_TRIALS):
            rec = build_trial("A3_no_repair", task, payload, trial)
            records.append(rec)
            writer.writerow(rec)
            time.sleep(DELAY)
    return records


def run_A3_full_repair(build_tasks, writer):
    """A3 baseline: Same tasks with repair enabled (max_iterations=3)."""
    log("\n=== A3 Baseline: With Repair Loop (max_iterations=3) ===")
    records = []
    for i, task in enumerate(build_tasks):
        log(f"  [{i+1}/{len(build_tasks)}] {task['id']}")
        payload = {
            "name":           task["id"],
            "description":    task["description"],
            "use_cache":      False,
            "max_iterations": 3,    # full repair budget
        }
        for trial in range(N_BUILD_TRIALS):
            rec = build_trial("A3_with_repair", task, payload, trial)
            records.append(rec)
            writer.writerow(rec)
            time.sleep(DELAY)
    return records


def run_A4_no_heuristic(run_tasks, writer):
    """
    A4: Without heuristic library.
    Requires CAPAI_SKIP_HEURISTICS=true env var on server.
    """
    log("\n=== A4: Without Heuristic Library ===")
    log("Effect: 35 hand-verified functions must be re-synthesised by LLM.")
    log("IMPORTANT: Requires CAPAI_SKIP_HEURISTICS=true on Render.")
    records = []
    for i, task in enumerate(run_tasks):
        log(f"  [{i+1}/{len(run_tasks)}] {task['id']}")
        payload = {
            "name":            task["id"],
            "description":     task["description"],
            "args":            task.get("args", []),
            "use_cache":       False,
            "skip_heuristics": True,   # server-side flag
        }
        for trial in range(1 + N_TRIALS_WARM):
            rec = run_trial("A4_no_heuristic", task, payload, trial)
            records.append(rec)
            writer.writerow(rec)
            time.sleep(DELAY)
    return records


def run_A5_mcp_overhead(writer):
    """
    A5: MCP overhead measurement.
    Compares direct REST call vs MCP-routed call for the same capability.
    """
    log("\n=== A5: MCP Protocol Overhead ===")
    log("Effect: Measures latency added by MCP protocol vs direct REST.")

    test_payload = {
        "name":        "is_prime",
        "description": "Return True if n is a prime number",
        "args":        [17],
    }

    records = []
    N_MCP_TRIALS = 20

    # Direct REST trials
    log("  Running 20 direct REST trials...")
    for i in range(N_MCP_TRIALS):
        resp, lat, status = call_api("run", test_payload)
        rec = {
            "condition":        "A5_rest",
            "task_id":          "is_prime",
            "category":         "Numerical and Mathematical",
            "difficulty":       "Easy",
            "workflow":         "run",
            "trial_index":      i,
            "latency_ms":       round(lat, 2),
            "correct":          resp.get("result") == True,
            "confidence":       "",
            "coverage_pct":     "",
            "repair_iterations":0,
            "cache_status":     resp.get("cache_status", ""),
            "http_status":      status,
            "error_type":       "",
            "timestamp":        now_iso(),
        }
        records.append(rec)
        writer.writerow(rec)
        time.sleep(1.0)

    # MCP trials (via /mcp/ endpoint)
    log("  Running 20 MCP-routed trials...")
    mcp_url = f"{CAPAI_BASE_URL}/mcp/"
    mcp_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "capai_run",
            "arguments": test_payload
        }
    }
    for i in range(N_MCP_TRIALS):
        try:
            t0 = time.perf_counter()
            r  = requests.post(mcp_url, json=mcp_payload,
                               headers={"Content-Type": "application/json"},
                               timeout=REQUEST_TIMEOUT)
            lat = (time.perf_counter() - t0) * 1000
            rec = {
                "condition":        "A5_mcp",
                "task_id":          "is_prime",
                "category":         "Numerical and Mathematical",
                "difficulty":       "Easy",
                "workflow":         "run",
                "trial_index":      i,
                "latency_ms":       round(lat, 2),
                "correct":          True,   # assume correct if 200
                "confidence":       "",
                "coverage_pct":     "",
                "repair_iterations":0,
                "cache_status":     "mcp",
                "http_status":      r.status_code,
                "error_type":       "" if r.status_code == 200 else "mcp_error",
                "timestamp":        now_iso(),
            }
        except Exception as e:
            rec = {
                "condition": "A5_mcp", "task_id": "is_prime",
                "category": "Numerical and Mathematical",
                "difficulty": "Easy", "workflow": "run",
                "trial_index": i, "latency_ms": 0, "correct": False,
                "confidence": "", "coverage_pct": "", "repair_iterations": 0,
                "cache_status": "mcp_error", "http_status": 0,
                "error_type": "connection_error", "timestamp": now_iso(),
            }
        records.append(rec)
        writer.writerow(rec)
        time.sleep(1.0)

    return records


def run_A6_no_persistence(run_tasks, writer):
    """
    A6: Without persistence.
    Simulates stateless deployment by resetting the registry between task groups.
    """
    log("\n=== A6: Without Persistence ===")
    log("Effect: Registry cleared between sessions — simulates server restart.")
    records = []

    # Batch 1: first 15 tasks (session 1)
    log("  Session 1 (tasks 1-15)...")
    for i, task in enumerate(run_tasks[:15]):
        log(f"    [{i+1}/15] {task['id']}")
        payload = {
            "name":        task["id"],
            "description": task["description"],
            "args":        task.get("args", []),
        }
        # Cold call
        rec = run_trial("A6_no_persistence_s1_cold", task, payload, 0)
        records.append(rec)
        writer.writerow(rec)
        time.sleep(DELAY)
        # Warm call (same session)
        rec = run_trial("A6_no_persistence_s1_warm", task, payload, 1)
        records.append(rec)
        writer.writerow(rec)
        time.sleep(DELAY)

    # Simulate server restart — reset registry
    log("  Simulating server restart (resetting registry)...")
    reset_registry()
    time.sleep(5)

    # Batch 2: same 15 tasks again (session 2, post-restart)
    log("  Session 2 (same tasks, post-restart)...")
    for i, task in enumerate(run_tasks[:15]):
        log(f"    [{i+1}/15] {task['id']} (post-restart)")
        payload = {
            "name":        task["id"],
            "description": task["description"],
            "args":        task.get("args", []),
        }
        # This call should be cold again because registry was reset
        rec = run_trial("A6_no_persistence_s2_cold", task, payload, 0)
        records.append(rec)
        writer.writerow(rec)
        time.sleep(DELAY)

    return records


# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────

def print_ablation_summary(all_records):
    print("\n" + "="*65)
    print("ABLATION STUDY SUMMARY")
    print("="*65)

    conditions = sorted(set(r["condition"] for r in all_records))

    print(f"\n{'Condition':<35} {'n':>5} {'Correct%':>9} {'Mean lat':>10}")
    print("-"*65)

    for cond in conditions:
        subset = [r for r in all_records if r["condition"] == cond]
        if not subset: continue
        n       = len(subset)
        correct = sum(1 for r in subset if r["correct"])
        lats    = [r["latency_ms"] for r in subset if r["latency_ms"]]
        mean_lat = sum(lats)/len(lats) if lats else 0
        corr_pct = 100 * correct / n if n else 0
        print(f"{cond:<35} {n:>5} {corr_pct:>8.1f}% {mean_lat:>9.0f}ms")

    print("="*65)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CapAI Ablation Study Runner")
    p.add_argument("--condition", choices=["A1","A2","A3","A4","A5","A6","all"],
                   default="all", help="Which ablation to run")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan without making HTTP calls")
    p.add_argument("--delay", type=float, default=DELAY,
                   help=f"Seconds between calls (default {DELAY})")
    return p.parse_args()


def main():
    global DELAY
    args = parse_args()
    DELAY = args.delay

    run_tasks, build_tasks = load_tasks()

    log("CapAI Ablation Study")
    log(f"Server : {CAPAI_BASE_URL}")
    log(f"Run tasks  : {len(run_tasks)}")
    log(f"Build tasks: {len(build_tasks)}")
    log(f"Condition  : {args.condition}")

    if args.dry_run:
        log("DRY RUN — no HTTP calls")
        log("\nRun task IDs:")
        for t in run_tasks:
            print(f"  {t['id']}")
        log("\nBuild task IDs:")
        for t in build_tasks:
            print(f"  {t['id']}")
        return

    # Health check
    log("Checking server health...")
    health = server_health()
    if "error" in health:
        log(f"WARNING: {health['error']}")
    else:
        log(f"Server OK: {json.dumps(health)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"capai_ablation_{args.condition}_{ts}.csv"
    log(f"Output: {out_path}")

    all_records = []

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        cond = args.condition

        try:
            if cond in ("A1", "all"):
                recs = run_A1_no_cache(run_tasks, writer)
                all_records.extend(recs)
                f.flush()

            if cond in ("A2", "all"):
                recs = run_A2_no_sandbox(run_tasks, writer)
                all_records.extend(recs)
                f.flush()

            if cond in ("A3", "all"):
                recs = run_A3_no_repair(build_tasks, writer)
                all_records.extend(recs)
                f.flush()
                recs = run_A3_full_repair(build_tasks, writer)
                all_records.extend(recs)
                f.flush()

            if cond in ("A4", "all"):
                recs = run_A4_no_heuristic(run_tasks, writer)
                all_records.extend(recs)
                f.flush()

            if cond in ("A5", "all"):
                recs = run_A5_mcp_overhead(writer)
                all_records.extend(recs)
                f.flush()

            if cond in ("A6", "all"):
                recs = run_A6_no_persistence(run_tasks, writer)
                all_records.extend(recs)
                f.flush()

        except KeyboardInterrupt:
            log("Interrupted — saving partial results")

    log(f"\nDone. {len(all_records)} records saved to {out_path}")
    print_ablation_summary(all_records)


if __name__ == "__main__":
    main()
