#!/usr/bin/env python3
"""
CapAI Multi-Model Evaluation Runner
=====================================
Tests CapAI with 3 different LLMs on the same 30-task subset.
No new API keys needed — just change CAPAI_GROQ_MODEL in Render.

Models tested:
  M1: llama-3.3-70b-versatile  (current, 70B — already have data)
  M2: gemma2-9b-it              (Google Gemma 2 9B)
  M3: llama-3.1-8b-instant      (LLaMA 3.1 8B, fast/small)

Usage:
    # Step 1: Change CAPAI_GROQ_MODEL in Render to gemma2-9b-it, then:
    python run_multimodel.py --model gemma2

    # Step 2: Change to llama-3.1-8b-instant, then:
    python run_multimodel.py --model llama8b

    # Analyze all results:
    python run_multimodel.py --analyze
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import math

import yaml
import requests

CAPAI_BASE_URL = os.environ.get(
    "CAPAI_URL",
    "https://capai-capability-bootstrapping-ai-fu58.onrender.com"
).rstrip("/")

OUTPUT_DIR    = Path(__file__).parent / "results"
BENCHMARK_FILE = Path(__file__).parent / "benchmark_tasks.yaml"
DELAY         = 3.0
REQUEST_TIMEOUT = 120
MAX_RETRIES   = 3
RETRY_BACKOFF = [5, 15, 45]

# Same 30 tasks used in ablation for fair comparison
EVAL_TASK_IDS = [
    "reverse_words", "count_vowels", "is_palindrome_str", "levenshtein_distance",
    "is_prime", "fibonacci", "sieve_of_eratosthenes", "matrix_multiply",
    "two_sum", "max_subarray_sum", "coin_change", "sliding_window_max",
    "is_valid_email", "is_valid_ipv4", "luhn_check", "is_valid_isbn13",
    "days_between", "is_leap_year", "business_days_between",
    "count_lines", "flatten_dict", "merge_dicts",
    "hex_to_rgb", "md5_hex", "encode_base64",
    "compute_gst", "validate_pan", "format_indian_number",
    "safe_divide", "merge_intervals",
]

MODEL_NAMES = {
    "llama70b":  "llama-3.3-70b-versatile",
    "llama8b":   "llama-3.1-8b-instant",
    "gptoss20b": "openai/gpt-oss-20b",
    "gptoss120b":"openai/gpt-oss-120b",
}

FIELDNAMES = [
    "model_key", "model_name", "task_id", "category", "difficulty",
    "trial_index", "latency_ms", "correct", "cache_status",
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

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs)/len(xs) if xs else 0.0

def wilson_ci(k, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p = k/n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n))/denom
    half = (z/denom)*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return (max(0.0, centre-half), min(1.0, centre+half))

def verify_output(task, actual):
    check    = task.get("check", "exact")
    expected = task.get("expected")
    if check == "exact":
        return actual == expected
    elif check == "exact_case_insensitive":
        return str(actual).lower().strip() == str(expected).lower().strip()
    elif check == "approx":
        try: return abs(float(actual) - float(expected)) <= 0.01
        except: return False
    elif check == "set_equal":
        try: return set(map(str, actual)) == set(map(str, expected))
        except: return False
    return False

def classify_error(resp_json, correct):
    if correct: return ""
    err = str(resp_json.get("error","")).lower()
    res = str(resp_json.get("result","")).lower()
    c = err + " " + res
    if "typeerror" in c: return "type_error"
    if "importerror" in c: return "import_error"
    if "nameerror" in c: return "hallucinated_api"
    if "timeout" in c: return "timeout"
    if resp_json.get("result") is None: return "logic_error"
    return "edge_case_miss"

def call_api(payload):
    url = f"{CAPAI_BASE_URL}/run"
    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.perf_counter()
            r  = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
            lat = (time.perf_counter() - t0) * 1000
            if r.status_code in (429, 503) and attempt < MAX_RETRIES - 1:
                log(f"  Rate-limited, retry in {RETRY_BACKOFF[attempt]}s")
                time.sleep(RETRY_BACKOFF[attempt])
                continue
            try: return r.json(), lat, r.status_code
            except: return {"error":"non-json"}, lat, r.status_code
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES - 1: time.sleep(RETRY_BACKOFF[attempt])
        except requests.exceptions.ConnectionError:
            if attempt < MAX_RETRIES - 1: time.sleep(RETRY_BACKOFF[attempt])
    return {"error":"connection_error"}, REQUEST_TIMEOUT*1000, 0

def check_current_model():
    """Verify which model the server is currently using."""
    try:
        r = requests.get(f"{CAPAI_BASE_URL}/health", timeout=15)
        h = r.json()
        return h.get("model", "unknown")
    except:
        return "unknown"

# ──────────────────────────────────────────────────────────────────────────────
# Run one model
# ──────────────────────────────────────────────────────────────────────────────

def run_model(model_key, tasks, n_trials=3):
    model_name = MODEL_NAMES.get(model_key, model_key)

    # Verify server model
    current = check_current_model()
    log(f"Server model: {current}")
    if model_key != "llama70b" and model_name not in current:
        log(f"WARNING: Expected {model_name} but server reports {current}")
        log("Make sure CAPAI_GROQ_MODEL is set correctly in Render.")
        confirm = input("Continue anyway? (y/n): ").strip().lower()
        if confirm != "y":
            sys.exit(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"capai_multimodel_{model_key}_{ts}.csv"
    log(f"Output: {out_path}")

    all_records = []
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for i, task in enumerate(tasks):
            log(f"[{i+1}/{len(tasks)}] {task['id']}")
            payload = {
                "name":        task["id"],
                "description": task["description"],
                "args":        task.get("args", []),
                "use_cache":   False,  # fresh synthesis every time for fair comparison
            }

            for trial in range(n_trials):
                resp, lat, status = call_api(payload)
                actual  = resp.get("result")
                correct = verify_output(task, actual)
                rec = {
                    "model_key":   model_key,
                    "model_name":  model_name,
                    "task_id":     task["id"],
                    "category":    task["category"],
                    "difficulty":  task["difficulty"],
                    "trial_index": trial,
                    "latency_ms":  round(lat, 2),
                    "correct":     correct,
                    "cache_status": resp.get("cache_status",""),
                    "http_status": status,
                    "error_type":  classify_error(resp, correct),
                    "timestamp":   now_iso(),
                }
                all_records.append(rec)
                writer.writerow(rec)
                f.flush()
                time.sleep(DELAY)

    corr  = sum(1 for r in all_records if r["correct"])
    total = len(all_records)
    lo, hi = wilson_ci(corr, total)
    log(f"\n{model_key} ({model_name})")
    log(f"  Pass@1 (n={total}): {corr}/{total} = {100*corr/total:.1f}%  [{100*lo:.1f}%, {100*hi:.1f}%]")
    log(f"  Mean latency: {mean([r['latency_ms'] for r in all_records]):.0f} ms")
    return all_records


# ──────────────────────────────────────────────────────────────────────────────
# Analyze all model results
# ──────────────────────────────────────────────────────────────────────────────

def analyze_multimodel():
    files = sorted(OUTPUT_DIR.glob("capai_multimodel_*.csv"))
    if not files:
        print("No multi-model result files found.")
        return

    records = []
    for f in files:
        with open(f, newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                row["latency_ms"] = float(row["latency_ms"]) if row["latency_ms"] else 0
                row["correct"]    = row["correct"].lower() in ("true","1","yes")
                records.append(row)

    by_model = defaultdict(list)
    for r in records:
        by_model[r["model_key"]].append(r)

    print("\n" + "="*75)
    print("MULTI-MODEL EVALUATION RESULTS")
    print("="*75)
    print(f"\n{'Model':<15} {'Name':<30} {'n':>5} {'Pass@1':>8} {'Wilson CI':>20} {'Mean lat':>10}")
    print("-"*75)

    model_order = ["llama70b", "gemma2", "llama8b", "mixtral"]
    for mk in model_order + [k for k in by_model if k not in model_order]:
        recs = by_model.get(mk)
        if not recs: continue
        corr = sum(1 for r in recs if r["correct"])
        n    = len(recs)
        lat  = mean([r["latency_ms"] for r in recs])
        lo, hi = wilson_ci(corr, n)
        name = recs[0]["model_name"] if recs else mk
        print(f"{mk:<15} {name:<30} {n:>5} {100*corr/n:>7.1f}%  "
              f"[{100*lo:.1f}%, {100*hi:.1f}%]  {lat:>9.0f}ms")

    # Per-category breakdown
    print("\n" + "="*75)
    print("PASS@1 BY CATEGORY")
    print("="*75)

    categories = sorted(set(r["category"] for r in records))
    model_keys = [k for k in model_order if k in by_model]

    header = f"{'Category':<28}" + "".join(f"{mk:>12}" for mk in model_keys)
    print(header)
    print("-"*(28 + 12*len(model_keys)))

    for cat in categories:
        row_str = f"{cat:<28}"
        for mk in model_keys:
            recs = [r for r in by_model[mk] if r["category"] == cat]
            if recs:
                corr = sum(1 for r in recs if r["correct"])
                row_str += f"{100*corr/len(recs):>11.0f}%"
            else:
                row_str += f"{'---':>12}"
        print(row_str)

    # Per-difficulty breakdown
    print("\n" + "="*75)
    print("PASS@1 BY DIFFICULTY")
    print("="*75)
    for diff in ["Easy", "Medium", "Hard"]:
        row_str = f"{diff:<28}"
        for mk in model_keys:
            recs = [r for r in by_model[mk] if r["difficulty"] == diff]
            if recs:
                corr = sum(1 for r in recs if r["correct"])
                row_str += f"{100*corr/len(recs):>11.0f}%"
            else:
                row_str += f"{'---':>12}"
        print(row_str)

    # LaTeX table
    print("\n" + "="*75)
    print("LATEX TABLE — MULTI-MODEL RESULTS (paste into paper)")
    print("="*75)
    print(r"""
\begin{table}[htbp]
  \centering
  \caption{Multi-Model Evaluation (30~Tasks, 3~Trials Each)}
  \label{tab:multimodel}
  \renewcommand{\arraystretch}{1.2}
  \begin{tabular}{llccc}
    \toprule
    \textbf{Model} & \textbf{Size} & \textbf{Pass@1}
      & \textbf{Wilson CI} & \textbf{Mean Lat.\ (ms)} \\
    \midrule""")

    model_meta = {
        "llama70b": ("LLaMA-3.3-70B", "70B"),
        "gemma2":   ("Gemma~2~9B",    "9B"),
        "llama8b":  ("LLaMA-3.1-8B",  "8B"),
        "mixtral":  ("Mixtral-8x7B",  "47B"),
    }
    for mk in model_order:
        recs = by_model.get(mk)
        if not recs: continue
        corr = sum(1 for r in recs if r["correct"])
        n    = len(recs)
        lat  = mean([r["latency_ms"] for r in recs])
        lo, hi = wilson_ci(corr, n)
        name, size = model_meta.get(mk, (mk, "?"))
        print(f"    {name} & {size} & {100*corr/n:.1f}\\% "
              f"& $[{100*lo:.1f}, {100*hi:.1f}]$ & {lat:.0f} \\\\")

    print(r"""    \bottomrule
    \multicolumn{5}{l}{\footnotesize All models served via Groq API at
      temperature~0. Pass@1 evaluated on 30-task fixed subset.}
  \end{tabular}
\end{table}""")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="CapAI Multi-Model Evaluation")
    p.add_argument("--model",
                   choices=list(MODEL_NAMES.keys()),
                   help="Which model to run (set CAPAI_GROQ_MODEL in Render first)")
    p.add_argument("--analyze", action="store_true",
                   help="Analyze existing multi-model result files")
    p.add_argument("--trials", type=int, default=3,
                   help="Trials per task (default: 3)")
    args = p.parse_args()

    if args.analyze:
        analyze_multimodel()
        return

    if not args.model:
        print("Specify --model or --analyze")
        print("Available models:", list(MODEL_NAMES.keys()))
        sys.exit(1)

    # Load tasks
    with open(BENCHMARK_FILE) as f:
        data = yaml.safe_load(f)
    all_tasks = {t["id"]: t for t in data["tasks"]}
    tasks = [all_tasks[tid] for tid in EVAL_TASK_IDS if tid in all_tasks]
    log(f"Loaded {len(tasks)} tasks")
    log(f"Model: {args.model} → {MODEL_NAMES[args.model]}")
    log(f"Trials per task: {args.trials}")
    log(f"Total calls: {len(tasks) * args.trials}")

    run_model(args.model, tasks, n_trials=args.trials)


if __name__ == "__main__":
    main()
