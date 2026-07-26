#!/usr/bin/env python3
"""
HumanEval runner v2 — correct verification approach.
Parses real test cases from HumanEval's check() function,
calls CapAI /run with actual args, verifies return values.
"""

import ast, csv, json, os, re, sys, time, yaml
from datetime import datetime, timezone
from pathlib import Path
import requests

CAPAI_BASE_URL = os.environ.get(
    "CAPAI_URL",
    "https://capai-capability-bootstrapping-ai-fu58.onrender.com"
).rstrip("/")

OUTPUT_DIR    = Path(__file__).parent / "results"
HE_YAML_FILE  = Path(__file__).parent / "humaneval_tasks.yaml"
DELAY         = 3.0
REQUEST_TIMEOUT = 120
MAX_RETRIES   = 3
RETRY_BACKOFF = [5, 15, 45]

FIELDNAMES = [
    "task_id", "he_task_id", "trial_index",
    "latency_ms", "correct", "pass_count", "total_tests",
    "http_status", "timestamp"
]

# ──────────────────────────────────────────────────────────────────────────────

def parse_test_cases(test_code):
    """
    Extract (args_list, expected) tuples from HumanEval check() function.
    Handles: assert candidate(a, b, ...) == expected
    """
    cases = []
    pattern = re.compile(
        r'assert\s+candidate\((.+?)\)\s*==\s*(.+?)(?:\n|$)',
        re.MULTILINE
    )
    for m in pattern.finditer(test_code):
        args_str     = m.group(1).strip()
        expected_str = m.group(2).strip()
        try:
            # wrap args in tuple for safe parsing
            args_tuple = ast.literal_eval(f"({args_str},)")
            expected   = ast.literal_eval(expected_str)
            cases.append((list(args_tuple), expected))
        except Exception:
            continue
    return cases


def call_capai(name, description, args, use_cache=True):
    url = f"{CAPAI_BASE_URL}/run"
    payload = {
        "name":        name,
        "description": description,
        "args":        args,
        "use_cache":   use_cache,
    }
    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.perf_counter()
            r  = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
            lat = (time.perf_counter() - t0) * 1000
            if r.status_code in (429, 503) and attempt < MAX_RETRIES - 1:
                print(f"    Rate-limited, retry in {RETRY_BACKOFF[attempt]}s")
                time.sleep(RETRY_BACKOFF[attempt])
                continue
            try:    return r.json(), lat, r.status_code
            except: return {"error": "non-json"}, lat, r.status_code
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES - 1: time.sleep(RETRY_BACKOFF[attempt])
        except requests.exceptions.ConnectionError:
            if attempt < MAX_RETRIES - 1: time.sleep(RETRY_BACKOFF[attempt])
    return {"error": "connection_error"}, REQUEST_TIMEOUT * 1000, 0


def run_humaneval(limit=164, start=0):
    print(f"Loading {HE_YAML_FILE}...")
    with open(HE_YAML_FILE) as f:
        data = yaml.safe_load(f)
    tasks = data["tasks"][start:limit]
    print(f"Running {len(tasks)} HumanEval tasks...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"capai_humaneval_v2_{ts}.csv"
    print(f"Output: {out_path}\n")

    total_pass = 0
    total_tasks = 0

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for i, task in enumerate(tasks):
            tid       = task["id"]
            ep        = task["entry_point"]
            prompt    = task["prompt"]
            test_code = task["test_code"]

            # Parse real test cases
            cases = parse_test_cases(test_code)
            if not cases:
                print(f"[{i+1}/{len(tasks)}] {tid}: no parseable test cases — skip")
                continue

            # Use first test case for synthesis (cold call)
            first_args, first_expected = cases[0]
            resp, lat, status = call_capai(
                name=ep,
                description=prompt,
                args=first_args,
                use_cache=False,   # cold synthesis
            )
            time.sleep(DELAY)

            # Evaluate against first test case
            result = resp.get("result")
            first_pass = (result == first_expected)

            # Run remaining test cases using cache
            pass_count = 1 if first_pass else 0
            for args, expected in cases[1:]:
                r2, _, _ = call_capai(
                    name=ep,
                    description=prompt,
                    args=args,
                    use_cache=True,  # warm
                )
                if r2.get("result") == expected:
                    pass_count += 1
                time.sleep(1.0)   # shorter delay for warm calls

            total_tests = len(cases)
            task_pass   = (pass_count == total_tests)
            if task_pass:
                total_pass += 1
            total_tasks += 1

            status_str = "✓" if task_pass else f"✗ ({pass_count}/{total_tests})"
            print(f"[{i+1}/{len(tasks)}] {tid}  {status_str}  ({lat:.0f}ms)")

            writer.writerow({
                "task_id":      tid,
                "he_task_id":   task.get("he_task_id", tid),
                "trial_index":  0,
                "latency_ms":   round(lat, 2),
                "correct":      task_pass,
                "pass_count":   pass_count,
                "total_tests":  total_tests,
                "http_status":  status,
                "timestamp":    datetime.now(timezone.utc).isoformat(),
            })
            f.flush()
            time.sleep(DELAY)

    pass_at_1 = 100 * total_pass / total_tasks if total_tasks else 0
    print(f"\n{'='*55}")
    print(f"HumanEval Pass@1: {total_pass}/{total_tasks} = {pass_at_1:.1f}%")
    print(f"Results: {out_path}")
    print(f"{'='*55}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=164)
    p.add_argument("--start", type=int, default=0)
    args = p.parse_args()
    run_humaneval(limit=args.limit, start=args.start)