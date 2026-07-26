#!/usr/bin/env python3
"""
CapAI Phase 2 — Results Analyzer
=================================
Reads the benchmark CSV and prints all statistics needed for the paper.

Usage:
    python analyze_results.py results/capai_benchmark_YYYYMMDD_HHMMSS.csv
    python analyze_results.py results/capai_benchmark_YYYYMMDD_HHMMSS.csv --latex
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ──────────────────────────────────────────────────────────────────────────────

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    n = len(xs)
    if not n:
        return 0.0
    if n % 2 == 1:
        return xs[n // 2]
    return (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def ci95(xs):
    """95% confidence interval half-width using normal approximation."""
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    return 1.96 * std(xs) / math.sqrt(len(xs))


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for proportion k/n. Returns (lo, hi)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def welch_t(xs, ys):
    """Welch's t-statistic and approximate p-value (two-sided)."""
    xs = [x for x in xs if x is not None]
    ys = [y for y in ys if y is not None]
    n1, n2 = len(xs), len(ys)
    if n1 < 2 or n2 < 2:
        return None, None
    m1, m2 = mean(xs), mean(ys)
    s1, s2 = std(xs), std(ys)
    v1, v2 = s1 ** 2 / n1, s2 ** 2 / n2
    t = (m1 - m2) / math.sqrt(v1 + v2)
    # Welch-Satterthwaite df
    df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    # Approximate p-value using normal (accurate for large df)
    # For small df use the t-table note below
    p_approx = 2 * (1 - _norm_cdf(abs(t)))
    return t, p_approx, df


def _norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def cohens_d(xs, ys):
    xs = [x for x in xs if x is not None]
    ys = [y for y in ys if y is not None]
    n1, n2 = len(xs), len(ys)
    if n1 < 2 or n2 < 2:
        return None
    s_pooled = math.sqrt(((n1 - 1) * std(xs) ** 2 + (n2 - 1) * std(ys) ** 2) / (n1 + n2 - 2))
    if s_pooled == 0:
        return None
    return (mean(xs) - mean(ys)) / s_pooled


def two_prop_z(k1, n1, k2, n2):
    """Two-proportion z-test. Returns z, p."""
    if n1 == 0 or n2 == 0:
        return None, None
    p1, p2 = k1 / n1, k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    denom = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if denom == 0:
        return None, None
    z = (p1 - p2) / denom
    p = 2 * (1 - _norm_cdf(abs(z)))
    return z, p


def cohens_h(p1, p2):
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


# ──────────────────────────────────────────────────────────────────────────────
# Load CSV
# ──────────────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> list[dict]:
    records = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Type coerce
            row["latency_ms"] = float(row["latency_ms"]) if row["latency_ms"] else None
            row["correct"] = row["correct"].lower() in ("true", "1", "yes")
            row["confidence"] = float(row["confidence"]) if row["confidence"] else None
            row["coverage_pct"] = float(row["coverage_pct"]) if row["coverage_pct"] else None
            row["repair_iterations"] = int(row["repair_iterations"]) if row["repair_iterations"] else 0
            records.append(row)
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Analysis sections
# ──────────────────────────────────────────────────────────────────────────────

def section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def analyze_latency_correctness(records, latex=False):
    section("TABLE II — Single-Function Latency and Correctness")
    run_records = [r for r in records if r["workflow"] == "run"]

    for cond in ["cold", "warm", "baseline"]:
        subset = [r for r in run_records if r["condition"] == cond]
        if not subset:
            continue
        lats  = [r["latency_ms"] for r in subset]
        corr  = [r for r in subset if r["correct"]]
        n     = len(subset)
        m     = mean(lats)
        s     = std(lats)
        med   = median(lats)
        hw    = ci95(lats)
        c_pct = 100 * len(corr) / n
        lo, hi = wilson_ci(len(corr), n)
        print(f"\nCondition: {cond.upper()}  (n={n})")
        print(f"  Mean ± SD  : {m:,.1f} ± {s:,.1f} ms")
        print(f"  95% CI     : [{m-hw:,.1f}, {m+hw:,.1f}] ms")
        print(f"  Median     : {med:,.1f} ms")
        print(f"  Correct    : {len(corr)}/{n} = {c_pct:.1f}%  Wilson CI [{100*lo:.1f}%, {100*hi:.1f}%]")

    # Cache speedup
    cold_lats = [r["latency_ms"] for r in run_records if r["condition"] == "cold"]
    warm_lats = [r["latency_ms"] for r in run_records if r["condition"] == "warm"]
    if cold_lats and warm_lats:
        speedup = mean(cold_lats) / mean(warm_lats)
        print(f"\nCache speedup: {speedup:.2f}×")

    # Statistical tests: cold vs warm latency
    if cold_lats and warm_lats:
        result = welch_t(cold_lats, warm_lats)
        if result[0] is not None:
            t, p, df = result
            d = cohens_d(cold_lats, warm_lats)
            print(f"\nWelch's t (cold vs warm latency):")
            print(f"  t({df:.1f}) = {t:.2f},  p = {p:.2e},  Cohen's d = {d:.2f}")

    # Statistical tests: correctness baseline vs warm
    run_records2 = [r for r in run_records]
    warm_corr  = [r for r in run_records2 if r["condition"] == "warm" and r["correct"]]
    warm_total = [r for r in run_records2 if r["condition"] == "warm"]
    base_corr  = [r for r in run_records2 if r["condition"] == "baseline" and r["correct"]]
    base_total = [r for r in run_records2 if r["condition"] == "baseline"]

    if warm_total and base_total:
        k1, n1 = len(warm_corr), len(warm_total)
        k2, n2 = len(base_corr), len(base_total)
        z, p = two_prop_z(k1, n1, k2, n2)
        h = cohens_h(k1/n1, k2/n2) if n1 > 0 and n2 > 0 else None
        print(f"\nTwo-proportion z-test (warm correctness vs baseline correctness):")
        print(f"  warm: {k1}/{n1} = {100*k1/n1:.1f}%")
        print(f"  baseline: {k2}/{n2} = {100*k2/n2:.1f}%")
        if z is not None:
            print(f"  z = {z:.3f},  p = {p:.4f},  Cohen's h = {h:.3f}")


def analyze_build_results(records, latex=False):
    section("TABLE III — Advanced-Build Outcomes")
    build_recs = [r for r in records if r["workflow"] == "build"]
    if not build_recs:
        print("No /build records found.")
        return

    by_task = defaultdict(list)
    for r in build_recs:
        by_task[r["task_id"]].append(r)

    print(f"\n{'Task':<35} {'Succ':>6} {'Mean Iter':>10} {'Mean Cov':>10} {'Mean Conf':>10}")
    print("-" * 75)

    all_success = []
    all_iters   = []
    all_covs    = []
    all_confs   = []

    for task_id, recs in sorted(by_task.items()):
        succ  = [r for r in recs if r["correct"]]
        iters = [r["repair_iterations"] for r in recs]
        covs  = [r["coverage_pct"] for r in recs if r["coverage_pct"] is not None]
        confs = [r["confidence"] for r in recs if r["confidence"] is not None]

        n = len(recs)
        succ_str  = f"{len(succ)}/{n}"
        iter_str  = f"{mean(iters):.2f} ± {std(iters):.1f}" if iters else "—"
        cov_str   = f"{mean(covs):.1f} ± {std(covs):.1f}%" if covs else "—"
        conf_str  = f"{mean(confs):.1f} ± {std(confs):.1f}" if confs else "—"

        print(f"{task_id:<35} {succ_str:>6}  {iter_str:>10}  {cov_str:>12}  {conf_str:>10}")
        all_success.extend([r["correct"] for r in recs])
        all_iters.extend(iters)
        all_covs.extend(covs)
        all_confs.extend(confs)

    print("-" * 75)
    total_succ = sum(all_success)
    total_n    = len(all_success)
    lo, hi = wilson_ci(total_succ, total_n)
    print(f"{'POOLED':<35} {total_succ}/{total_n}  {mean(all_iters):.2f} ± {std(all_iters):.1f}  {mean(all_covs):.1f} ± {std(all_covs):.1f}%  {mean(all_confs):.1f} ± {std(all_confs):.1f}")
    print(f"\nPooled success rate: {100*total_succ/total_n:.1f}%  Wilson CI [{100*lo:.1f}%, {100*hi:.1f}%]")


def analyze_by_category(records):
    section("Correctness by Category")
    run_recs = [r for r in records if r["workflow"] == "run"]
    by_cat = defaultdict(list)
    for r in run_recs:
        by_cat[r["category"]].append(r)

    print(f"\n{'Category':<40} {'Condition':<12} {'Correct':>8} {'n':>5} {'%':>7}")
    print("-" * 76)
    for cat, recs in sorted(by_cat.items()):
        for cond in ["cold", "warm", "baseline"]:
            sub = [r for r in recs if r["condition"] == cond]
            if not sub:
                continue
            corr = sum(r["correct"] for r in sub)
            n    = len(sub)
            print(f"{cat:<40} {cond:<12} {corr:>8} {n:>5} {100*corr/n:>6.1f}%")


def analyze_by_difficulty(records):
    section("Correctness by Difficulty")
    run_recs = [r for r in records if r["workflow"] == "run"]
    by_diff = defaultdict(list)
    for r in run_recs:
        by_diff[r["difficulty"]].append(r)

    for diff in ["Easy", "Medium", "Hard"]:
        recs = by_diff.get(diff, [])
        if not recs:
            continue
        for cond in ["cold", "warm", "baseline"]:
            sub = [r for r in recs if r["condition"] == cond]
            if not sub:
                continue
            corr = sum(r["correct"] for r in sub)
            n    = len(sub)
            print(f"  {diff:<8} {cond:<12}: {corr}/{n} = {100*corr/n:.1f}%")


def analyze_error_taxonomy(records):
    section("Error Taxonomy")
    from collections import Counter
    errors = [r["error_type"] for r in records if r["error_type"] and r["error_type"] not in ("", "runner_exception")]
    if not errors:
        print("No errors recorded.")
        return
    total_wrong = len([r for r in records if not r["correct"]])
    print(f"\nTotal incorrect results: {total_wrong}")
    print(f"\n{'Error Type':<25} {'Count':>6} {'% of errors':>12}")
    print("-" * 45)
    for etype, cnt in Counter(errors).most_common():
        print(f"{etype:<25} {cnt:>6} {100*cnt/len(errors):>11.1f}%")


def analyze_confidence_correlation(records):
    section("Confidence Score Validation (Task 7)")
    build_recs = [r for r in records if r["workflow"] == "build"
                  and r["confidence"] is not None]
    if not build_recs:
        print("No build records with confidence scores found.")
        return

    confs   = [r["confidence"] for r in build_recs]
    correct = [1.0 if r["correct"] else 0.0 for r in build_recs]
    n = len(confs)

    # Pearson
    mc, ms = mean(confs), mean(correct)
    num = sum((c - mc) * (s - ms) for c, s in zip(confs, correct))
    denom = math.sqrt(sum((c - mc)**2 for c in confs) * sum((s - ms)**2 for s in correct))
    pearson_r = num / denom if denom > 0 else 0.0
    print(f"\nPearson r (confidence vs correct) = {pearson_r:.4f}  (n={n})")

    # Bucket analysis
    bucket_size = 10
    print(f"\n{'Bucket':<15} {'n':>5} {'Mean conf':>10} {'Frac correct':>13}")
    print("-" * 48)
    for lo in range(0, 100, bucket_size):
        hi = lo + bucket_size
        bucket = [r for r in build_recs if r["confidence"] is not None and lo <= r["confidence"] < hi]
        if not bucket:
            continue
        avg_conf = mean([r["confidence"] for r in bucket])
        frac_corr = mean([1.0 if r["correct"] else 0.0 for r in bucket])
        print(f"[{lo:3d}, {hi:3d})       {len(bucket):>5} {avg_conf:>10.1f} {frac_corr:>12.3f}")


def analyze_cache_performance(records):
    section("Cache Performance")
    run_recs = [r for r in records if r["workflow"] == "run"]
    hits   = [r for r in run_recs if r["cache_status"] == "hit"]
    misses = [r for r in run_recs if r["cache_status"] == "miss"]
    heur   = [r for r in run_recs if r["cache_status"] == "heuristic"]
    total  = len(run_recs)

    print(f"\nTotal /run calls : {total}")
    print(f"Cache hits       : {len(hits)} ({100*len(hits)/total:.1f}%)")
    print(f"Heuristic hits   : {len(heur)} ({100*len(heur)/total:.1f}%)")
    print(f"Cache misses     : {len(misses)} ({100*len(misses)/total:.1f}%)")

    if hits:
        print(f"\nMean latency (cache hits)   : {mean([r['latency_ms'] for r in hits]):,.1f} ms")
    if misses:
        print(f"Mean latency (cache misses) : {mean([r['latency_ms'] for r in misses]):,.1f} ms")
    if heur:
        print(f"Mean latency (heuristics)   : {mean([r['latency_ms'] for r in heur]):,.1f} ms")


def analyze_provider_breakdown(records):
    section("Provider Usage Breakdown")
    from collections import Counter
    providers = [r["provider_used"] for r in records if r["provider_used"]]
    for prov, cnt in Counter(providers).most_common():
        corr = [r for r in records if r["provider_used"] == prov and r["correct"]]
        total = [r for r in records if r["provider_used"] == prov]
        print(f"  {prov:<20}: {cnt:>5} calls, {100*len(corr)/len(total):.1f}% correct")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="CapAI Phase 2 Results Analyzer")
    p.add_argument("csv_file", help="Path to benchmark CSV file")
    p.add_argument("--latex",  action="store_true", help="Print LaTeX table snippets")
    args = p.parse_args()

    path = Path(args.csv_file)
    if not path.exists():
        print(f"ERROR: file not found: {path}")
        sys.exit(1)

    records = load_csv(str(path))
    print(f"\nLoaded {len(records)} records from {path.name}")

    analyze_latency_correctness(records, args.latex)
    analyze_build_results(records, args.latex)
    analyze_by_category(records)
    analyze_by_difficulty(records)
    analyze_error_taxonomy(records)
    analyze_confidence_correlation(records)
    analyze_cache_performance(records)
    analyze_provider_breakdown(records)


if __name__ == "__main__":
    main()
