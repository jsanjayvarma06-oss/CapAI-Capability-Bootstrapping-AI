#!/usr/bin/env python3
"""
CapAI Ablation Study Analyzer
==============================
Reads ablation CSVs and prints paper-ready Table VI.

Usage:
    python analyze_ablation.py results/ablation_*.csv
    python analyze_ablation.py results/ablation_all_*.csv
"""

import csv
import math
import sys
import glob
from collections import defaultdict
from pathlib import Path


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0

def std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2: return 0.0
    m = mean(xs)
    return math.sqrt(sum((x-m)**2 for x in xs) / len(xs))

def ci95(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2: return 0.0
    return 1.96 * std(xs) / math.sqrt(len(xs))

def wilson_ci(k, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p = k / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = (z/denom) * math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return (max(0.0, centre-half), min(1.0, centre+half))

def welch_t(xs, ys):
    xs = [x for x in xs if x is not None]
    ys = [y for y in ys if y is not None]
    if len(xs) < 2 or len(ys) < 2: return None, None, None
    n1, n2 = len(xs), len(ys)
    m1, m2 = mean(xs), mean(ys)
    s1, s2 = std(xs), std(ys)
    v1, v2 = s1**2/n1, s2**2/n2
    if v1+v2 == 0: return None, None, None
    t = (m1-m2)/math.sqrt(v1+v2)
    df = (v1+v2)**2 / (v1**2/(n1-1) + v2**2/(n2-1))
    p = 2*(1 - _norm_cdf(abs(t)))
    return t, p, df

def cohens_d(xs, ys):
    xs = [x for x in xs if x is not None]
    ys = [y for y in ys if y is not None]
    n1, n2 = len(xs), len(ys)
    if n1 < 2 or n2 < 2: return None
    s_pooled = math.sqrt(((n1-1)*std(xs)**2 + (n2-1)*std(ys)**2) / (n1+n2-2))
    if s_pooled == 0: return None
    return (mean(xs) - mean(ys)) / s_pooled

def two_prop_z(k1, n1, k2, n2):
    if n1 == 0 or n2 == 0: return None, None
    p1, p2 = k1/n1, k2/n2
    p_pool = (k1+k2)/(n1+n2)
    denom = math.sqrt(p_pool*(1-p_pool)*(1/n1+1/n2))
    if denom == 0: return None, None
    z = (p1-p2)/denom
    p = 2*(1-_norm_cdf(abs(z)))
    return z, p

def _norm_cdf(x):
    return (1.0 + math.erf(x/math.sqrt(2.0)))/2.0


def load_records(paths):
    records = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["latency_ms"]      = float(row["latency_ms"]) if row["latency_ms"] else None
                row["correct"]         = row["correct"].lower() in ("true","1","yes")
                row["confidence"]      = float(row["confidence"]) if row["confidence"] else None
                row["repair_iterations"] = int(row["repair_iterations"]) if row["repair_iterations"] else 0
                records.append(row)
    return records


def section(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def analyze(records):

    # Group by condition
    by_cond = defaultdict(list)
    for r in records:
        by_cond[r["condition"]].append(r)

    section("ABLATION SUMMARY TABLE (Table VI in paper)")

    # Define display order and labels
    condition_meta = {
        "A1_no_cache":           ("A1", "Without Cache",        "run"),
        "A2_no_sandbox":         ("A2", "Without Sandbox",      "run"),
        "A3_no_repair":          ("A3", "Without Repair Loop",  "build"),
        "A3_with_repair":        ("A3b","With Repair (baseline)","build"),
        "A4_no_heuristic":       ("A4", "Without Heuristics",   "run"),
        "A5_rest":               ("A5", "Direct REST",          "run"),
        "A5_mcp":                ("A5b","MCP-routed",           "run"),
        "A6_no_persistence_s1_cold": ("A6a","No Persistence (S1 cold)","run"),
        "A6_no_persistence_s1_warm": ("A6b","No Persistence (S1 warm)","run"),
        "A6_no_persistence_s2_cold": ("A6c","No Persistence (S2 cold)","run"),
    }

    print(f"\n{'ID':<4} {'Condition':<28} {'n':>5} {'Correct%':>9} "
          f"{'Wilson CI':>18} {'Mean lat':>10} {'±95CI':>8}")
    print("-"*85)

    results = {}
    for cond_key, (cid, label, wf) in condition_meta.items():
        recs = by_cond.get(cond_key, [])
        if not recs:
            continue
        n       = len(recs)
        corr    = sum(1 for r in recs if r["correct"])
        lats    = [r["latency_ms"] for r in recs if r["latency_ms"] is not None]
        m_lat   = mean(lats)
        ci_lat  = ci95(lats)
        lo, hi  = wilson_ci(corr, n)
        c_pct   = 100*corr/n if n else 0

        print(f"{cid:<4} {label:<28} {n:>5} {c_pct:>8.1f}% "
              f"  [{100*lo:.1f}%, {100*hi:.1f}%]  "
              f"{m_lat:>9.0f}ms {ci_lat:>7.0f}ms")

        results[cond_key] = {
            "n": n, "corr": corr, "lats": lats,
            "m_lat": m_lat, "c_pct": c_pct, "lo": lo, "hi": hi
        }

    # ── Statistical comparisons ──────────────────────────────────────────────
    section("STATISTICAL TESTS")

    # A1: No cache vs full-system warm
    # (Compare A1 latency against what we know from full benchmark)
    a1 = results.get("A1_no_cache")
    if a1:
        print(f"\nA1 — No Cache:")
        print(f"  Mean latency: {a1['m_lat']:.0f} ms  (full-system warm was 1,095 ms)")
        print(f"  Correctness:  {a1['c_pct']:.1f}%  (full-system warm was 83.0%)")
        print(f"  Note: All calls are cold synthesis (LLM every time)")

    # A3: No repair vs with repair
    a3_no  = results.get("A3_no_repair")
    a3_yes = results.get("A3_with_repair")
    if a3_no and a3_yes:
        print(f"\nA3 — Repair Loop Contribution:")
        print(f"  Without repair: {a3_no['corr']}/{a3_no['n']} = {a3_no['c_pct']:.1f}%")
        print(f"  With repair:    {a3_yes['corr']}/{a3_yes['n']} = {a3_yes['c_pct']:.1f}%")
        z, p = two_prop_z(a3_no["corr"], a3_no["n"],
                          a3_yes["corr"], a3_yes["n"])
        if z is not None:
            print(f"  Two-prop z-test: z={z:.3f}, p={p:.4f}")
        t, p_t, df = welch_t(a3_no["lats"], a3_yes["lats"])
        d = cohens_d(a3_no["lats"], a3_yes["lats"])
        if t is not None:
            print(f"  Welch's t (latency): t({df:.1f})={t:.2f}, p={p_t:.4f}, d={d:.2f}")

    # A5: MCP overhead
    a5r = results.get("A5_rest")
    a5m = results.get("A5_mcp")
    if a5r and a5m:
        print(f"\nA5 — MCP Protocol Overhead:")
        print(f"  Direct REST:  {a5r['m_lat']:.0f} ms")
        print(f"  MCP-routed:   {a5m['m_lat']:.0f} ms")
        overhead = a5m["m_lat"] - a5r["m_lat"]
        print(f"  MCP overhead: {overhead:.0f} ms ({overhead/a5r['m_lat']*100:.1f}%)")
        t, p, df = welch_t(a5m["lats"], a5r["lats"])
        if t is not None:
            d = cohens_d(a5m["lats"], a5r["lats"])
            print(f"  Welch's t: t({df:.1f})={t:.2f}, p={p:.4f}, d={d:.2f}")

    # A6: Persistence effect
    a6s1c = results.get("A6_no_persistence_s1_cold")
    a6s1w = results.get("A6_no_persistence_s1_warm")
    a6s2c = results.get("A6_no_persistence_s2_cold")
    if a6s1c and a6s1w and a6s2c:
        print(f"\nA6 — Persistence Effect:")
        print(f"  Session 1 cold (first synthesis): {a6s1c['m_lat']:.0f} ms")
        print(f"  Session 1 warm (cached):          {a6s1w['m_lat']:.0f} ms")
        print(f"  Session 2 cold (post-restart):    {a6s2c['m_lat']:.0f} ms")
        print(f"  S1 warm vs S2 cold latency gap: "
              f"{a6s2c['m_lat'] - a6s1w['m_lat']:.0f} ms")
        print(f"  Interpretation: Without persistence, post-restart calls "
              f"pay full synthesis cost again.")

    # ── LaTeX Table ──────────────────────────────────────────────────────────
    section("LATEX TABLE VI — ABLATION RESULTS (paste into paper)")

    print(r"""
\begin{table}[htbp]
  \centering
  \caption{Ablation Study Results (30 /run Tasks, 10 /build Tasks)}
  \label{tab:ablation}
  \renewcommand{\arraystretch}{1.2}
  \begin{tabular}{llccc}
    \toprule
    \textbf{ID} & \textbf{Condition} & \textbf{$n$}
      & \textbf{Correct} & \textbf{Mean Lat.\ (ms)} \\
    \midrule""")

    rows = [
        ("A1", "Without Cache",         "A1_no_cache"),
        ("A2", "Without Sandbox",       "A2_no_sandbox"),
        ("A3", "Without Repair Loop",   "A3_no_repair"),
        ("A4", "Without Heuristics",    "A4_no_heuristic"),
        ("A5", "MCP vs REST overhead",  None),
        ("A6", "Without Persistence",   None),
    ]

    for rid, label, key in rows:
        r = results.get(key)
        if r:
            corr_str = f"{r['corr']}/{r['n']} ({r['c_pct']:.1f}\\%)"
            lat_str  = f"{r['m_lat']:.0f}"
            n_str    = str(r["n"])
        else:
            corr_str = "---"
            lat_str  = "---"
            n_str    = "---"
        print(f"    {rid} & {label} & {n_str} & {corr_str} & {lat_str} \\\\")

    print(r"""    \midrule
    \multicolumn{3}{l}{Full system (warm, from Table~\ref{tab:latency})}
      & 83.0\% & 1{,}095 \\
    \bottomrule
    \multicolumn{5}{l}{\footnotesize $n$: trials.
      Full system result from Section~\ref{sec:evaluation}.}
  \end{tabular}
\end{table}""")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_ablation.py results/ablation_*.csv")
        sys.exit(1)

    paths = []
    for pattern in sys.argv[1:]:
        paths.extend(glob.glob(pattern))
    paths = sorted(set(paths))

    if not paths:
        print(f"No files found matching: {sys.argv[1:]}")
        sys.exit(1)

    print(f"Loading {len(paths)} file(s)...")
    for p in paths:
        print(f"  {p}")

    records = load_records(paths)
    print(f"Total records: {len(records)}")

    analyze(records)


if __name__ == "__main__":
    main()
