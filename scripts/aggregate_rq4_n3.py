#!/usr/bin/env python3
"""Aggregate the GPT-4o + Gemini-Flash N=3 cross-vendor RQ4 runs and emit
mean ± SD per (method, dimension) plus the headline mean spread / SD across
runs, in the same shape as the §4.4 table in DRAFT.md.

Usage:
    cd repos/persona-caps
    python3 scripts/aggregate_rq4_n3.py

Reads:  runs/rq4/exp01_gpt4o_n3.json
        runs/rq4/exp01_gemini_flash_n3.json

Writes: runs/rq4/aggregate_n3.json
        (and prints a Markdown-flavoured table to stdout for paste into §4.4).
"""
from __future__ import annotations
import json
import statistics as st
from pathlib import Path

DIMS = ["neuroticism", "extraversion", "conscientiousness", "agreeableness", "openness"]
METHODS = ["caps", "trait_label", "persona_description", "random_demos", "wrong_trait"]
ROOT = Path(__file__).resolve().parent.parent
RUNS = {
    "GPT-4o (N=3)":           ROOT / "runs" / "rq4" / "exp01_gpt4o_n3.json",
    "Gemini-Flash (N=3)":     ROOT / "runs" / "rq4" / "exp01_gemini_flash_n3.json",
}


def per_method_dim_means(results, method: str):
    """Return list of (dim_mean_across_runs, dim_sd_across_runs) for each dim."""
    out = []
    for dim in DIMS:
        gaps = []
        for run_idx in range(len(results[dim]["high"][method])):
            high = results[dim]["high"][method][run_idx]["dimension_scores"][dim]
            low = results[dim]["low"][method][run_idx]["dimension_scores"][dim]
            gaps.append(high - low)
        m = st.mean(gaps)
        sd = st.stdev(gaps) if len(gaps) >= 2 else 0.0
        out.append((m, sd))
    return out


def per_method_run_means(results, method: str):
    """For each run, collapse 5 dims into one mean spread, then return run means."""
    n_runs = len(results[DIMS[0]]["high"][method])
    out = []
    for run_idx in range(n_runs):
        gaps = []
        for dim in DIMS:
            h = results[dim]["high"][method][run_idx]["dimension_scores"][dim]
            l = results[dim]["low"][method][run_idx]["dimension_scores"][dim]
            gaps.append(h - l)
        out.append(st.mean(gaps))
    return out


def fmt(m, sd):
    return f"{m:+.2f} ± {sd:.2f}"


report = {}
for label, path in RUNS.items():
    if not path.exists():
        print(f"[skip] {label}: {path} does not exist yet")
        continue
    payload = json.loads(path.read_text())
    res = payload["results"]
    print(f"\n=== {label}  ({path.name}) ===")
    print(f"{'method':<22} " + " ".join(f"{d[:5]:>13}" for d in DIMS) + "      mean       SD-across-runs")
    method_summary = {}
    for method in METHODS:
        per_dim = per_method_dim_means(res, method)
        per_run = per_method_run_means(res, method)
        run_mean = st.mean(per_run)
        run_sd = st.stdev(per_run) if len(per_run) >= 2 else 0.0
        cells = " ".join(f"{m:>+6.2f}±{sd:.2f}" for (m, sd) in per_dim)
        print(f"{method:<22} {cells}    {run_mean:+.2f}   {run_sd:.2f}")
        method_summary[method] = {
            "per_dim": [{"dim": d, "mean": m, "sd": sd} for d, (m, sd) in zip(DIMS, per_dim)],
            "run_means": per_run,
            "mean_spread": run_mean,
            "sd_across_runs": run_sd,
        }
    report[label] = method_summary

OUT = ROOT / "runs" / "rq4" / "aggregate_n3.json"
OUT.write_text(json.dumps(report, indent=2))
print(f"\n[ok] wrote -> {OUT}")
