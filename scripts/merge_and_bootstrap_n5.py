#!/usr/bin/env python3
"""Merge N=3 + N=2 RQ1/RQ4 runs into N=5 and compute bootstrap 95% CIs.

Sources:
  - N=3 originals:
      runs/exp01_beck_v2_full_N3.json            (MiniMax)
      runs/rq4/exp01_gpt4o_n3.json               (GPT-4o)
      runs/rq4/exp01_gemini_flash_n3.json        (Gemini-2.5-Flash)
  - N=2 extras (this run):
      runs/n5_extra/exp01_minimax_n2extra.json
      runs/n5_extra/exp01_gpt4o_n2extra.json
      runs/n5_extra/exp01_gemini_flash_n2extra.json

Outputs:
  runs/n5_extra/exp01_minimax_N5.json
  runs/n5_extra/exp01_gpt4o_N5.json
  runs/n5_extra/exp01_gemini_flash_N5.json
  runs/n5_extra/aggregate_n5.json
  runs/n5_extra/cross_vendor_summary_n5.json

Bootstrap method: 10,000 resamples WITH replacement of the per-run mean spreads
(N=5 → 5^5 = 3,125 unique resamples, but BCa-style with replacement is fine).
We report 95% CI as the [2.5%, 97.5%] percentile of resampled means.

Usage:  cd repos/persona-caps && python3 scripts/merge_and_bootstrap_n5.py
"""
from __future__ import annotations
import json
import random
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIMS = ["neuroticism", "extraversion", "conscientiousness", "agreeableness", "openness"]
METHODS = ["caps", "trait_label", "persona_description", "random_demos", "wrong_trait"]
POLARITIES = ["high", "low"]

PAIRS = [
    ("MiniMax-M2.7", "minimax",
     ROOT / "runs" / "exp01_beck_v2_full_N3.json",
     ROOT / "runs" / "n5_extra" / "exp01_minimax_n2extra.json",
     ROOT / "runs" / "n5_extra" / "exp01_minimax_N5.json"),
    ("GPT-4o-2024-08-06", "gpt-4o",
     ROOT / "runs" / "rq4" / "exp01_gpt4o_n3.json",
     ROOT / "runs" / "n5_extra" / "exp01_gpt4o_n2extra.json",
     ROOT / "runs" / "n5_extra" / "exp01_gpt4o_N5.json"),
    ("Gemini-2.5-Flash", "gemini-flash",
     ROOT / "runs" / "rq4" / "exp01_gemini_flash_n3.json",
     ROOT / "runs" / "n5_extra" / "exp01_gemini_flash_n2extra.json",
     ROOT / "runs" / "n5_extra" / "exp01_gemini_flash_N5.json"),
]

BOOT_N = 10_000
RNG = random.Random(20260425)


def merge(n3_path: Path, n2_path: Path, out_path: Path) -> dict:
    n3 = json.loads(n3_path.read_text())
    n2 = json.loads(n2_path.read_text())
    merged = {"config": dict(n3["config"]), "results": {}}
    merged["config"]["runs"] = 5
    merged["config"]["merged_from"] = [str(n3_path), str(n2_path)]
    for dim in DIMS:
        merged["results"][dim] = {}
        for pol in POLARITIES:
            merged["results"][dim][pol] = {}
            for method in METHODS:
                a = n3["results"][dim][pol].get(method, [])
                b = n2["results"][dim][pol].get(method, [])
                merged["results"][dim][pol][method] = list(a) + list(b)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    return merged


def per_run_target_spreads(results: dict, method: str) -> list[float]:
    """For each run index r, mean over the 5 dimensions of (high_target - low_target).
    The per-dim spread already isolates the *target* dimension (we read
    results[dim][pol][method][r]['dimension_scores'][dim])."""
    n_runs_min = min(
        len(results[dim][pol][method])
        for dim in DIMS for pol in POLARITIES
        if results[dim][pol].get(method)
    )
    out = []
    for r in range(n_runs_min):
        gaps = []
        for dim in DIMS:
            h = results[dim]["high"][method][r]["dimension_scores"][dim]
            l = results[dim]["low"][method][r]["dimension_scores"][dim]
            gaps.append(h - l)
        out.append(st.mean(gaps))
    return out


def per_dim_spread_runs(results: dict, method: str, dim: str) -> list[float]:
    """List of per-run (high - low) spreads on `dim`."""
    n_runs = min(len(results[dim]["high"][method]), len(results[dim]["low"][method]))
    return [
        results[dim]["high"][method][r]["dimension_scores"][dim]
        - results[dim]["low"][method][r]["dimension_scores"][dim]
        for r in range(n_runs)
    ]


def bootstrap_ci(values: list[float], n_boot: int = BOOT_N, alpha: float = 0.05):
    if not values:
        return (0.0, 0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0], values[0])
    means = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[RNG.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return (st.mean(values), lo, hi)


def fmt_ci(mean_lo_hi):
    m, lo, hi = mean_lo_hi
    return f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def main() -> None:
    summary = {"vendors": [], "per_vendor": {}}
    spread_per_cell: dict[str, float] = {}

    for label, alias, n3_path, n2_path, out_path in PAIRS:
        print(f"\n=== merging {label} ===")
        merged = merge(n3_path, n2_path, out_path)
        results = merged["results"]
        n_runs_caps = len(results[DIMS[0]]["high"]["caps"])
        print(f"  -> {out_path.name}  (caps n_runs = {n_runs_caps})")

        vendor_summary = {}
        for method in METHODS:
            per_run = per_run_target_spreads(results, method)
            mean_spread, ci_lo, ci_hi = bootstrap_ci(per_run)
            sd = st.stdev(per_run) if len(per_run) >= 2 else 0.0
            print(f"  {method:22s} N={len(per_run)}  mean spread = {mean_spread:+.2f}  "
                  f"95% CI [{ci_lo:+.2f}, {ci_hi:+.2f}]  SD={sd:.2f}")
            vendor_summary[method] = {
                "n_runs": len(per_run),
                "per_run": per_run,
                "mean_spread": mean_spread,
                "ci95_lo": ci_lo,
                "ci95_hi": ci_hi,
                "sd_across_runs": sd,
            }
            for dim in DIMS:
                runs = per_dim_spread_runs(results, method, dim)
                if runs:
                    spread_per_cell[f"{label}|{method}|{dim}"] = round(st.mean(runs), 4)
        summary["vendors"].append(label)
        summary["per_vendor"][label] = vendor_summary

    # Cross-vendor Pearson r matrix on per-(method,dim) mean spread.
    def vec(label):
        return [spread_per_cell[f"{label}|{m}|{d}"] for m in METHODS for d in DIMS
                if f"{label}|{m}|{d}" in spread_per_cell]
    def pearson(xs, ys):
        n = len(xs)
        if n < 2:
            return None
        mx, my = sum(xs)/n, sum(ys)/n
        num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
        dx = (sum((x-mx)**2 for x in xs)) ** 0.5
        dy = (sum((y-my)**2 for y in ys)) ** 0.5
        if dx == 0 or dy == 0:
            return None
        return num / (dx * dy)

    rmat = {}
    for a in summary["vendors"]:
        rmat[a] = {}
        for b in summary["vendors"]:
            rmat[a][b] = pearson(vec(a), vec(b))
    summary["pearson_r_per_method_dim"] = rmat
    summary["spread_per_cell"] = spread_per_cell

    out_summary = ROOT / "runs" / "n5_extra" / "cross_vendor_summary_n5.json"
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[ok] wrote -> {out_summary}")

    out_agg = ROOT / "runs" / "n5_extra" / "aggregate_n5.json"
    out_agg.write_text(json.dumps(summary["per_vendor"], indent=2, ensure_ascii=False))
    print(f"[ok] wrote -> {out_agg}")

    print("\n=== Cross-vendor Pearson r on per-(method,dim) mean spread (N=5) ===")
    print("           " + "  ".join(f"{v[:14]:>14}" for v in summary["vendors"]))
    for a in summary["vendors"]:
        row = [f"{a[:10]:<10}"]
        for b in summary["vendors"]:
            r = rmat[a][b]
            row.append(f"{r:+.3f}".rjust(15) if r is not None else "         n/a")
        print("  " + "".join(row))


if __name__ == "__main__":
    main()
