#!/usr/bin/env python
"""Statistical defence of the §4.5 monotonicity claim.

For each (vendor, dim) cell of either the CAPS π_d sweep or the
prompt-baseline π sweep, this script computes:

  • Spearman ρ(w, mean_score) — rank correlation across 5 weight points
  • Permutation p-value: probability that a random permutation of the
    5 weights would yield a ρ as extreme as observed (1-tailed test
    for positive monotonicity). Exact null = 5! = 120 permutations.
  • Combined-cells permutation null: probability that, under the
    null of independent random orderings per cell, ALL N cells would
    simultaneously be strictly monotonic in the observed direction.
    For 15 cells with the "all 5 weights in increasing order" event
    of probability 1/120, this is (1/120)^15 ≈ 7e-32 (lower bound).
    We compute the actual product of per-cell tail probabilities.

Output: a small JSON + console summary used in §4.5 Statistical
defence note.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from statistics import mean


def spearman_rho(xs, ys):
    """Compute Spearman ρ. xs, ys: equal-length sequences."""
    def rank(seq):
        sorted_idx = sorted(range(len(seq)), key=lambda i: seq[i])
        ranks = [0.0] * len(seq)
        # Average ranks for ties
        i = 0
        while i < len(seq):
            j = i
            while j + 1 < len(seq) and seq[sorted_idx[j + 1]] == seq[sorted_idx[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = avg
            i = j + 1
        return ranks

    rx, ry = rank(xs), rank(ys)
    mx, my = mean(rx), mean(ry)
    num = sum((rxi - mx) * (ryi - my) for rxi, ryi in zip(rx, ry))
    dx = math.sqrt(sum((rxi - mx) ** 2 for rxi in rx))
    dy = math.sqrt(sum((ryi - my) ** 2 for ryi in ry))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def perm_p_value(xs, ys):
    """Exact 1-tailed permutation p for ρ(xs, ys) >= ρ_observed.

    With n=5 there are 5! = 120 permutations; we enumerate exactly.
    """
    rho_obs = spearman_rho(xs, ys)
    n = len(xs)
    perms = list(itertools.permutations(range(n)))
    count_ge = 0
    for perm in perms:
        ys_perm = [ys[i] for i in perm]
        rho_p = spearman_rho(xs, ys_perm)
        if rho_p >= rho_obs - 1e-9:
            count_ge += 1
    return count_ge / len(perms), rho_obs


def strict_monotone_up(ys, tol=0.0):
    """True iff ys is strictly increasing (ys[i+1] >= ys[i] + tol)."""
    return all(ys[i + 1] >= ys[i] + tol for i in range(len(ys) - 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["caps", "promptbase"], required=True,
                   help="caps = runs/pi_sweep_<short>_<vendor>/summary.json (full); "
                        "promptbase = runs/pi_sweep_<short>_<vendor>_promptbase_mixed/summary.json")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    DIMS = [("n", "neuroticism"), ("e", "extraversion"), ("c", "conscientiousness"),
            ("a", "agreeableness"), ("o", "openness")]
    VENDORS = ["gem", "g4o", "mm"]

    results = []
    for vendor in VENDORS:
        for short, dim in DIMS:
            if args.method == "caps":
                p_summary = Path(f"runs/pi_sweep_{short}_{vendor}/summary.json")
            else:
                p_summary = Path(f"runs/pi_sweep_{short}_{vendor}_promptbase_mixed/summary.json")
            if not p_summary.exists():
                print(f"  MISSING: {p_summary}")
                continue
            s = json.loads(p_summary.read_text())
            agg = s["agg"]
            ws = [a["w_high"] for a in agg]
            means = [a["mean"] for a in agg]
            p_val, rho = perm_p_value(ws, means)
            mono_strict = strict_monotone_up(means)
            mono_eps = strict_monotone_up(means, tol=-0.05)  # noise tolerance
            results.append({
                "vendor": vendor, "dim": dim,
                "ws": ws, "means": means,
                "spearman_rho": rho,
                "perm_p_one_sided": p_val,
                "strict_monotone": mono_strict,
                "monotone_with_0p05_tolerance": mono_eps,
            })

    print(f"\n=== {args.method.upper()}: per-cell Spearman ρ + 1-sided permutation p ===")
    print(f"{'vendor':>6} {'dim':>20} {'ρ':>7} {'p (1-sided)':>13} {'strict↑':>8} {'mono(±0.05)':>12}")
    n_strict = sum(1 for r in results if r["strict_monotone"])
    n_eps = sum(1 for r in results if r["monotone_with_0p05_tolerance"])
    for r in results:
        print(f"{r['vendor']:>6} {r['dim']:>20} {r['spearman_rho']:+.3f} {r['perm_p_one_sided']:>13.4f}"
              f"  {'✓' if r['strict_monotone'] else ' ':>6}    {'✓' if r['monotone_with_0p05_tolerance'] else ' ':>6}")
    n_total = len(results)
    print(f"\nTotal cells: {n_total}; strict monotonic: {n_strict}/{n_total};"
          f" monotone within 0.05 noise: {n_eps}/{n_total}")

    # Joint null: probability that ALL n cells would be strictly monotonic
    # under the IID null of "weights randomly ordered per cell". Each cell has
    # n!=120 permutations; only 1 is the strictly increasing order, so per-cell
    # P(strict↑) = 1/120 = 0.00833...
    # For independent cells: P(all strict↑) = (1/120)^k.
    if n_strict == n_total:
        joint_log10 = n_total * math.log10(1.0 / 120.0)
        print(f"Joint chance-null probability of {n_total}/{n_total} strictly monotonic"
              f" if weights were randomly ordered: 10^{joint_log10:.1f} ≈ {10**joint_log10:.2e}")

    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps({
            "method": args.method,
            "n_cells": n_total,
            "n_strict_monotone": n_strict,
            "n_monotone_within_0p05": n_eps,
            "results": results,
        }, indent=2))
        print(f"\nWrote {out_p}")


if __name__ == "__main__":
    main()
