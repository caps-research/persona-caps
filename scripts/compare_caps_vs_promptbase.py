#!/usr/bin/env python
"""Compare CAPS pi_d sweep against the continuous-prompt baseline.

Three diagnostics that distinguish CAPS from the natural mixed-percentage
prompt baseline ("You are a person whose Big Five {dim} is X% high and
(100-X)% low") on the same 3 vendors x 5 dims x 5 weights x N=3 protocol:

  1. Target-spread per-cell (w=1.00 - w=0.00)
  2. Calibration error: |empirical_mean - linear_ideal| at intermediate w
     where linear_ideal(w) = 1 + 4*w.
  3. Off-target leakage at w=1.00: how much do the OTHER 4 dims drift
     from neutral (3.0) when the persona is fully on the target dim?

Output:
  - Console table for paper §4.5 prose
  - JSON of every per-cell number for the appendix
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean as _mean


DIMS = [("n", "neuroticism", "N"), ("e", "extraversion", "E"),
        ("c", "conscientiousness", "C"), ("a", "agreeableness", "A"),
        ("o", "openness", "O")]
VENDORS = [("gem", "Gem"), ("g4o", "G4o"), ("mm", "MM")]


def load_summary(method: str, vendor: str, short: str) -> dict:
    if method == "caps":
        p = Path(f"runs/pi_sweep_{short}_{vendor}/summary.json")
    else:
        p = Path(f"runs/pi_sweep_{short}_{vendor}_promptbase_mixed/summary.json")
    return json.loads(p.read_text())


def per_w_means(summary: dict) -> list[float]:
    return [a["mean"] for a in summary["agg"]]


def calibration_err(means: list[float]) -> float:
    """Mean |observed - linear_ideal| over intermediate weights {0.25, 0.5, 0.75}."""
    ideal = [1.0 + 4 * w for w in [0.0, 0.25, 0.5, 0.75, 1.0]]
    return _mean(abs(means[i] - ideal[i]) for i in [1, 2, 3])


def off_target_leak(summary: dict, target_dim: str) -> float:
    """At w=1.00, mean |score - 3.0| over the OTHER 4 Big Five dims, averaged
    over the N runs at that weight. Lower = better (no spillover)."""
    by_dim = {d: [] for d in
              ["neuroticism", "extraversion", "conscientiousness",
               "agreeableness", "openness"] if d != target_dim}
    for r in summary["rows"]:
        if r["w_high"] != 1.0:
            continue
        for d, v in r.get("all_dim_scores", {}).items():
            if d == target_dim or v is None:
                continue
            by_dim[d].append(abs(v - 3.0))
    # average within each off-dim, then across off-dims
    per_dim = []
    for d, vs in by_dim.items():
        if vs:
            per_dim.append(_mean(vs))
    return _mean(per_dim) if per_dim else float("nan")


def main():
    rows = []
    for vt, vn in VENDORS:
        for short, dim, sd in DIMS:
            try:
                s_c = load_summary("caps", vt, short)
                s_p = load_summary("promptbase", vt, short)
            except FileNotFoundError as e:
                print(f"  MISSING {e.filename}")
                continue
            mc = per_w_means(s_c)
            mp = per_w_means(s_p)
            spread_c = round(mc[-1], 2) - round(mc[0], 2)
            spread_p = round(mp[-1], 2) - round(mp[0], 2)
            cal_c = calibration_err(mc)
            cal_p = calibration_err(mp)
            off_c = off_target_leak(s_c, dim)
            off_p = off_target_leak(s_p, dim)
            rows.append({
                "vendor": vn, "dim": sd, "dim_full": dim,
                "spread_caps": spread_c, "spread_prompt": spread_p,
                "calibration_err_caps": cal_c, "calibration_err_prompt": cal_p,
                "off_target_leak_caps": off_c, "off_target_leak_prompt": off_p,
            })

    # 1. Spread
    print("\n=== Spread comparison (w=1.00 minus w=0.00) ===")
    print(f"{'vendor':>4} {'dim':>4}  {'CAPS':>6} {'Prompt':>6}  {'CAPS-Prompt':>11}")
    by_v = {}
    for r in rows:
        by_v.setdefault(r["vendor"], []).append(r)
    for v, vrs in by_v.items():
        for r in vrs:
            print(f"{v:>4} {r['dim']:>4}  {r['spread_caps']:+6.2f} {r['spread_prompt']:+6.2f}"
                  f"  {r['spread_caps']-r['spread_prompt']:+6.2f}")
        mc = _mean(r["spread_caps"] for r in vrs)
        mp = _mean(r["spread_prompt"] for r in vrs)
        print(f"{v:>4} mean  {mc:+6.2f} {mp:+6.2f}  {mc-mp:+6.2f}")
        print()

    # 2. Calibration error
    print("=== Calibration: mean |observed - (1+4w)| at intermediate w in {0.25,0.5,0.75} ===")
    print(f"(lower = closer to LINEAR ideal, NOT necessarily better. See §4.5 prose.)")
    print(f"{'vendor':>4} {'dim':>4}  {'CAPS':>6} {'Prompt':>6}")
    for v, vrs in by_v.items():
        for r in vrs:
            print(f"{v:>4} {r['dim']:>4}  {r['calibration_err_caps']:6.2f} {r['calibration_err_prompt']:6.2f}")
        mc = _mean(r["calibration_err_caps"] for r in vrs)
        mp = _mean(r["calibration_err_prompt"] for r in vrs)
        print(f"{v:>4} mean  {mc:6.2f} {mp:6.2f}")
        print()

    # 3. Off-target leakage at w=1.00
    print("=== Off-target leakage at w=1.00 (mean |score-3.0| over the 4 OFF-target dims) ===")
    print(f"(lower = persona stays on target, doesn't spill into other dims)")
    print(f"{'vendor':>4} {'target':>6}  {'CAPS':>6} {'Prompt':>6}  {'Prompt-CAPS':>11}")
    for v, vrs in by_v.items():
        for r in vrs:
            print(f"{v:>4} {r['dim']:>6}  {r['off_target_leak_caps']:6.2f} {r['off_target_leak_prompt']:6.2f}"
                  f"  {r['off_target_leak_prompt']-r['off_target_leak_caps']:+6.2f}")
        mc = _mean(r["off_target_leak_caps"] for r in vrs)
        mp = _mean(r["off_target_leak_prompt"] for r in vrs)
        print(f"{v:>4} mean  {mc:6.2f} {mp:6.2f}  {mp-mc:+6.2f}")
        print()

    # Summary stats
    print("=== Cross-vendor summary ===")
    overall_spread_c = _mean(r["spread_caps"] for r in rows)
    overall_spread_p = _mean(r["spread_prompt"] for r in rows)
    overall_cal_c = _mean(r["calibration_err_caps"] for r in rows)
    overall_cal_p = _mean(r["calibration_err_prompt"] for r in rows)
    overall_off_c = _mean(r["off_target_leak_caps"] for r in rows)
    overall_off_p = _mean(r["off_target_leak_prompt"] for r in rows)
    print(f"Mean target spread:      CAPS {overall_spread_c:+.2f}  Prompt {overall_spread_p:+.2f}  diff {overall_spread_c-overall_spread_p:+.2f}")
    print(f"Mean calibration error:  CAPS {overall_cal_c:.2f}    Prompt {overall_cal_p:.2f}    diff {overall_cal_c-overall_cal_p:+.2f}")
    print(f"Mean off-target leakage: CAPS {overall_off_c:.2f}    Prompt {overall_off_p:.2f}    diff {overall_off_p-overall_off_c:+.2f} (pos = baseline leaks more)")

    out = Path("runs/stats/caps_vs_prompt_baseline.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "rows": rows,
        "summary": {
            "mean_spread_caps": overall_spread_c,
            "mean_spread_prompt": overall_spread_p,
            "mean_calibration_err_caps": overall_cal_c,
            "mean_calibration_err_prompt": overall_cal_p,
            "mean_off_target_leakage_caps": overall_off_c,
            "mean_off_target_leakage_prompt": overall_off_p,
        },
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
