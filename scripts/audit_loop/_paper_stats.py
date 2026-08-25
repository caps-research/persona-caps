"""Compute every number needed for paper section 4.7 from raw aggregate.json.

Run from repo root: python scripts/audit_loop/_paper_stats.py
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AGG = REPO / "runs" / "audit_loop" / "aggregate.json"
P2_MAN = REPO / "runs" / "audit_loop" / "p2_sweep_manifest.json"
SANITY_DIR = REPO / "runs" / "audit_loop" / "sanity"


def mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def fmt(x, n=2):
    return "-" if x is None or (isinstance(x, float) and x != x) else f"{x:.{n}f}"


def main():
    agg = json.load(AGG.open())["rows"]

    by_wc = defaultdict(list)
    for r in agg:
        by_wc[(r["workflow"], r["cell"])].append(r)

    cells = ["high_extraversion", "high_neuroticism", "low_agreeableness", "low_openness"]
    wflows = ["caps_a", "caps_b", "p2_oracle", "p2_blind", "p2_detect"]

    print("=" * 100)
    print("BLOCK 1: per (workflow, cell) means")
    print("=" * 100)
    print(f"{'workflow':10s} {'cell':22s} {'n':>3s} {'P':>6s} {'R':>6s} {'F1':>6s} {'fix_n/N':>9s} {'gap':>6s} {'off':>6s} {'edits':>7s} {'chars':>8s}")
    for w in wflows:
        for c in cells:
            v = by_wc.get((w, c), [])
            if not v:
                continue
            P = [r["p"] for r in v if r["p"] is not None]
            R = [r["r"] for r in v if r["r"] is not None]
            F = [r["f1"] for r in v if r["f1"] is not None]
            fx = [r["fix_success"] for r in v]
            tg = [r["target_gap"] for r in v]
            ot = [r["off_target_mean"] for r in v]
            ed = [r["atoms_edited"] for r in v]
            ch = [r["chars_edited"] for r in v if r["chars_edited"] >= 0]
            print(f"{w:10s} {c:22s} {len(v):>3d} "
                  f"{fmt(mean(P)) if P else '   -':>6s} "
                  f"{fmt(mean(R)) if R else '   -':>6s} "
                  f"{fmt(mean(F)) if F else '   -':>6s} "
                  f"{sum(fx):>3d}/{len(fx):<5d} "
                  f"{fmt(mean(tg)):>6s} "
                  f"{fmt(mean(ot)):>6s} "
                  f"{fmt(mean(ed),1):>7s} "
                  f"{fmt(mean(ch),0) if ch else '   -':>8s}")

    print()
    print("=" * 100)
    print("BLOCK 2: K-stratified means per (workflow, cell, K)")
    print("=" * 100)
    by_wck = defaultdict(list)
    for r in agg:
        by_wck[(r["workflow"], r["cell"], r["k"])].append(r)
    print(f"{'workflow':10s} {'cell':22s} {'K':>3s} {'n':>3s} {'P':>6s} {'R':>6s} {'fix':>6s} {'edits':>6s}")
    for w in ["caps_a", "caps_b"]:
        for c in cells:
            for K in [5, 10, 15]:
                v = by_wck.get((w, c, K), [])
                if not v:
                    continue
                P = [r["p"] for r in v if r["p"] is not None]
                R = [r["r"] for r in v if r["r"] is not None]
                fx = [r["fix_success"] for r in v]
                ed = [r["atoms_edited"] for r in v]
                print(f"{w:10s} {c:22s} {K:>3d} {len(v):>3d} "
                      f"{fmt(mean(P)) if P else '   -':>6s} "
                      f"{fmt(mean(R)) if R else '   -':>6s} "
                      f"{sum(fx)}/{len(fx):<3d}  "
                      f"{fmt(mean(ed),1):>6s}")

    print()
    print("=" * 100)
    print("BLOCK 3: P2 sanity pass / fail per cell from manifest")
    print("=" * 100)
    man = json.load(P2_MAN.open())
    conds = man.get("conditions", [])
    sn = defaultdict(lambda: [0, 0])  # cell -> [pass, total]
    for c in conds:
        cell = c["cell"]
        sn[cell][1] += 1
        if c.get("p2_sanity_passed"):
            sn[cell][0] += 1
    print(f"{'cell':22s} {'P2_sanity_pass / total':30s}")
    for cell in cells:
        p, t = sn.get(cell, [0, 0])
        print(f"{cell:22s} {p}/{t}")

    print()
    print("=" * 100)
    print("BLOCK 4: CAPS sanity per cell (score_shift mean / max, direction_match mean)")
    print("=" * 100)
    all_cells = [
        "high_extraversion", "low_extraversion",
        "high_agreeableness", "low_agreeableness",
        "high_conscientiousness", "low_conscientiousness",
        "high_neuroticism", "low_neuroticism",
        "high_openness", "low_openness",
    ]
    print(f"{'cell':25s} {'n':>3s} {'ss_mean':>8s} {'ss_max':>8s} {'dir_mean':>8s}")
    for c in all_cells:
        files = sorted(SANITY_DIR.glob(f"{c}__*_targeted.json"))
        if not files:
            continue
        ss = []
        dm = []
        for f in files:
            d = json.load(f.open())
            ss.append(d.get("score_shift", 0))
            dm.append(d.get("direction_match", 0))
        print(f"{c:25s} {len(ss):>3d} {fmt(mean(ss)):>8s} {fmt(max(ss)):>8s} {fmt(mean(dm)):>8s}")

    print()
    print("=" * 100)
    print("BLOCK 5: key takeaways verified")
    print("=" * 100)
    caps_a_e = by_wc.get(("caps_a", "high_extraversion"), [])
    caps_b_e = by_wc.get(("caps_b", "high_extraversion"), [])
    p2_o_e = by_wc.get(("p2_oracle", "high_extraversion"), [])
    p2_d_e = by_wc.get(("p2_detect", "high_extraversion"), [])

    if caps_a_e and p2_o_e:
        a_edits = mean([r["atoms_edited"] for r in caps_a_e])
        o_edits = mean([r["atoms_edited"] for r in p2_o_e])
        a_fix = sum(r["fix_success"] for r in caps_a_e) / len(caps_a_e)
        o_fix = sum(r["fix_success"] for r in p2_o_e) / len(p2_o_e)
        print(f"high_E: CAPS-A edits {a_edits:.1f} fix-rate {a_fix:.0%} "
              f"vs P2-Oracle edits {o_edits:.1f} fix-rate {o_fix:.0%} "
              f"(ratio: {a_edits/o_edits:.2f}x edits)")

    # CAPS-A precision per cell
    print("\nCAPS-A precision per cell (mean across K, seeds):")
    for c in cells:
        v = by_wc.get(("caps_a", c), [])
        if v:
            P = [r["p"] for r in v if r["p"] is not None]
            R = [r["r"] for r in v if r["r"] is not None]
            print(f"  {c:25s} P={mean(P):.2f}, R={mean(R):.2f}, n={len(v)}")

    print("\nCAPS-B precision per cell (mean across K, seeds):")
    for c in cells:
        v = by_wc.get(("caps_b", c), [])
        if v:
            P = [r["p"] for r in v if r["p"] is not None]
            R = [r["r"] for r in v if r["r"] is not None]
            print(f"  {c:25s} P={mean(P):.2f}, R={mean(R):.2f}, n={len(v)}")

    print("\nLow-O P2 sanity outcome:")
    lo_p2 = [c for c in conds if c["cell"] == "low_openness"]
    lo_pass = sum(c.get("p2_sanity_passed", False) for c in lo_p2)
    print(f"  P2 sanity pass {lo_pass}/{len(lo_p2)} on low_openness")

    print("\nHigh-N P2 sanity outcome:")
    hn_p2 = [c for c in conds if c["cell"] == "high_neuroticism"]
    hn_pass = sum(c.get("p2_sanity_passed", False) for c in hn_p2)
    print(f"  P2 sanity pass {hn_pass}/{len(hn_p2)} on high_neuroticism (at sanity-passing K=15)")


if __name__ == "__main__":
    main()
