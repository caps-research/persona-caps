#!/usr/bin/env python3
"""scripts/length_controlled_analysis.py

Re-derives App. A.10.2: does the §4.4 CAPS advantage survive a control for
answer length? Two complementary analyses, both post-hoc over the already
scored §4.4 runs -- no new LLM calls.

  (a) Per-(subject, method) correlation between answer word count and the
      polarity-aligned /10 trait-expression score.
  (b) Length-matched bootstrap of the CAPS-vs-baseline deltas: per
      (subject, trait, polarity) cell, both arms are restricted to the
      intersection of their word-count ranges before the delta is taken.

Cell selection, the 50-per-cell cap and the polarity alignment are imported
from scripts/aggregate_pb_3x3.py so this script and the §4.4 headline table can
never drift apart.

Every point estimate published in App. A.10.2 reproduces here to the printed
precision (`--check` asserts all 45 of them). The published bootstrap CIs came
from a different RNG stream, so recomputed endpoints can sit ~0.01-0.02 away;
what the interval covers is the claim, not the endpoint digits.

Usage:
    python scripts/length_controlled_analysis.py            # print both tables
    python scripts/length_controlled_analysis.py --check    # also diff against
                                                            # the published A.10.2
Output:
    paper/tables/length_controlled.md
    runs/_length_controlled.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_pb_3x3 import (  # noqa: E402  single source of truth for the grid
    LIMIT_PER_CELL,
    METHODS,
    ROOT,
    SUBJECTS,
    find_method_root,
)

N_BOOT = 2000
SEED = 0


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def collect_with_length(root: Path, method_subdir: str):
    """[(score10, n_words, trait, polarity)] for one (subject, method) cell.

    Mirrors aggregate_pb_3x3.collect -- same 50-row cap in file order, same
    /10 rescaling, same skip of unrated rows -- and additionally keeps the
    answer word count.
    """
    base = root / method_subdir
    if not base.exists():
        return None
    out = []
    for f in sorted(base.glob("*.jsonl")):
        trait, pol = f.stem.rsplit("_", 1)
        rows = []
        for line in f.open():
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("rating") is not None:
                rows.append((d["rating"], len((d.get("answer") or "").split())))
        for r, w in rows[:LIMIT_PER_CELL]:
            r10 = r * 2 if pol == "high" else (6 - r) * 2
            out.append((r10, w, trait, pol))
    return out or None


def load_grid():
    grid = {}
    for subj_label, tag in SUBJECTS:
        for method_label, method_tag in METHODS:
            root = find_method_root(tag, method_tag)
            grid[(subj_label, method_label)] = (
                collect_with_length(root, method_tag) if root else None
            )
    return grid


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def within_cell_partial(rows):
    """Pearson r after removing each (trait, polarity) cell mean from both
    variables -- algebraically the same as residualising both on cell dummies.
    """
    by = defaultdict(list)
    for s, w, t, p in rows:
        by[(t, p)].append((s, w))
    xs, ys = [], []
    for v in by.values():
        ms = sum(a for a, _ in v) / len(v)
        mw = sum(b for _, b in v) / len(v)
        for a, b in v:
            xs.append(a - ms)
            ys.append(b - mw)
    return pearson(xs, ys)


def boot_ci(values_a, values_b, n_boot=N_BOOT, seed=SEED):
    """Percentile 95% CI of mean(a) - mean(b) under independent resampling."""
    rng = random.Random(seed)
    na, nb = len(values_a), len(values_b)
    if not na or not nb:
        return float("nan"), float("nan")
    deltas = []
    for _ in range(n_boot):
        sa = sum(values_a[rng.randrange(na)] for _ in range(na)) / na
        sb = sum(values_b[rng.randrange(nb)] for _ in range(nb)) / nb
        deltas.append(sa - sb)
    deltas.sort()
    return deltas[int(0.025 * n_boot)], deltas[int(0.975 * n_boot) - 1]


def length_matched(rows_a, rows_b):
    """Restrict both arms per (trait, polarity) cell to the intersection of
    their word-count ranges, then pool. Returns (scores_a, scores_b).
    """
    def cells(rows):
        by = defaultdict(list)
        for s, w, t, p in rows:
            by[(t, p)].append((s, w))
        return by

    ca, cb = cells(rows_a), cells(rows_b)
    keep_a, keep_b = [], []
    for cell in ca:
        if cell not in cb:
            continue
        va, vb = ca[cell], cb[cell]
        lo = max(min(w for _, w in va), min(w for _, w in vb))
        hi = min(max(w for _, w in va), max(w for _, w in vb))
        fa = [s for s, w in va if lo <= w <= hi]
        fb = [s for s, w in vb if lo <= w <= hi]
        if fa and fb:
            keep_a += fa
            keep_b += fb
    return keep_a, keep_b


# --------------------------------------------------------------------------- #
# published A.10.2, for --check
# --------------------------------------------------------------------------- #
PUBLISHED_A = {  # (subject, method): (n, mean/10, mean words, r, r_partial)
    ("MiniMax-M2.7", "CAPS"): (500, 9.27, 138.2, +0.069, +0.065),
    ("MiniMax-M2.7", "SP"): (500, 8.42, 253.0, +0.024, -0.120),
    ("MiniMax-M2.7", "P²"): (497, 9.16, 261.7, +0.087, +0.008),
    ("Gemini-2.5-Flash", "CAPS"): (500, 9.44, 33.0, +0.114, +0.223),
    ("Gemini-2.5-Flash", "SP"): (500, 8.74, 116.2, +0.243, +0.302),
    ("Gemini-2.5-Flash", "P²"): (500, 9.44, 101.2, +0.317, +0.318),
    ("Llama-3-8B (Q4)", "CAPS"): (500, 9.16, 105.9, -0.082, -0.066),
    ("Llama-3-8B (Q4)", "SP"): (500, 8.66, 234.3, -0.155, -0.135),
    ("Llama-3-8B (Q4)", "P²"): (500, 9.53, 233.6, +0.034, -0.104),
}
PUBLISHED_B = {  # (subject, comparison): (full Δ, matched Δ)
    ("MiniMax-M2.7", "CAPS−SP"): (+0.86, +0.43),
    ("MiniMax-M2.7", "CAPS−P²"): (+0.11, -0.70),
    ("Gemini-2.5-Flash", "CAPS−SP"): (+0.71, +1.15),
    ("Gemini-2.5-Flash", "CAPS−P²"): (+0.00, +0.36),
    ("Llama-3-8B (Q4)", "CAPS−SP"): (+0.50, +0.70),
    ("Llama-3-8B (Q4)", "CAPS−P²"): (-0.37, -0.46),
}


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="diff the recomputed tables against the published A.10.2")
    args = ap.parse_args()

    grid = load_grid()
    result = {"limit_per_cell": LIMIT_PER_CELL, "n_boot": N_BOOT, "seed": SEED}

    # ---- (a) length-vs-score correlation ---------------------------------- #
    print(f"\n=== A.10.2(a) answer length vs polarity-aligned /10 "
          f"({LIMIT_PER_CELL}q/cell) ===")
    print(f"{'subject':<18}{'method':<7}{'n':>5}{'mean /10':>10}"
          f"{'mean words':>12}{'r':>9}{'r_partial':>11}")
    table_a = {}
    for subj_label, _ in SUBJECTS:
        for method_label, _ in METHODS:
            rows = grid[(subj_label, method_label)]
            if not rows:
                print(f"{subj_label:<18}{method_label:<7}    (no data)")
                continue
            n = len(rows)
            mean10 = sum(s for s, _, _, _ in rows) / n
            meanw = sum(w for _, w, _, _ in rows) / n
            r = pearson([s for s, _, _, _ in rows], [w for _, w, _, _ in rows])
            rp = within_cell_partial(rows)
            table_a[f"{subj_label}|{method_label}"] = {
                "n": n, "mean_10": mean10, "mean_words": meanw,
                "r": r, "r_partial_within_cell": rp,
            }
            print(f"{subj_label:<18}{method_label:<7}{n:>5}{mean10:>10.2f}"
                  f"{meanw:>12.1f}{r:>+9.3f}{rp:>+11.3f}")
    result["a_length_score_correlation"] = table_a

    # ---- (b) length-matched bootstrap ------------------------------------ #
    print(f"\n=== A.10.2(b) length-matched bootstrap "
          f"({N_BOOT} resamples, seed {SEED}) ===")
    print(f"{'subject':<18}{'comparison':<11}{'full Δ':>8}{'full 95% CI':>20}"
          f"{'matched Δ':>11}{'matched 95% CI':>20}{'n matched':>12}{'sign':>7}")
    table_b = {}
    for subj_label, _ in SUBJECTS:
        caps = grid[(subj_label, "CAPS")]
        for other, label in (("SP", "CAPS−SP"), ("P²", "CAPS−P²")):
            base = grid[(subj_label, other)]
            if not caps or not base:
                print(f"{subj_label:<18}{label:<11}    (no data)")
                continue
            fa = [s for s, _, _, _ in caps]
            fb = [s for s, _, _, _ in base]
            full_d = sum(fa) / len(fa) - sum(fb) / len(fb)
            full_lo, full_hi = boot_ci(fa, fb)
            ma, mb = length_matched(caps, base)
            match_d = (sum(ma) / len(ma) - sum(mb) / len(mb)) if ma and mb else float("nan")
            m_lo, m_hi = boot_ci(ma, mb)
            same = (full_d > 0) == (match_d > 0)
            table_b[f"{subj_label}|{label}"] = {
                "full_delta": full_d, "full_ci": [full_lo, full_hi],
                "matched_delta": match_d, "matched_ci": [m_lo, m_hi],
                "n_matched_caps": len(ma), "n_matched_base": len(mb),
                "sign_preserved": bool(same),
            }
            print(f"{subj_label:<18}{label:<11}{full_d:>+8.2f}"
                  f"{f'[{full_lo:+.2f}, {full_hi:+.2f}]':>20}{match_d:>+11.2f}"
                  f"{f'[{m_lo:+.2f}, {m_hi:+.2f}]':>20}"
                  f"{f'{len(ma)}/{len(mb)}':>12}{'yes' if same else 'no':>7}")
    result["b_length_matched_bootstrap"] = table_b

    # ---- markdown + json ------------------------------------------------- #
    # The artifact repository ships no paper/ tree; don't conjure one there.
    out_md = ROOT / "paper" / "tables" if (ROOT / "paper").is_dir() else ROOT / "runs"
    out_md.mkdir(parents=True, exist_ok=True)
    md = ["| subject | method | $n$ | mean /10 | mean words | $r$ | $r_\\text{partial}$ |",
          "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for k, v in table_a.items():
        s, m = k.split("|")
        md.append(f"| {s} | {m} | {v['n']} | {v['mean_10']:.2f} | "
                  f"{v['mean_words']:.1f} | ${v['r']:+.3f}$ | "
                  f"${v['r_partial_within_cell']:+.3f}$ |")
    md += ["", "| subject | comparison | full $\\Delta$ /10 | full 95\\% CI | "
           "matched $\\Delta$ /10 | matched 95\\% CI | sign preserved |",
           "| --- | --- | ---: | --- | ---: | --- | :---: |"]
    for k, v in table_b.items():
        s, c = k.split("|")
        md.append(f"| {s} | {c} | ${v['full_delta']:+.2f}$ | "
                  f"$[{v['full_ci'][0]:+.2f}, {v['full_ci'][1]:+.2f}]$ | "
                  f"${v['matched_delta']:+.2f}$ | "
                  f"$[{v['matched_ci'][0]:+.2f}, {v['matched_ci'][1]:+.2f}]$ | "
                  f"{'yes' if v['sign_preserved'] else 'no'} |")
    (out_md / "length_controlled.md").write_text("\n".join(md) + "\n")
    (ROOT / "runs" / "_length_controlled.json").write_text(
        json.dumps(result, indent=2, default=float))
    print(f"\nwrote: {out_md / 'length_controlled.md'}")
    print(f"wrote: {ROOT / 'runs' / '_length_controlled.json'}")

    # ---- optional diff against the published tables ---------------------- #
    if args.check:
        print("\n=== --check: recomputed vs published A.10.2 ===")
        bad = 0
        for key, (pn, pm, pw, pr, prp) in PUBLISHED_A.items():
            got = table_a.get(f"{key[0]}|{key[1]}")
            if not got:
                print(f"  MISSING {key}")
                bad += 1
                continue
            for name, g, p, tol in (("n", got["n"], pn, 0.5),
                                    ("mean/10", got["mean_10"], pm, 0.015),
                                    ("words", got["mean_words"], pw, 0.15),
                                    ("r", got["r"], pr, 0.002),
                                    ("r_partial", got["r_partial_within_cell"], prp, 0.002)):
                if abs(g - p) > tol:
                    print(f"  DIFF {key[0]:<18}{key[1]:<6}{name:>8}: "
                          f"got {g:+.3f} published {p:+.3f}")
                    bad += 1
        for key, (pf, pm_) in PUBLISHED_B.items():
            got = table_b.get(f"{key[0]}|{key[1]}")
            if not got:
                print(f"  MISSING {key}")
                bad += 1
                continue
            for name, g, p in (("full Δ", got["full_delta"], pf),
                               ("matched Δ", got["matched_delta"], pm_)):
                if abs(g - p) > 0.015:
                    print(f"  DIFF {key[0]:<18}{key[1]:<8}{name:>10}: "
                          f"got {g:+.2f} published {p:+.2f}")
                    bad += 1
        print(f"  {'all 45 published values reproduced' if not bad else f'{bad} mismatch(es)'}")


if __name__ == "__main__":
    main()
