"""Per-cell × K × workflow breakdown for the final report."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs" / "audit_loop"


def _fmt(x, fmt):
    return "-" if x is None else format(x, fmt)


def main() -> None:
    rows = list(csv.DictReader(open(OUT / "aggregate.csv")))
    for r in rows:
        for k in ("p", "r", "f1", "target_gap", "dir_match_vs_clean", "off_target_mean"):
            r[k] = float(r[k]) if r[k] not in (None, "", "None") else None
        for k in ("k", "atoms_edited", "chars_edited"):
            r[k] = int(float(r[k])) if r[k] not in (None, "", "None") else None
        r["fix_success"] = r["fix_success"] == "True"

    # P² sanity outcome counts
    p2_san_files = list((OUT / "p2_sanity").glob("*_targeted.json"))
    p2_pass = 0
    p2_fail_by_cell = defaultdict(int)
    for f in p2_san_files:
        d = json.loads(f.read_text())
        if d.get("pass_overall"):
            p2_pass += 1
        else:
            cell = f.stem.split("__")[0]
            p2_fail_by_cell[cell] += 1
    print(f"=== P² sanity (on the 20 CAPS-passing conditions) ===")
    print(f"  PASS: {p2_pass}/{len(p2_san_files)}")
    for cell, n in sorted(p2_fail_by_cell.items()):
        print(f"  FAIL cell={cell}: {n}")

    # Per (cell, K, workflow): mean
    print()
    print("=== Per-condition workflow comparison (cell × K × workflow) ===")
    by_key = defaultdict(list)
    for r in rows:
        by_key[(r["cell"], r["k"], r["workflow"])].append(r)
    print(f"{'cell':24s} {'K':>2s} {'workflow':12s}  n  {'P':>4s} {'R':>4s}  atoms  chars  {'gap':>4s} {'dm':>4s} {'off':>4s}  succ")
    last_cell = None
    last_k = None
    for (cell, k, wf), es in sorted(by_key.items()):
        if cell != last_cell:
            print()
            last_cell = cell
        if k != last_k:
            last_k = k
        n = len(es)
        mp = statistics.mean([e["p"] for e in es if e["p"] is not None]) if any(e["p"] is not None for e in es) else None
        mr = statistics.mean([e["r"] for e in es if e["r"] is not None]) if any(e["r"] is not None for e in es) else None
        ma = statistics.mean([e["atoms_edited"] for e in es if e["atoms_edited"] is not None]) if any(e["atoms_edited"] is not None for e in es) else None
        mc = statistics.mean([e["chars_edited"] for e in es if e["chars_edited"] is not None]) if any(e["chars_edited"] is not None for e in es) else None
        mg = statistics.mean([e["target_gap"] for e in es if e["target_gap"] is not None])
        mdm = statistics.mean([e["dir_match_vs_clean"] for e in es if e["dir_match_vs_clean"] is not None]) if any(e["dir_match_vs_clean"] is not None for e in es) else None
        mo = statistics.mean([e["off_target_mean"] for e in es if e["off_target_mean"] is not None])
        succ = sum(1 for e in es if e["fix_success"]) / n
        print(f"{cell:24s} {k:>2d} {wf:12s}  {n}  {_fmt(mp,'>4.2f'):>4s} {_fmt(mr,'>4.2f'):>4s}  {_fmt(ma,'>4.1f'):>5s}  {_fmt(mc,'>4.0f'):>5s}  {mg:>4.2f} {_fmt(mdm,'>4.2f'):>4s} {mo:>4.2f}  {succ:>3.0%}")

    # Cross-workflow summary at K=15 only (the main success regime)
    print()
    print("=== Workflow summary at K=15 (cross-cell mean) ===")
    rows15 = [r for r in rows if r["k"] == 15]
    by_wf = defaultdict(list)
    for r in rows15:
        by_wf[r["workflow"]].append(r)
    print(f"{'workflow':12s}  n  {'P':>4s} {'R':>4s}  atoms  chars   {'gap':>4s} {'dm':>4s} {'off':>4s}  succ")
    for wf in ("caps_a", "caps_b", "p2_oracle", "p2_detect", "p2_blind"):
        es = by_wf.get(wf, [])
        if not es:
            continue
        n = len(es)
        mp = statistics.mean([e["p"] for e in es if e["p"] is not None]) if any(e["p"] is not None for e in es) else None
        mr = statistics.mean([e["r"] for e in es if e["r"] is not None]) if any(e["r"] is not None for e in es) else None
        ma = statistics.mean([e["atoms_edited"] for e in es if e["atoms_edited"] is not None]) if any(e["atoms_edited"] is not None for e in es) else None
        mc = statistics.mean([e["chars_edited"] for e in es if e["chars_edited"] is not None]) if any(e["chars_edited"] is not None for e in es) else None
        mg = statistics.mean([e["target_gap"] for e in es])
        mdm = statistics.mean([e["dir_match_vs_clean"] for e in es if e["dir_match_vs_clean"] is not None]) if any(e["dir_match_vs_clean"] is not None for e in es) else None
        mo = statistics.mean([e["off_target_mean"] for e in es])
        succ = sum(1 for e in es if e["fix_success"]) / n
        print(f"{wf:12s}  {n}  {_fmt(mp,'>4.2f'):>4s} {_fmt(mr,'>4.2f'):>4s}  {_fmt(ma,'>4.1f'):>5s}  {_fmt(mc,'>4.0f'):>5s}   {mg:>4.2f} {_fmt(mdm,'>4.2f'):>4s} {mo:>4.2f}  {succ:>3.0%}")


if __name__ == "__main__":
    main()
