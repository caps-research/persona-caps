"""Ad-hoc summary printer for the audit-loop sweep state. Throwaway."""

from __future__ import annotations

import json
import glob
from collections import defaultdict
from pathlib import Path


def parse_name(stem: str) -> tuple[str, int, int] | None:
    """`high_neuroticism__k15_s1_targeted` -> (cell, k, seed)."""
    parts = stem.split("__")
    if len(parts) < 2:
        return None
    cell = parts[0]
    ks = parts[1].split("_")
    try:
        k = int(ks[0].lstrip("k"))
        s = int(ks[1].lstrip("s"))
    except Exception:
        return None
    return cell, k, s


def main() -> None:
    san_rows = []
    for f in sorted(glob.glob("runs/audit_loop/sanity/*_targeted.json")):
        d = json.loads(Path(f).read_text())
        parsed = parse_name(Path(f).stem)
        if not parsed:
            continue
        cell, k, seed = parsed
        if "score_shift" not in d:
            continue
        san_rows.append({
            "cell": cell, "k": k, "seed": seed,
            "clean": d["clean_dimension_scores"].get(d["targeted_dim"]),
            "poll": d["poll_dimension_scores"].get(d["targeted_dim"]),
            "shift": d["score_shift"],
            "dm": d["direction_match_rate"],
            "pass": d["pass_overall"],
        })

    by_ck = defaultdict(list)
    for r in san_rows:
        by_ck[(r["cell"], r["k"])].append(r)

    print("=== CAPS sanity by (cell, K), n=3 seeds each ===")
    print(f"{'cell':24s} {'K':>2s}  pass    clean  poll   shift   dir-match")
    for (cell, k), entries in sorted(by_ck.items()):
        p = sum(1 for e in entries if e["pass"])
        ac = sum(e["clean"] for e in entries) / len(entries)
        ap = sum(e["poll"] for e in entries) / len(entries)
        sh = sum(e["shift"] for e in entries) / len(entries)
        dm = sum(e["dm"] for e in entries) / len(entries)
        flag = "  PASS" if p == len(entries) else ("  PART" if p > 0 else "  FAIL")
        print(f"{cell:24s} {k:>2d}   {p}/{len(entries)} {flag}  {ac:>5.2f}  {ap:>5.2f}  {sh:>5.2f}    {dm:>5.2f}")

    # Locator A on passing conditions
    passing = {(r["cell"], r["k"], r["seed"]) for r in san_rows if r["pass"]}
    la_rows = []
    for f in sorted(glob.glob("runs/audit_loop/locate_a/*_targeted.json")):
        parsed = parse_name(Path(f).stem)
        if not parsed:
            continue
        cell, k, seed = parsed
        if (cell, k, seed) not in passing:
            continue
        d = json.loads(Path(f).read_text())
        m = d["metrics_at_oracle_k"]
        la_rows.append({"cell": cell, "k": k, "p": m["precision"], "r": m["recall"], "f1": m["f1"], "n_sig": d["n_atoms_with_signal"]})

    print()
    print("=== Locator A (per-response attribution) on sanity-passing conds ===")
    print(f"{'cell':24s} {'K':>2s}   P    R    F1   atoms-with-signal")
    by_ck = defaultdict(list)
    for r in la_rows:
        by_ck[(r["cell"], r["k"])].append(r)
    for (cell, k), es in sorted(by_ck.items()):
        p = sum(e["p"] for e in es) / len(es)
        r = sum(e["r"] for e in es) / len(es)
        f = sum(e["f1"] for e in es) / len(es)
        n = sum(e["n_sig"] for e in es) / len(es)
        print(f"{cell:24s} {k:>2d}  {p:.2f} {r:.2f} {f:.2f}    {n:>5.1f}")

    # Locator B on passing conditions
    lb_rows = []
    for f in sorted(glob.glob("runs/audit_loop/locate_b/*_targeted.json")):
        parsed = parse_name(Path(f).stem)
        if not parsed:
            continue
        cell, k, seed = parsed
        if (cell, k, seed) not in passing:
            continue
        d = json.loads(Path(f).read_text())
        m = d["metrics"]
        lb_rows.append({"cell": cell, "k": k, "p": m["precision"], "r": m["recall"], "f1": m["f1"], "tp": m["tp"], "fp": m["fp"]})

    print()
    print("=== Locator B (corpus metadata sweep) on sanity-passing conds ===")
    print(f"{'cell':24s} {'K':>2s}   P    R    F1   TP  FP")
    by_ck = defaultdict(list)
    for r in lb_rows:
        by_ck[(r["cell"], r["k"])].append(r)
    for (cell, k), es in sorted(by_ck.items()):
        p = sum(e["p"] for e in es) / len(es)
        r = sum(e["r"] for e in es) / len(es)
        f = sum(e["f1"] for e in es) / len(es)
        tp = sum(e["tp"] for e in es) / len(es)
        fp = sum(e["fp"] for e in es) / len(es)
        print(f"{cell:24s} {k:>2d}  {p:.2f} {r:.2f} {f:.2f}  {tp:>3.1f} {fp:>3.1f}")

    # Summary counts
    n_pass = sum(1 for r in san_rows if r["pass"])
    print()
    print(f"=== Totals ===")
    print(f"  sanity files seen   : {len(san_rows)}")
    print(f"  sanity PASS         : {n_pass}")
    print(f"  locate_a files done : {len(la_rows)} (on passing conds)")
    print(f"  locate_b files done : {len(lb_rows)} (on passing conds)")


if __name__ == "__main__":
    main()
