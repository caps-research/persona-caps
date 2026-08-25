"""Aggregate the audit-loop sweep into a single table and a CSV ready
for §4.7's Table 6 / Figure 3.

For every (cell, K, seed) condition with sanity_passed = True we
collect five workflow rows:

* caps_a  -- CAPS Locator-A flag set + apply_fix
* caps_b  -- CAPS Locator-B flag set + apply_fix
* p2_oracle / p2_blind / p2_detect  -- the three P² workflows

Each row carries:
* localization P@K, R@K, F1@K   (Locator-A uses oracle K; Locator-B uses its full flag; P² oracle is trivially 1; P² blind is n/a)
* edit footprint (atoms / chars)
* fix-success bool
* targeted-dim |fixed - clean| gap
* mean off-target |fixed - clean| over the four non-targeted dims

Output: ``runs/audit_loop/aggregate.csv`` and ``aggregate.json``.
The CSV is the table; the JSON keeps the same data plus per-dim
off-target gaps for the appendix narrow table.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs" / "audit_loop"


def _read(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _row_from_caps_fix(fix_doc, locator_doc, *, workflow: str, cell: str, k: int, seed: int) -> dict:
    foot = fix_doc.get("footprint", {})
    loc_metrics = locator_doc.get("metrics_at_oracle_k") if workflow == "caps_a" else locator_doc.get("metrics")
    return {
        "cell": cell, "k": k, "seed": seed, "workflow": workflow,
        "p": loc_metrics.get("precision") if loc_metrics else None,
        "r": loc_metrics.get("recall") if loc_metrics else None,
        "f1": loc_metrics.get("f1") if loc_metrics else None,
        "atoms_edited": foot.get("atoms_restored"),
        "chars_edited": foot.get("chars_replaced"),
        "target_gap": fix_doc.get("target_gap"),
        "dir_match_vs_clean": fix_doc.get("direction_match_vs_clean"),
        "off_target_mean": fix_doc.get("off_target_gap_mean"),
        "off_target_per_dim": fix_doc.get("off_target_gap_per_dim", {}),
        "fix_success": fix_doc.get("fix_success"),
    }


def _row_from_p2_audit(p2_doc, *, cell: str, k: int, seed: int, strategy: str) -> dict:
    foot = p2_doc.get("footprint", {})
    loc = p2_doc.get("localization_metrics") or {}
    return {
        "cell": cell, "k": k, "seed": seed, "workflow": f"p2_{strategy}",
        "p": loc.get("precision"),
        "r": loc.get("recall"),
        "f1": loc.get("f1"),
        "atoms_edited": foot.get("atoms_restored"),
        "chars_edited": foot.get("chars_replaced"),
        "target_gap": p2_doc.get("target_gap"),
        "dir_match_vs_clean": p2_doc.get("direction_match_vs_clean"),
        "off_target_mean": p2_doc.get("off_target_gap_mean"),
        "off_target_per_dim": p2_doc.get("off_target_gap_per_dim", {}),
        "fix_success": p2_doc.get("fix_success"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--caps-manifest", default=str(OUT / "sweep_manifest.json"))
    p.add_argument("--csv", default=str(OUT / "aggregate.csv"))
    p.add_argument("--json", default=str(OUT / "aggregate.json"))
    args = p.parse_args()

    caps_man = json.loads(Path(args.caps_manifest).read_text(encoding="utf-8"))
    rows: list[dict] = []
    for cond in caps_man["conditions"]:
        if not cond.get("sanity_passed"):
            continue
        cell, k, seed = cond["cell"], cond["k"], cond["seed"]
        name = f"{cell}__k{k}_s{seed}_targeted"

        # CAPS A
        loc_a = _read(OUT / "locate_a" / f"{name}.json")
        fix_a = _read(OUT / "fix" / f"{name}__locatea.json")
        if loc_a and fix_a:
            rows.append(_row_from_caps_fix(fix_a, loc_a, workflow="caps_a", cell=cell, k=k, seed=seed))
        # CAPS B
        loc_b = _read(OUT / "locate_b" / f"{name}.json")
        fix_b = _read(OUT / "fix" / f"{name}__locateb.json")
        if loc_b and fix_b:
            rows.append(_row_from_caps_fix(fix_b, loc_b, workflow="caps_b", cell=cell, k=k, seed=seed))
        # P² three strategies
        for strat in ("oracle", "blind", "detect"):
            doc = _read(OUT / "p2_audit" / f"{name}__{strat}.json")
            if doc:
                rows.append(_row_from_p2_audit(doc, cell=cell, k=k, seed=seed, strategy=strat))

    # CSV
    fieldnames = [
        "cell", "k", "seed", "workflow",
        "p", "r", "f1",
        "atoms_edited", "chars_edited",
        "target_gap", "dir_match_vs_clean",
        "off_target_mean",
        "fix_success",
    ]
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})
    print(f"[aggregate] wrote {len(rows)} rows -> {args.csv}")

    # JSON (keeps per-dim off-target breakdown)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[aggregate] wrote -> {args.json}")

    # Summary table by (workflow, K), averaged across cells and seeds
    print()
    print("=== Summary by workflow x K (mean across cells and seeds) ===")
    by_key: dict[tuple[str, int], list[dict]] = {}
    for r in rows:
        by_key.setdefault((r["workflow"], r["k"]), []).append(r)
    header = ("workflow", "K", "n", "P", "R", "atoms", "chars", "gap", "off_t", "succ_rate")
    print(f"{header[0]:12s}  {header[1]:>3s}  {header[2]:>3s}  "
          f"{header[3]:>5s}  {header[4]:>5s}  {header[5]:>6s}  {header[6]:>7s}  "
          f"{header[7]:>5s}  {header[8]:>6s}  {header[9]:>9s}")
    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return statistics.mean(xs) if xs else None
    for (wf, k), entries in sorted(by_key.items()):
        n = len(entries)
        mean_p = _mean([e["p"] for e in entries])
        mean_r = _mean([e["r"] for e in entries])
        mean_a = _mean([e["atoms_edited"] for e in entries])
        mean_c = _mean([e["chars_edited"] for e in entries])
        mean_g = _mean([e["target_gap"] for e in entries])
        mean_o = _mean([e["off_target_mean"] for e in entries])
        succ = sum(1 for e in entries if e.get("fix_success")) / max(n, 1)
        def _fmt(x, fmt):
            return ("-" if x is None else format(x, fmt))
        print(
            f"{wf:12s}  {k:>3d}  {n:>3d}  "
            f"{_fmt(mean_p, '>5.2f'):>5s}  {_fmt(mean_r, '>5.2f'):>5s}  "
            f"{_fmt(mean_a, '>6.1f'):>6s}  {_fmt(mean_c, '>7.0f'):>7s}  "
            f"{_fmt(mean_g, '>5.2f'):>5s}  {_fmt(mean_o, '>6.2f'):>6s}  "
            f"{succ:>9.0%}"
        )


if __name__ == "__main__":
    main()
