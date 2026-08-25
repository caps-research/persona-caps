"""Aggregate the full GPT-4o cross-vendor grid for §4.7 / App.~A.11.

Grid:
  * high_E  K ∈ {5, 10, 15}            -- K-saturation stress test on the pilot cell
  * low_A   K = 15
  * high_N  K = 15
  * low_O   K = 15

For each (cell, K) we report:
  * CAPS sanity pass-rate (n/3) + mean |Δs̄| on target dim
  * P²  sanity pass-rate (n/3) + mean |Δs̄| on target dim
  * Per-workflow mean fix-success, target-gap, off-target, atoms, chars (when applicable)
  * Side-by-side vs matched MiniMax conditions

Output: prints a paper-ready table to stdout AND writes
  runs/audit_loop_xvendor/grid_summary.json
  runs/audit_loop_xvendor/grid_paper_table.md
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[2]
XV = REPO / "runs" / "audit_loop_xvendor"
MAIN = REPO / "runs" / "audit_loop"

GRID = [
    ("high_extraversion", 5),
    ("high_extraversion", 10),
    ("high_extraversion", 15),
    ("low_agreeableness", 15),
    ("high_neuroticism", 15),
    ("low_openness", 15),
]
CELL_LABEL = {
    "high_extraversion": "high-E",
    "low_agreeableness": "low-A",
    "high_neuroticism": "high-N",
    "low_openness": "low-O",
}
SEEDS = (1, 2, 3)


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _name(cell, k, s):
    return f"{cell}__k{k}_s{s}_targeted"


def _safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None


def _row_for_subject(root: Path, cell: str, k: int) -> dict:
    """Pull per-seed dicts for one (cell, K), one subject (root)."""
    sanity = [_load(root/"sanity"/f"{_name(cell,k,s)}.json") for s in SEEDS]
    p2san  = [_load(root/"p2_sanity"/f"{_name(cell,k,s)}.json") for s in SEEDS]
    fixa   = [_load(root/"fix"/f"{_name(cell,k,s)}__locatea.json") for s in SEEDS]
    fixb   = [_load(root/"fix"/f"{_name(cell,k,s)}__locateb.json") for s in SEEDS]
    p2 = {strat: [_load(root/"p2_audit"/f"{_name(cell,k,s)}__{strat}.json") for s in SEEDS]
          for strat in ("oracle","blind","detect")}

    def fix_stats(rows):
        rows = [r for r in rows if r]
        if not rows: return {}
        return {
            "n": len(rows),
            "fix_succ": sum(r["fix_success"] for r in rows),
            "fix_rate": sum(r["fix_success"] for r in rows)/len(rows),
            "target_gap": mean(r["target_gap"] for r in rows),
            "off_target": mean(r["off_target_gap_mean"] for r in rows),
            "atoms": mean(r["footprint"]["atoms_restored"] for r in rows) if "footprint" in rows[0] else None,
            "chars": mean(r["footprint"]["chars_replaced"] for r in rows) if "footprint" in rows[0] else None,
        }

    def p2_stats(rows):
        rows = [r for r in rows if r]
        if not rows: return {}
        return {
            "n": len(rows),
            "fix_succ": sum(r["fix_success"] for r in rows),
            "fix_rate": sum(r["fix_success"] for r in rows)/len(rows),
            "target_gap": mean(r["target_gap"] for r in rows),
            "off_target": mean(r["off_target_gap_mean"] for r in rows),
        }

    return {
        "caps_sanity_passes": sum(1 for d in sanity if d and d.get("pass_overall")),
        "caps_sanity_shift_mean": _safe_mean([d["score_shift"] for d in sanity if d]),
        "p2_sanity_passes": sum(1 for d in p2san if d and d.get("pass_overall")),
        "p2_sanity_shift_mean": _safe_mean([d["score_shift"] for d in p2san if d]),
        "caps_a": fix_stats(fixa),
        "caps_b": fix_stats(fixb),
        "p2_oracle": p2_stats(p2["oracle"]),
        "p2_blind":  p2_stats(p2["blind"]),
        "p2_detect": p2_stats(p2["detect"]),
    }


def _per_seed_chars(root: Path, cell: str, k: int):
    """Return list of per-seed chars-replaced for CAPS-A and CAPS-B (for byte-identity check)."""
    out = {"a": [], "b": []}
    for s in SEEDS:
        for kind, sub in (("a","locatea"),("b","locateb")):
            r = _load(root/"fix"/f"{_name(cell,k,s)}__{sub}.json")
            if r: out[kind].append((s, r["footprint"]["chars_replaced"], r["footprint"]["atoms_restored"]))
    return out


def main():
    summary = {}
    print("\n" + "="*88)
    print(f"{'cell':<8}{'K':>4}{'subj':<7}{'sanCAPS':>9}{'sanP2':>7}"
          f"{'A:fix':>7}{'A:off':>8}{'A:chars':>9}"
          f"{'B:fix':>7}{'B:off':>8}{'B:chars':>9}"
          f"{'Bl:fix':>8}{'Bl:off':>8}{'De:fix':>8}{'De:off':>8}")
    print("="*88)

    for cell, k in GRID:
        for label, root in [("MM", MAIN), ("GPT4o", XV)]:
            r = _row_for_subject(root, cell, k)
            key = f"{cell}_k{k}_{label}"
            summary[key] = r
            a, b, bl, de = r["caps_a"], r["caps_b"], r["p2_blind"], r["p2_detect"]
            def f(x,d=2,w=6):
                if x is None or x == {}: return "—".rjust(w)
                if isinstance(x, float): return f"{x:.{d}f}".rjust(w)
                return str(x).rjust(w)
            print(f"{CELL_LABEL[cell]:<8}{k:>4}{label:<7}"
                  f"{r['caps_sanity_passes']:>3}/3({f(r['caps_sanity_shift_mean'],1,3)})"
                  f"{r['p2_sanity_passes']:>3}/3"
                  f"{a.get('fix_succ','-'):>5}/3 {f(a.get('off_target'),2,5)} {f(a.get('chars'),0,8)}"
                  f"{b.get('fix_succ','-'):>5}/3 {f(b.get('off_target'),2,5)} {f(b.get('chars'),0,8)}"
                  f"{bl.get('fix_succ','-'):>6}/3 {f(bl.get('off_target'),2,5)}"
                  f"{de.get('fix_succ','-'):>6}/3 {f(de.get('off_target'),2,5)}")

    print("\n" + "="*88)
    print("BYTE-IDENTICAL EDIT FOOTPRINT CHECK — CAPS-A char counts per seed")
    print("="*88)
    print(f"{'cell':<8}{'K':>4}{'seed':>5}{'MM chars':>10}{'GPT4o chars':>13}{'match':>7}")
    for cell, k in GRID:
        mm = {s:c for s,c,_ in _per_seed_chars(MAIN, cell, k)["a"]}
        xv = {s:c for s,c,_ in _per_seed_chars(XV, cell, k)["a"]}
        for s in sorted(set(mm) | set(xv)):
            m = mm.get(s); x = xv.get(s)
            match = "✓" if (m is not None and x is not None and m == x) else ("✗" if (m and x) else "n/a")
            print(f"{CELL_LABEL[cell]:<8}{k:>4}{s:>5}{m if m else '—':>10}{x if x else '—':>13}{match:>7}")

    out = XV / "grid_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    # paper-ready markdown table
    md_lines = [
        "| cell | K | subj | CAPS-sanity | P²-sanity | A: fix | A: off | A: chars | B: fix | B: off | B: chars | Bl: fix | Bl: off | De: fix | De: off |",
        "| --- | ---: | --- | :---: | :---: | :---: | ---: | ---: | :---: | ---: | ---: | :---: | ---: | :---: | ---: |",
    ]
    for cell, k in GRID:
        for label, root in [("MM", MAIN), ("GPT-4o", XV)]:
            r = _row_for_subject(root, cell, k)
            a, b, bl, de = r["caps_a"], r["caps_b"], r["p2_blind"], r["p2_detect"]
            def F(d, key, fmt="{:.2f}", default="—"):
                v = d.get(key)
                if v is None: return default
                return fmt.format(v)
            def Fi(d, key, default="—"):
                v = d.get(key)
                if v is None: return default
                return str(int(v))
            md_lines.append(
                f"| {CELL_LABEL[cell]} | {k} | {label} "
                f"| {r['caps_sanity_passes']}/3 "
                f"| {r['p2_sanity_passes']}/3 "
                f"| {Fi(a,'fix_succ')}/3 | {F(a,'off_target')} | {Fi(a,'chars')} "
                f"| {Fi(b,'fix_succ')}/3 | {F(b,'off_target')} | {Fi(b,'chars')} "
                f"| {Fi(bl,'fix_succ')}/3 | {F(bl,'off_target')} "
                f"| {Fi(de,'fix_succ')}/3 | {F(de,'off_target')} |"
            )
    md_out = XV / "grid_paper_table.md"
    md_out.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"wrote {md_out}")


if __name__ == "__main__":
    main()
