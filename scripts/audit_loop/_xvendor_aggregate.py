"""Aggregate the GPT-4o cross-vendor pilot results for §4.7 paper text.

Reads runs/audit_loop_xvendor/{fix,p2_audit}/ and emits a head-to-head
table (per-workflow mean over 3 seeds) plus a short paragraph-ready
summary that mirrors the main MiniMax table structure.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[2]
XV = REPO / "runs" / "audit_loop_xvendor"
MAIN = REPO / "runs" / "audit_loop"


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _gather_caps(root: Path) -> dict[str, list[dict]]:
    out = {"caps_a": [], "caps_b": []}
    for fp in sorted((root / "fix").glob("*.json")):
        name = fp.stem
        suffix = name.split("__")[-1]
        key = "caps_a" if suffix == "locatea" else "caps_b"
        out[key].append(_load(fp))
    return out


def _gather_p2(root: Path) -> dict[str, list[dict]]:
    out = {"oracle": [], "blind": [], "detect": []}
    for fp in sorted((root / "p2_audit").glob("*.json")):
        strat = fp.stem.split("__")[-1]
        if strat in out:
            out[strat].append(_load(fp))
    return out


def _caps_row(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "atoms_restored": mean(r["footprint"]["atoms_restored"] for r in rows),
        "chars_replaced": mean(r["footprint"]["chars_replaced"] for r in rows),
        "target_gap": mean(r["target_gap"] for r in rows),
        "dir_match": mean(r["direction_match_vs_clean"] for r in rows),
        "off_target_mean": mean(r["off_target_gap_mean"] for r in rows),
        "fix_success_rate": sum(r["fix_success"] for r in rows) / len(rows),
    }


def _p2_row(rows: list[dict]) -> dict:
    row = {
        "n": len(rows),
        "target_gap": mean(r["target_gap"] for r in rows),
        "dir_match": mean(r["direction_match_vs_clean"] for r in rows),
        "off_target_mean": mean(r["off_target_gap_mean"] for r in rows),
        "fix_success_rate": sum(r["fix_success"] for r in rows) / len(rows),
    }
    locs = [r.get("localization", {}) for r in rows]
    if all("precision" in l for l in locs):
        row["precision"] = mean(l["precision"] for l in locs)
        row["recall"] = mean(l["recall"] for l in locs)
        row["f1"] = mean(l["f1"] for l in locs)
    return row


def main() -> None:
    caps = _gather_caps(XV)
    p2 = _gather_p2(XV)

    summary = {
        "subject": "openai:gpt-4o-2024-08-06",
        "cell": "high_extraversion",
        "K": 15,
        "seeds": [1, 2, 3],
        "caps_a": _caps_row(caps["caps_a"]),
        "caps_b": _caps_row(caps["caps_b"]),
        "p2_oracle": _p2_row(p2["oracle"]),
        "p2_blind": _p2_row(p2["blind"]),
        "p2_detect": _p2_row(p2["detect"]),
    }

    print("\n=== CROSS-VENDOR PILOT (subject = GPT-4o-2024-08-06) ===")
    print(f"cell=high_extraversion  K=15  n_seeds=3\n")
    print(f"{'workflow':<14} {'fix%':>5} {'gap':>5} {'dirM':>5} {'off':>5} {'atoms':>6} {'chars':>6} {'P':>5} {'R':>5}")
    rows = [
        ("CAPS-A",     summary["caps_a"]),
        ("CAPS-B",     summary["caps_b"]),
        ("P2-Oracle",  summary["p2_oracle"]),
        ("P2-Blind",   summary["p2_blind"]),
        ("P2-Detect",  summary["p2_detect"]),
    ]
    for label, r in rows:
        atoms = r.get("atoms_restored", float("nan"))
        chars = r.get("chars_replaced", float("nan"))
        prec = r.get("precision", float("nan"))
        rec = r.get("recall", float("nan"))
        print(f"{label:<14} {r['fix_success_rate']*100:>4.0f}% "
              f"{r['target_gap']:>5.2f} {r['dir_match']:>5.2f} "
              f"{r['off_target_mean']:>5.2f} "
              f"{atoms:>6.1f} {chars:>6.0f} "
              f"{prec:>5.2f} {rec:>5.2f}")

    print("\n=== KEY CONTRASTS ===")
    a, b = summary["caps_a"], summary["caps_b"]
    blind = summary["p2_blind"]
    print(f"CAPS-A edits {a['chars_replaced']:.0f} chars vs P²-Blind whole-paragraph rewrite")
    print(f"CAPS-A off-target gap {a['off_target_mean']:.2f}  vs P²-Blind {blind['off_target_mean']:.2f} "
          f"({blind['off_target_mean']/max(a['off_target_mean'],1e-6):.1f}x)")
    print(f"CAPS-A footprint {a['atoms_restored']:.1f}/15 atoms; P²-Oracle requires all 15 paragraph-flips")

    out_path = XV / "xvendor_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
