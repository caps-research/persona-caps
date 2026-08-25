#!/usr/bin/env python3
"""scripts/aggregate_pb_3x3.py

Reads the 9 run directories produced by ``scripts/run_pb_3backbone_3method.sh``
and emits the §4.4 headline 3×3 table at uniform 50q/cell, GPT-4o judge.

Output:
    paper/tables/pb_3x3.md          (markdown table for the paper)
    runs/_aggregate_pb_3x3.json     (raw numbers for downstream scripts)

This is the canonical re-deriving script for the §4.4 headline numbers; every
number in the §4.4 main-paper table comes from this aggregator.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIMIT_PER_CELL = 50  # apples-to-apples cap (§4.4)

# (subject_label, run_tag) — must match scripts/run_pb_3backbone_3method.sh
SUBJECTS = [
    ("MiniMax-M2.7", "mm"),
    ("Gemini-2.5-Flash", "gem"),
    ("Llama-3-8B (Q4)", "l8b"),
]
METHODS = [("CAPS", "caps"), ("SP", "sp"), ("P²", "p2")]

# Some runs were produced before the standardised layout; we honour both:
#   runs/personalitybench_<tag>_<method>_scored/summary.json  (new)
#   runs/personalitybench_<tag>_scored/{caps,sp}_*.jsonl      (legacy MM/Gem/L8B CAPS+SP)
LEGACY_CAPS_SP = {
    "mm":  ROOT / "runs" / "personalitybench_scored_gpt4o",
    "gem": ROOT / "runs" / "personalitybench_gemini_scored_gpt4o",
    "l8b": ROOT / "runs" / "personalitybench_l8b_scored",
}


def collect(root: Path, method_subdir: str, limit: int = LIMIT_PER_CELL):
    """Per-trait list of /10-scaled ratings, capped at `limit` per (trait,polarity)."""
    base = root / method_subdir
    per_trait: dict[str, list[float]] = defaultdict(list)
    if not base.exists():
        return None
    for f in sorted(base.glob("*.jsonl")):
        trait, pol = f.stem.rsplit("_", 1)
        rows: list[int] = []
        for line in f.open():
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("rating") is not None:
                rows.append(d["rating"])
        for r in rows[:limit]:
            r10 = r * 2 if pol == "high" else (6 - r) * 2
            per_trait[trait].append(r10)
    if not per_trait:
        return None
    return per_trait


def find_method_root(tag: str, method_run_tag: str) -> Path | None:
    """Try the new (per-method) path first, then fall back to legacy CAPS/SP combined."""
    new = ROOT / "runs" / f"personalitybench_{tag}_{method_run_tag}_scored"
    if (new / method_run_tag).exists():
        return new
    if (new).exists():
        return new
    if method_run_tag in {"caps", "sp"} and tag in LEGACY_CAPS_SP:
        return LEGACY_CAPS_SP[tag]
    return None


def main() -> None:
    table: dict[str, dict[str, dict]] = {}
    for subj_label, tag in SUBJECTS:
        table[subj_label] = {}
        for method_label, method_tag in METHODS:
            root = find_method_root(tag, method_tag)
            if root is None:
                table[subj_label][method_label] = None
                continue
            pt = collect(root, method_tag)
            if pt is None:
                table[subj_label][method_label] = None
                continue
            per_trait = {t: sum(v) / len(v) if v else None for t, v in pt.items()}
            avg = sum(per_trait.values()) / len(per_trait)
            n_per_cell = {t: len(v) for t, v in pt.items()}
            table[subj_label][method_label] = {
                "per_trait": per_trait,
                "avg": avg,
                "n_per_cell": n_per_cell,
            }

    # Console summary
    print(f"\n=== §4.4 headline 3×3 (uniform {LIMIT_PER_CELL}q/cell, GPT-4o judge) ===")
    print(f"{'Subject':<22} {'CAPS':>7} {'SP':>7} {'P²':>7}  | "
          f"{'CAPS−SP':>9} {'CAPS−P²':>9}")
    for subj_label, _ in SUBJECTS:
        r = table[subj_label]
        c = r.get("CAPS", {}) or {}
        s = r.get("SP", {}) or {}
        p = r.get("P²", {}) or {}
        cv = c.get("avg", float("nan"))
        sv = s.get("avg", float("nan"))
        pv = p.get("avg", float("nan"))
        print(f"{subj_label:<22} {cv:>7.2f} {sv:>7.2f} {pv:>7.2f}  | "
              f"{cv - sv:>+9.2f} {cv - pv:>+9.2f}")

    # Markdown for the paper
    out_md = ROOT / "paper" / "tables"
    out_md.mkdir(parents=True, exist_ok=True)
    md = ["| subject | CAPS /10 | SP /10 | P² /10 | **CAPS − SP** | **CAPS − P²** |",
          "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for subj_label, _ in SUBJECTS:
        r = table[subj_label]
        c = (r.get("CAPS") or {}).get("avg", float("nan"))
        s = (r.get("SP") or {}).get("avg", float("nan"))
        p = (r.get("P²") or {}).get("avg", float("nan"))
        md.append(
            f"| {subj_label} | {c:.2f} | {s:.2f} | {p:.2f} | "
            f"**{c - s:+.2f}** | **{c - p:+.2f}** |"
        )
    (out_md / "pb_3x3.md").write_text("\n".join(md) + "\n")

    # Raw JSON for downstream
    out_json = ROOT / "runs" / "_aggregate_pb_3x3.json"
    out_json.write_text(json.dumps(
        {"limit_per_cell": LIMIT_PER_CELL, "judge": "gpt-4o-2024-08-06", "table": table},
        indent=2,
        default=float,
    ))
    print(f"\nwrote: {out_md / 'pb_3x3.md'}")
    print(f"wrote: {out_json}")


if __name__ == "__main__":
    main()
