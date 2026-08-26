"""Cross-judge agreement analysis for §4.4 PersonalityBench (App. A.10.1).

Compares per-item ratings from MiniMax-M2.7-highspeed (runs/personalitybench_scored/)
and GPT-4o-2024-08-06 (independent judge, runs/personalitybench_scored_gpt4o/).
The point is to bound the same-model judge bias listed in §6: if both judges
agree at the item level *and* preserve the CAPS-SP gap direction and magnitude,
then the §4.4 headline is not an artifact of MiniMax judging itself.

Two aggregations are reported and they do not coincide:

* Item-level agreement (exact / within-1 / Pearson r / level offset) runs over
  every question both judges scored, pooled flat. This is the right denominator
  for "do the judges agree", and it is where A.10.1's n=1775, 75.9%, 98.0%,
  r=+0.951 and +0.12/10 come from.
* The CAPS-SP gap is instead reported under the §4.4 headline aggregation --
  first 50 rows per (trait, polarity) cell, low-polarity reversed, per-trait
  means averaged -- because that is the quantity A.10.1 claims judge choice
  moves by 0.08/10 (+0.86 under GPT-4o, +0.78 under MiniMax). Pooling the
  paired items flat instead gives +0.93 / +0.86: a different denominator, not
  a different conclusion. Both are printed so neither can be mistaken for the
  other.

Outputs: prints summary, writes runs/personalitybench_scored/cross_judge_agreement.json.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MM_DIR = ROOT / "runs" / "personalitybench_scored"
G4_DIR = ROOT / "runs" / "personalitybench_scored_gpt4o"


def pearson(pairs: list[tuple[float, float]]) -> float:
    if len(pairs) < 2:
        return float("nan")
    xm = statistics.mean(p[0] for p in pairs)
    ym = statistics.mean(p[1] for p in pairs)
    num = sum((a - xm) * (b - ym) for a, b in pairs)
    denx = (sum((a - xm) ** 2 for a, _ in pairs)) ** 0.5
    deny = (sum((b - ym) ** 2 for _, b in pairs)) ** 0.5
    return num / (denx * deny) if denx * deny > 0 else float("nan")


LIMIT_PER_CELL = 50  # matches scripts/aggregate_pb_3x3.py


def headline_4_4(judge_dir: Path) -> dict:
    """CAPS/SP means under the §4.4 aggregation, for one judge's scored tree."""
    per_method: dict[str, float] = {}
    for method in ["caps", "sp"]:
        per_trait: dict[str, list[float]] = {}
        for fp in sorted((judge_dir / method).glob("*.jsonl")):
            trait, pol = fp.stem.rsplit("_", 1)
            rows = [
                d["rating"]
                for d in (json.loads(l) for l in fp.read_text().splitlines() if l.strip())
                if d.get("rating") is not None
            ]
            for r in rows[:LIMIT_PER_CELL]:
                per_trait.setdefault(trait, []).append(r * 2 if pol == "high" else (6 - r) * 2)
        if not per_trait:
            return {}
        per_method[method] = statistics.mean(statistics.mean(v) for v in per_trait.values())
    return {
        "caps": per_method["caps"],
        "sp": per_method["sp"],
        "caps_minus_sp": per_method["caps"] - per_method["sp"],
    }


def collect() -> dict:
    pairs_all: list[tuple[int, int]] = []
    pairs_per_method: dict[str, list[tuple[int, int]]] = {"caps": [], "sp": []}
    aligned_per_cell: dict[tuple[str, str], list[tuple[int, int]]] = {}

    for method in ["caps", "sp"]:
        for fp in sorted((MM_DIR / method).glob("*.jsonl")):
            g4_fp = G4_DIR / method / fp.name
            if not g4_fp.exists():
                continue
            trait, pol = fp.stem.rsplit("_", 1)
            mm = {r["question"]: r for r in (json.loads(l) for l in fp.read_text().splitlines() if l.strip())}
            g4 = {r["question"]: r for r in (json.loads(l) for l in g4_fp.read_text().splitlines() if l.strip())}
            for q in set(mm) & set(g4):
                a, b = mm[q].get("rating"), g4[q].get("rating")
                if a is None or b is None:
                    continue
                pairs_all.append((a, b))
                pairs_per_method[method].append((a, b))
                aligned = (6 - a, 6 - b) if pol == "low" else (a, b)
                aligned_per_cell.setdefault((method, trait), []).append(aligned)

    def summarise(pairs: list[tuple[int, int]]) -> dict:
        n = len(pairs)
        if n == 0:
            return {"n": 0}
        return {
            "n": n,
            "exact_agreement": sum(1 for a, b in pairs if a == b) / n,
            "within_1": sum(1 for a, b in pairs if abs(a - b) <= 1) / n,
            "pearson_r": pearson(pairs),
            "mean_diff_g4_minus_mm": sum(b - a for a, b in pairs) / n,
        }

    out = {
        "all": summarise(pairs_all),
        "caps": summarise(pairs_per_method["caps"]),
        "sp": summarise(pairs_per_method["sp"]),
        "headline_aligned_x2_per_method": {},
        "per_trait_aligned_x2": {},
        "caps_minus_sp_gap_x2": {},
    }

    for method in ["caps", "sp"]:
        cells = [c for (m, _), c in aligned_per_cell.items() if m == method]
        flat = [p for c in cells for p in c]
        if not flat:
            continue
        mm = 2 * statistics.mean(p[0] for p in flat)
        g4 = 2 * statistics.mean(p[1] for p in flat)
        out["headline_aligned_x2_per_method"][method] = {
            "minimax_judge": mm,
            "gpt4o_judge": g4,
        }
        out["per_trait_aligned_x2"][method] = {}
        for trait in ["Agreeableness", "Conscientiousness", "Extraversion", "Neuroticism", "Openness"]:
            cell = aligned_per_cell.get((method, trait), [])
            if not cell:
                continue
            out["per_trait_aligned_x2"][method][trait] = {
                "minimax_judge": 2 * statistics.mean(p[0] for p in cell),
                "gpt4o_judge": 2 * statistics.mean(p[1] for p in cell),
                "n": len(cell),
            }

    h = out["headline_aligned_x2_per_method"]
    if "caps" in h and "sp" in h:
        out["caps_minus_sp_gap_x2"] = {
            "minimax_judge": h["caps"]["minimax_judge"] - h["sp"]["minimax_judge"],
            "gpt4o_judge": h["caps"]["gpt4o_judge"] - h["sp"]["gpt4o_judge"],
        }

    mm_h, g4_h = headline_4_4(MM_DIR), headline_4_4(G4_DIR)
    if mm_h and g4_h:
        out["headline_4_4_aggregation"] = {
            "minimax_judge": mm_h,
            "gpt4o_judge": g4_h,
            "judge_swap_sensitivity": abs(g4_h["caps_minus_sp"] - mm_h["caps_minus_sp"]),
        }
    return out


def main() -> None:
    out = collect()
    out_path = MM_DIR / "cross_judge_agreement.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[wrote] {out_path}")
    print()
    a = out["all"]
    print(f"All paired items (n={a['n']}): exact={100*a['exact_agreement']:.1f}%, "
          f"within±1={100*a['within_1']:.1f}%, Pearson r={a['pearson_r']:+.3f}, "
          f"mean(GPT4o−MM)={a['mean_diff_g4_minus_mm']:+.3f}")
    g = out["caps_minus_sp_gap_x2"]
    h = out["headline_aligned_x2_per_method"]
    print()
    print("Paired items only, pooled flat (NOT the §4.4 aggregation):")
    print(f"  CAPS /10: MM={h['caps']['minimax_judge']:.2f}  GPT4o={h['caps']['gpt4o_judge']:.2f}")
    print(f"  SP   /10: MM={h['sp']['minimax_judge']:.2f}  GPT4o={h['sp']['gpt4o_judge']:.2f}")
    print(f"  CAPS−SP gap: MM={g['minimax_judge']:+.2f}  GPT4o={g['gpt4o_judge']:+.2f}")

    if "headline_4_4_aggregation" in out:
        f = out["headline_4_4_aggregation"]
        mm, g4 = f["minimax_judge"], f["gpt4o_judge"]
        print()
        print("§4.4 aggregation (50/cell, low reversed, per-trait means) — the A.10.1 figures:")
        print(f"  CAPS /10: MM={mm['caps']:.2f}  GPT4o={g4['caps']:.2f}")
        print(f"  SP   /10: MM={mm['sp']:.2f}  GPT4o={g4['sp']:.2f}")
        print(f"  CAPS−SP gap: MM={mm['caps_minus_sp']:+.2f}  GPT4o={g4['caps_minus_sp']:+.2f}")
        print(f"  judge-swap sensitivity: {f['judge_swap_sensitivity']:.2f}")


if __name__ == "__main__":
    main()
