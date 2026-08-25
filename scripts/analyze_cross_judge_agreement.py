"""Cross-judge agreement analysis for §4.8 PersonalityBench.

Compares per-item ratings from MiniMax-M2.7-highspeed (current §4.8 numbers)
and GPT-4o-2024-08-06 (independent judge, runs/personalitybench_scored_gpt4o/).
The point is to bound the same-model judge bias listed in §6: if both judges
agree at the item level *and* preserve the CAPS-SP gap direction and magnitude,
then the §4.8 headline is not an artifact of MiniMax judging itself.

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
    print(f"Headline /10 (CAPS): MM={h['caps']['minimax_judge']:.2f}  GPT4o={h['caps']['gpt4o_judge']:.2f}")
    print(f"Headline /10 (SP):   MM={h['sp']['minimax_judge']:.2f}  GPT4o={h['sp']['gpt4o_judge']:.2f}")
    print(f"CAPS−SP gap: MM={g['minimax_judge']:+.2f}  GPT4o={g['gpt4o_judge']:+.2f}")


if __name__ == "__main__":
    main()
