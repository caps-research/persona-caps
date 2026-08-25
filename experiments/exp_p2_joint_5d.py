"""App. C.4 — P² (NPTI paragraph-prompt) 5-D joint baseline for §4.6.

Stacks 5 single-trait P² descriptions (one per Big-Five dim) into a single
multi-trait paragraph prompt, runs the same 20-item Mini-IPIP used in
§4.6's CAPS 5-D joint experiment, scores per-dim direction match, and
reports per-profile and aggregate direction-match rates so that the
resulting numbers are directly comparable to Table 5 (§4.6) and App. C.2.

Same protocol as exp_conditional_retrieval.py --sweep B:
  - 7 representative profiles (HHHHH, single-polarity-flip × 5, LLLLL)
  - N = 3 runs per profile
  - per-dim Mini-IPIP score; direction-match = score moves >= 0.25
    in the prescribed polarity from the midpoint of 3.0
  - subject backend: MiniMax-M2.7-highspeed (default, matches §4.6 CAPS cell)

We sample one description per dim per run (NPTI's per-call random-choice
protocol) and concatenate them with sentence boundaries. The "softmax"
analogue is not meaningful for a paragraph baseline, so we report
argmax-equivalent (single deterministic system-prompt assembly per run)
with the same 3 runs per profile.

Output: runs/p2_joint_5d_baseline/summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from persona_caps.agent import PersonaDescriptionAgent
from persona_caps.eval import Inventory, administer
from persona_caps.llm import LLMClient


# Big-Five dim names as used by Mini-IPIP / agent.py
DIMS = ["neuroticism", "extraversion", "conscientiousness", "agreeableness", "openness"]
# NPTI description.json key per (dim, polarity)
NPTI_KEY = {
    ("neuroticism", "high"): "Neuroticism",
    ("neuroticism", "low"): "Neuroticism_reversed",
    ("extraversion", "high"): "Extraversion",
    ("extraversion", "low"): "Extraversion_reversed",
    ("conscientiousness", "high"): "Conscientiousness",
    ("conscientiousness", "low"): "Conscientiousness_reversed",
    ("agreeableness", "high"): "Agreeableness",
    ("agreeableness", "low"): "Agreeableness_reversed",
    ("openness", "high"): "Openness",
    ("openness", "low"): "Openness_reversed",
}

# 7 profiles, matching §4.6 / App. C.2
PROFILES = [
    {"neuroticism": "high", "extraversion": "high", "conscientiousness": "high", "agreeableness": "high", "openness": "high"},
    {"neuroticism": "low",  "extraversion": "high", "conscientiousness": "high", "agreeableness": "high", "openness": "high"},
    {"neuroticism": "high", "extraversion": "low",  "conscientiousness": "high", "agreeableness": "high", "openness": "high"},
    {"neuroticism": "high", "extraversion": "high", "conscientiousness": "low",  "agreeableness": "high", "openness": "high"},
    {"neuroticism": "high", "extraversion": "high", "conscientiousness": "high", "agreeableness": "low",  "openness": "high"},
    {"neuroticism": "high", "extraversion": "high", "conscientiousness": "high", "agreeableness": "high", "openness": "low"},
    {"neuroticism": "low",  "extraversion": "low",  "conscientiousness": "low",  "agreeableness": "low",  "openness": "low"},
]
INV_PATH = ROOT / "data" / "inventories" / "ipip_short.json"
DESC_BANK_PATH = ROOT / "baselines" / "NPTI" / "NPTI" / "dataset" / "description.json"
OUT_DIR_BASE = ROOT / "runs" / "p2_joint_5d_baseline"


def _load_desc_bank() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with DESC_BANK_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _label_for_profile(profile: dict[str, str]) -> str:
    return "-".join(f"{profile[d][0].upper()}{d[0].upper()}" for d in DIMS)


def _stack_p2_prompt(profile: dict[str, str], desc_bank: list[dict[str, str]], rng: random.Random) -> str:
    """Concatenate one description per dim into a single multi-trait system prompt."""
    parts = []
    for dim in DIMS:
        pol = profile[dim]
        key = NPTI_KEY[(dim, pol)]
        # sample one description (NPTI per-call random-choice protocol)
        entry = rng.choice(desc_bank)
        parts.append(entry[key])
    # Stack into one paragraph with sentence-boundary separators
    return " ".join(parts)


def _expected_direction(score: float | None, pol: str) -> int:
    if score is None:
        return 0
    mid = 3.0
    if pol == "high":
        return +1 if score > mid + 0.25 else (-1 if score < mid - 0.25 else 0)
    return +1 if score < mid - 0.25 else (-1 if score > mid + 0.25 else 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="minimax", choices=["minimax", "openai", "gemini"])
    ap.add_argument("--model", default=None, help="model name (defaults to backend-specific)")
    ap.add_argument("--n-runs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260501)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    out_dir = OUT_DIR_BASE if args.backend == "minimax" else OUT_DIR_BASE.with_name(OUT_DIR_BASE.name + f"_{args.backend}")
    out_dir.mkdir(parents=True, exist_ok=True)
    default_model = {
        "minimax": os.getenv("MINIMAX_MODEL", "MiniMax-M2.7-highspeed"),
        "openai": "gpt-4o-2024-08-06",
        "gemini": "gemini-2.5-flash",
    }[args.backend]
    model = args.model or default_model
    subj = LLMClient(args.backend, model, temperature=args.temperature)
    inventory = Inventory.load(INV_PATH)
    desc_bank = _load_desc_bank()
    print(f"[info] loaded {len(desc_bank)} P² description entries from {DESC_BANK_PATH.name}")

    rows: list[dict] = []
    for prof_idx, profile in enumerate(PROFILES):
        prof_label = _label_for_profile(profile)
        for run_id in range(args.n_runs):
            seed = args.seed + 10_000 * prof_idx + 1000 * run_id
            rng = random.Random(seed)
            t0 = time.time()
            system_prompt = _stack_p2_prompt(profile, desc_bank, rng)
            agent = PersonaDescriptionAgent(trait_prompt=system_prompt, llm=subj)
            report = administer(agent, inventory, workers=args.workers)
            dt = time.time() - t0

            per_dim_status: dict[str, dict] = {}
            num_correct = 0
            for dim in DIMS:
                pol = profile[dim]
                score = report.dimension_scores.get(dim)
                sgn = _expected_direction(score, pol)
                per_dim_status[dim] = {"expected_pol": pol, "score": score, "direction_match": sgn}
                if sgn > 0:
                    num_correct += 1

            row = {
                "profile_label": prof_label,
                "profile": dict(profile),
                "run_id": run_id,
                "seed": seed,
                "all_dim_scores": dict(report.dimension_scores),
                "per_dim_status": per_dim_status,
                "num_active": 5,
                "num_correct_direction": num_correct,
                "parse_failures": report.parse_failures,
                "wall_s": round(dt, 1),
                "system_prompt_len_words": len(system_prompt.split()),
            }
            rows.append(row)
            print(
                f"[P2-joint prof={prof_label:25s} run={run_id}] correct={num_correct}/5  "
                f"sys-prompt={row['system_prompt_len_words']}w  ({dt:.1f}s)"
            )

    # Aggregate per profile + overall
    agg: list[dict] = []
    for profile in PROFILES:
        prof_label = _label_for_profile(profile)
        prof_rows = [r for r in rows if r["profile_label"] == prof_label]
        per_dim_match: dict[str, float] = {}
        for dim in DIMS:
            ms = [1.0 if r["per_dim_status"][dim]["direction_match"] > 0 else 0.0 for r in prof_rows]
            per_dim_match[dim] = round(sum(ms) / len(ms), 3) if ms else float("nan")
        all_dim_correct_rate = round(
            sum(r["num_correct_direction"] for r in prof_rows) / (5.0 * len(prof_rows)), 3
        )
        agg.append({
            "profile_label": prof_label,
            "profile": dict(profile),
            "n_runs": len(prof_rows),
            "per_dim_direction_match_rate": per_dim_match,
            "all_dim_correct_rate": all_dim_correct_rate,
        })

    overall = round(sum(a["all_dim_correct_rate"] for a in agg) / len(agg), 3)
    payload = {
        "config": {
            "subject_backend": args.backend,
            "n_runs_per_profile": args.n_runs,
            "temperature": args.temperature,
            "method": "P2_stacked_5_descriptions",
        },
        "overall_direction_match_rate": overall,
        "agg": agg,
        "rows": rows,
    }
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")
    print(f"\n=== P² 5-D joint baseline ({args.backend}, N={args.n_runs}/profile) ===")
    print(f"  overall direction-match rate: {overall}")
    print(f"  vs CAPS §4.6 MiniMax argmax overall: 0.87")
    print(f"  per profile:")
    for a in agg:
        print(f"    {a['profile_label']:25s} {a['all_dim_correct_rate']:.3f}")


if __name__ == "__main__":
    main()
