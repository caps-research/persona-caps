#!/usr/bin/env python
"""Continuous-prompt baseline for the π_d sweep.

This is the *natural reviewer baseline* for the §4.5 continuous-control
claim: instead of varying retrieval mixture weights $\\pi_d$ over an
empirically grounded if-then corpus, just *tell the LLM in the system
prompt* what mixture you want — e.g. "You are 75% high-Neuroticism and
25% low-Neuroticism." If this trivial prompt also produces a monotonic
Mini-IPIP curve at comparable spread, the π_d novelty collapses to
"continuous control with audit trail" rather than "continuous control
per se".

Two baseline styles are supported:

  --style mixed_pct
    "You are someone who is X% high in {dim} and (100-X)% low in {dim}."
    Direct semantic mirror of π_d's mixture-weight framing.

  --style scalar_score
    "You are someone whose Big Five {dim} score is Y out of 5."
    where Y = 1 + 4*w. Closer to what a practitioner actually writes.

Sweep matches experiments/exp_pi_sweep.py exactly:
  • dims: neuroticism, extraversion, conscientiousness, agreeableness, openness
  • w_high in {0.0, 0.25, 0.5, 0.75, 1.0}
  • N=3 runs per weight
  • Mini-IPIP target dim score is the dependent variable

Output: runs/pi_sweep_<short>_<vendor>_promptbase_<style>/summary.json
in the same shape as exp_pi_sweep.py for direct comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from persona_caps.agent import TraitPromptAgent
from persona_caps.eval import Inventory, administer
from persona_caps.llm import LLMClient


DIM_DISPLAY = {
    "neuroticism": "Neuroticism",
    "extraversion": "Extraversion",
    "conscientiousness": "Conscientiousness",
    "agreeableness": "Agreeableness",
    "openness": "Openness",
}


def build_system_prompt(dim: str, w_high: float, style: str) -> str:
    name = DIM_DISPLAY[dim]
    if style == "mixed_pct":
        # direct mirror of pi_d framing
        hi_pct = int(round(w_high * 100))
        lo_pct = 100 - hi_pct
        if hi_pct == 100:
            return (
                f"You are someone who is high in the Big Five trait of {name}."
            )
        if hi_pct == 0:
            return (
                f"You are someone who is low in the Big Five trait of {name}."
            )
        return (
            f"You are a person whose {name} is {hi_pct}% on the high pole and "
            f"{lo_pct}% on the low pole. Behave so that your overall expression "
            f"of {name} reflects this {hi_pct}/{lo_pct} mixture."
        )
    elif style == "scalar_score":
        # 1.00 at w=0  → 5.00 at w=1, exactly matching the Mini-IPIP scale
        score = 1.0 + 4.0 * w_high
        return (
            f"You are someone whose Big Five {name} score, on the standard "
            f"1-to-5 personality inventory scale where 1 is the lowest and 5 "
            f"is the highest, is approximately {score:.2f}. "
            f"Behave consistently with that level of {name}."
        )
    else:
        raise ValueError(f"unknown --style: {style}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--inventory", default="data/inventories/ipip_short.json")
    p.add_argument("--target-dim", default="neuroticism")
    p.add_argument("--subject-backend", default="minimax")
    p.add_argument("--subject-model",   default=os.getenv("MINIMAX_MODEL", "MiniMax-M2.7-highspeed"))
    p.add_argument("--subject-temp", type=float, default=0.7)
    p.add_argument("--style", choices=["mixed_pct", "scalar_score"], default="mixed_pct")
    p.add_argument("--n-runs", type=int, default=3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260508)
    p.add_argument("--weights", default="0.0,0.25,0.5,0.75,1.0")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inv = Inventory.load(args.inventory)
    weights = [float(w) for w in args.weights.split(",")]

    llm = LLMClient(args.subject_backend, args.subject_model, temperature=args.subject_temp)

    print(f"Subject: {args.subject_backend} / {args.subject_model}  T={args.subject_temp}")
    print(f"Style:    {args.style}")
    print(f"Target:   {args.target_dim}")
    print(f"Weights:  {weights}")
    print(f"Runs/w:   {args.n_runs}")
    print()

    rows = []
    for w in weights:
        sys_prompt = build_system_prompt(args.target_dim, w, args.style)
        for run_id in range(args.n_runs):
            agent = TraitPromptAgent(sys_prompt, llm)
            t0 = time.time()
            report = administer(agent, inv, workers=args.workers)
            wall = time.time() - t0
            target_score = report.dimension_scores.get(args.target_dim)
            row = {
                "w_high": w,
                "run_id": run_id,
                "system_prompt": sys_prompt,
                f"{args.target_dim}_score": target_score,
                "all_dim_scores": report.dimension_scores,
                "parse_failures": report.parse_failures,
                "wall_s": round(wall, 1),
            }
            rows.append(row)
            print(f"  w_high={w:.2f}  run={run_id}  {args.target_dim}={target_score:.2f}  ({wall:.1f}s)")

    agg = []
    for w in weights:
        scores = [r[f"{args.target_dim}_score"] for r in rows
                  if r["w_high"] == w and r[f"{args.target_dim}_score"] is not None]
        if scores:
            agg.append({
                "w_high": w,
                "n": len(scores),
                "mean": statistics.mean(scores),
                "stdev": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
            })

    out = {
        "config": {
            "experiment": "exp_pi_sweep_promptbase",
            "subject_backend": args.subject_backend,
            "subject_model": args.subject_model,
            "subject_temperature": args.subject_temp,
            "target_dim": args.target_dim,
            "n_runs_per_weight": args.n_runs,
            "weights": weights,
            "style": args.style,
            "inventory": args.inventory,
        },
        "rows": rows,
        "agg": agg,
    }
    summary = out_dir / "summary.json"
    summary.write_text(json.dumps(out, indent=2))

    print()
    print(f"=== Aggregated: P(high) -> mean {args.target_dim} score ===")
    for a in agg:
        bar = "#" * max(0, int((a["mean"] - 1) * 8))
        print(f"  w_high={a['w_high']:.2f}  mean={a['mean']:.2f} ± {a['stdev']:.2f}  {bar}")
    means = [a["mean"] for a in agg]
    diffs = [means[i+1] - means[i] for i in range(len(means)-1)]
    monotone_up = all(d >= -0.05 for d in diffs)
    print()
    print(f"Monotonic increasing (with 0.05 noise tolerance)? {monotone_up}")
    print(f"Total spread (w=1 minus w=0): {means[-1] - means[0]:+.2f}")
    print(f"\nWrote {summary}")


if __name__ == "__main__":
    main()
