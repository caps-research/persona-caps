#!/usr/bin/env python
"""π_d weight sweep — does dialing per-cell mixture weights monotonically
move the corresponding inventory dimension score?

Setup:
  • Activate two contradictory cells for the SAME dimension (high_N + low_N).
  • Vary w := P(high_N atom) ∈ {0.0, 0.25, 0.5, 0.75, 1.0}; P(low_N) = 1-w.
  • sample_cells=True, cells_per_step=1: each query draws one cell from the
    weighted Bernoulli, then retrieves top_k atoms only from that cell.
  • Mini-IPIP N items: 4 items, all targeting neuroticism. Score = mean
    rating after reverse-keying ⇒ proxy for "expressed N".

Hypothesis:  Mean N score is monotonically INCREASING in w.

Output: summary.json + console table.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

from persona_caps.agent import JointCapsAgent
from persona_caps.corpus import IfThenCorpus
from persona_caps.eval import Inventory, administer
from persona_caps.llm import get_client


def load_corpus_paper_only(path: Path) -> IfThenCorpus:
    return IfThenCorpus.from_jsonl(str(path))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus-dir", default=os.getenv("PERSONA_CAPS_CORPUS_DIR", "data/if_then/paper/beck_2022_v2"))
    p.add_argument("--out-dir",    default=os.getenv("PERSONA_CAPS_OUT_DIR",    "runs/pi_sweep"))
    p.add_argument("--inventory",  default="data/inventories/ipip_short.json")
    p.add_argument("--target-dim", default="neuroticism")
    p.add_argument("--subject-backend", default="minimax")
    p.add_argument("--subject-model",   default=os.getenv("MINIMAX_MODEL", "MiniMax-M2.7-highspeed"))
    p.add_argument("--subject-temp", type=float, default=0.7)
    p.add_argument("--top-k-per-dim", type=int, default=4)
    p.add_argument("--n-runs", type=int, default=3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260502)
    p.add_argument("--weights", default="0.0,0.25,0.5,0.75,1.0",
                   help="Comma-separated list of w (= P(high_d atom)) values.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_dir = Path(args.corpus_dir)
    high_corpus = load_corpus_paper_only(corpus_dir / f"high_{args.target_dim}.jsonl")
    low_corpus  = load_corpus_paper_only(corpus_dir / f"low_{args.target_dim}.jsonl")
    cells = {f"high_{args.target_dim}": high_corpus, f"low_{args.target_dim}": low_corpus}
    inv = Inventory.load(args.inventory)
    weights = [float(w) for w in args.weights.split(",")]

    from persona_caps.llm import LLMClient
    llm = LLMClient(args.subject_backend, args.subject_model, temperature=args.subject_temp)

    print(f"Subject: {args.subject_backend} / {args.subject_model}  T={args.subject_temp}")
    print(f"Corpus:  {args.corpus_dir}")
    print(f"Target dim: {args.target_dim}")
    print(f"Weights (P(high) values): {weights}")
    print(f"Runs/weight: {args.n_runs}, top_k_per_dim={args.top_k_per_dim}")
    print()

    rows = []
    for w in weights:
        for run_id in range(args.n_runs):
            seed = args.seed + 1000 * int(round(w * 100)) + run_id
            rng = np.random.default_rng(seed)
            agent = JointCapsAgent(
                cells, llm,
                dim_weights={f"high_{args.target_dim}": w, f"low_{args.target_dim}": 1.0 - w},
                top_k_per_dim=args.top_k_per_dim,
                mode="argmax",
                rng=rng,
                sample_cells=True,
                cells_per_step=1,
            )
            t0 = time.time()
            report = administer(agent, inv, workers=args.workers)
            wall = time.time() - t0
            target_score = report.dimension_scores.get(args.target_dim)
            row = {
                "w_high": w,
                "run_id": run_id,
                "seed": seed,
                f"{args.target_dim}_score": target_score,
                "all_dim_scores": report.dimension_scores,
                "parse_failures": report.parse_failures,
                "wall_s": round(wall, 1),
            }
            rows.append(row)
            print(f"  w_high={w:.2f}  run={run_id}  {args.target_dim}={target_score:.2f}  ({wall:.1f}s)")

    # aggregate
    agg = []
    for w in weights:
        scores = [r[f"{args.target_dim}_score"] for r in rows if r["w_high"] == w and r[f"{args.target_dim}_score"] is not None]
        if scores:
            agg.append({
                "w_high": w,
                "n": len(scores),
                "mean": statistics.mean(scores),
                "stdev": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
            })

    out = {
        "config": {
            "subject_backend": args.subject_backend,
            "subject_model": args.subject_model,
            "subject_temperature": args.subject_temp,
            "corpus_dir": args.corpus_dir,
            "target_dim": args.target_dim,
            "top_k_per_dim": args.top_k_per_dim,
            "n_runs_per_weight": args.n_runs,
            "weights": weights,
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
        bar = "#" * int((a["mean"] - 1) * 8)
        print(f"  w_high={a['w_high']:.2f}  mean={a['mean']:.2f} ± {a['stdev']:.2f}  {bar}")
    print()

    # monotonicity check
    means = [a["mean"] for a in agg]
    diffs = [means[i+1] - means[i] for i in range(len(means)-1)]
    monotone_up = all(d >= -0.05 for d in diffs)  # tolerate small noise
    print(f"Monotonic increasing (with 0.05 noise tolerance)? {monotone_up}")
    print(f"Total spread (w=1 minus w=0): {means[-1] - means[0]:+.2f}")
    print(f"\nWrote {summary}")


if __name__ == "__main__":
    main()
