"""Experiment — Scalability curve: target-dimension Mini-IPIP spread vs corpus size.

For each (dimension, polarity) condition we subsample the Beck v2 paper-track
corpus down to K atoms (K in {3, 5, 10, FULL}), build a CapsAgent on the
subsample, administer the Mini-IPIP, and record the dimension scores.

The headline plot is *target-dimension spread* M(high) - M(low) as a function
of K, averaged across the five Big Five dimensions. CAPS theory predicts a
monotonically increasing curve that saturates as the per-pole atom budget
covers the situation space the inventory items probe.

Usage (cheap default ~720 calls on Gemini-Flash, ~$0.50):
    python experiments/exp_scalability.py \\
        --gen-backend gemini-flash --runs 1 \\
        --sizes 3 5 10 0 \\
        --out runs/scalability/run01.json

Where size=0 means "use the full per-pole pool".

The script is intentionally low-cost so it can be re-run quickly across
backends; sample sizes per cell are 12 (Mini-IPIP items), so per-cell variance
is across-question rather than across-run.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from persona_caps.agent import CapsAgent  # noqa: E402
from persona_caps.corpus import IfThenCorpus, IfThenAtom  # noqa: E402
from persona_caps.eval import Inventory, administer  # noqa: E402
from persona_caps.llm import get_client  # noqa: E402

INVENTORY = ROOT / "data" / "inventories" / "ipip_short.json"
CORPUS_DIR = ROOT / "data" / "if_then" / "paper" / "beck_2022_v2"
DIMENSIONS = ("neuroticism", "extraversion", "conscientiousness", "agreeableness", "openness")
POLARITIES = ("high", "low")


def _load_pool(dim: str, pol: str) -> list[IfThenAtom]:
    path = CORPUS_DIR / f"{pol}_{dim}.jsonl"
    if not path.exists():
        return []
    return list(IfThenCorpus.from_jsonl(path).atoms)


def _subsample(pool: list[IfThenAtom], k: int, rng: random.Random) -> tuple[list[IfThenAtom], int]:
    """Return (subsample, effective_size). k=0 means use full pool."""
    if k == 0 or k >= len(pool):
        return list(pool), len(pool)
    chosen = rng.sample(pool, k)
    return chosen, k


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gen-backend", default="gemini-flash")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--runs", type=int, default=1,
                        help="independent random subsamples per (dim, pol, size) cell")
    parser.add_argument("--sizes", type=int, nargs="+", default=[3, 5, 10, 0],
                        help="atom-budget per pole; use 0 to mean 'full pool'")
    parser.add_argument("--dimensions", nargs="+", default=list(DIMENSIONS), choices=list(DIMENSIONS))
    parser.add_argument("--workers", type=int, default=2,
                        help="parallel LLM calls per inventory administration")
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument("--out", default="runs/scalability/run01.json")
    args = parser.parse_args()

    inventory = Inventory.load(INVENTORY)
    llm = get_client(args.gen_backend, temperature=args.temperature)

    # Pre-load pools so we can report effective sizes once.
    pools: dict[tuple[str, str], list[IfThenAtom]] = {}
    for dim in args.dimensions:
        for pol in POLARITIES:
            pools[(dim, pol)] = _load_pool(dim, pol)

    print("[scal] per-pole pool sizes:")
    for (dim, pol), pool in pools.items():
        print(f"  {pol:5s} {dim:18s}: {len(pool)} atoms")
    print()

    # results[dim][pol][size_label] = list[run_dict]
    results: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {
        dim: {pol: {} for pol in POLARITIES} for dim in args.dimensions
    }

    cell_id = 0
    for k in args.sizes:
        size_label = "full" if k == 0 else str(k)
        for dim in args.dimensions:
            for pol in POLARITIES:
                pool = pools[(dim, pol)]
                if not pool:
                    print(f"[scal] SKIP {pol} {dim} (no pool)")
                    continue
                bucket = results[dim][pol].setdefault(size_label, [])
                for r in range(args.runs):
                    cell_id += 1
                    rng = random.Random(args.seed + cell_id)
                    subsample, eff = _subsample(pool, k, rng)
                    print(f"[scal] {pol:5s} {dim:18s} k={size_label:>4s} eff={eff:>3d} "
                          f"run {r+1}/{args.runs}")
                    corpus = IfThenCorpus(atoms=subsample)
                    agent = CapsAgent(corpus=corpus, llm=llm, top_k=4)
                    report = administer(agent, inventory, workers=args.workers)
                    bucket.append({
                        "effective_size": eff,
                        "requested_size": k,
                        "dimension_scores": report.dimension_scores,
                        "parse_failures": report.parse_failures,
                        "atom_ids": [a.id for a in subsample],
                    })

    # Summary: target-dim spread per size, averaged across dimensions.
    print()
    print("=== scalability summary: M(high) - M(low) on the target dimension ===")
    size_labels = ["full" if k == 0 else str(k) for k in args.sizes]
    print(f"{'dim':<18s} | " + " | ".join(f"k={s:>4s}" for s in size_labels))
    print("-" * (20 + 14 * len(size_labels)))
    per_size_means: dict[str, list[float]] = {s: [] for s in size_labels}
    for dim in args.dimensions:
        row = [f"{dim:<18s}"]
        for s in size_labels:
            highs = [r["dimension_scores"].get(dim) for r in results[dim]["high"].get(s, [])
                     if dim in r["dimension_scores"]]
            lows = [r["dimension_scores"].get(dim) for r in results[dim]["low"].get(s, [])
                    if dim in r["dimension_scores"]]
            if not highs or not lows:
                row.append("    n/a    ")
                continue
            spread = mean(highs) - mean(lows)
            per_size_means[s].append(spread)
            row.append(f"{spread:+5.2f}")
        print(" | ".join(row))
    print("-" * (20 + 14 * len(size_labels)))
    avg_row = [f"{'mean(5 dims)':<18s}"]
    for s in size_labels:
        vals = per_size_means[s]
        if vals:
            avg_row.append(f"{mean(vals):+5.2f}")
        else:
            avg_row.append("    n/a    ")
    print(" | ".join(avg_row))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {
            "gen_backend": args.gen_backend,
            "temperature": args.temperature,
            "runs": args.runs,
            "sizes": args.sizes,
            "dimensions": args.dimensions,
            "seed": args.seed,
            "workers": args.workers,
        },
        "pool_sizes": {f"{p}_{d}": len(pools[(d, p)]) for d in args.dimensions for p in POLARITIES},
        "results": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[scal] wrote -> {out}")


if __name__ == "__main__":
    main()
