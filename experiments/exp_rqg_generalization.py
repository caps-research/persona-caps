"""RQ-G — Generalization vs. lookup probe.

For each Mini-IPIP item answered by a CapsAgent, we record the maximum
top-k cosine similarity between the item-prompt and the retrieved atoms.
We then bin items by that similarity and report the trait-aligned
hit-rate per bin.

The question: do items where retrieval scrapes nothing very similar
still get pulled in the personality-consistent direction?

If hit-rate stays high in the *low-similarity* bin, retrieval is acting
as **persona induction** (the LLM compresses the 4 atoms into a coherent
character that generalises to unseen items). If hit-rate collapses in the
low-similarity bin, retrieval is closer to **lookup** and the system
would only work on questions whose phrasing happens to neighbour the
corpus.

Definitions
-----------
- *Trait-aligned*: for high_X corpora, the scored value (after reverse
  keying) is >= scale_midpoint + 0.5 on a target-dimension item; for
  low_X corpora, it is <= scale_midpoint - 0.5. We only score on items
  whose ``dimension`` matches the persona's target dimension.
- *Max similarity*: the cosine of the top-1 retrieved atom. We use
  top-1 because the system prompt is dominated by the most relevant
  contingency; lower-ranked atoms tend to drift off-topic anyway.

Usage
-----
    python experiments/exp_rqg_generalization.py \
        --backend minimax --workers 4 \
        --dimensions extraversion neuroticism \
        --corpus-dir data/if_then/paper/beck_2022_v2 \
        --out runs/rqg/probe_v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from persona_caps.agent import CapsAgent  # noqa: E402
from persona_caps.corpus import IfThenCorpus  # noqa: E402
from persona_caps.eval import Inventory, administer  # noqa: E402
from persona_caps.llm import get_client  # noqa: E402

INVENTORY = ROOT / "data" / "inventories" / "ipip_short.json"
DEFAULT_CORPUS_DIR = ROOT / "data" / "if_then" / "paper" / "beck_2022_v2"

DIMENSIONS = ("neuroticism", "extraversion", "conscientiousness", "agreeableness", "openness")
POLARITIES = ("high", "low")

# Default similarity bins. These are absolute cosine values, *not*
# percentiles, so the bin-occupancy itself is meaningful: if 90% of items
# fall in [0, 0.2) we have an honest picture of the typical regime.
DEFAULT_BINS = [(0.0, 0.20), (0.20, 0.35), (0.35, 1.01)]


def trait_aligned(scored: float | None, polarity: str, scale_mid: float, margin: float) -> bool | None:
    """Return True if the scored value is on the persona-consistent side
    of the scale by at least ``margin``. None means unscorable."""
    if scored is None:
        return None
    if polarity == "high":
        return scored >= scale_mid + margin
    return scored <= scale_mid - margin


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", default="minimax")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--dimensions", nargs="+", default=["extraversion"], choices=list(DIMENSIONS)
    )
    p.add_argument(
        "--polarities", nargs="+", default=list(POLARITIES), choices=list(POLARITIES)
    )
    p.add_argument("--runs", type=int, default=1, help="repeats per (dim,pol) for stability")
    p.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    p.add_argument("--margin", type=float, default=0.5,
                   help="how far from scale midpoint we require for 'aligned'")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    inv = Inventory.load(INVENTORY)
    scale_mid = (inv.scale_min + inv.scale_max) / 2.0
    llm = get_client(args.backend, temperature=args.temperature)

    # Per-item records across all (dim, pol, run) we run.
    # Each: {dim, pol, run, item_id, item_dim, keyed, rating, scored,
    #        max_sim, top_atom_ids, aligned}
    records: list[dict] = []

    for dim in args.dimensions:
        for pol in args.polarities:
            corpus_path = args.corpus_dir / f"{pol}_{dim}.jsonl"
            if not corpus_path.exists():
                print(f"[rqg] SKIP missing corpus {corpus_path}")
                continue
            corpus = IfThenCorpus.from_jsonl(corpus_path)
            agent = CapsAgent(corpus=corpus, llm=llm, top_k=4)
            for r in range(args.runs):
                print(f"[rqg] {pol:5s} {dim:18s} run {r+1}/{args.runs}")
                report = administer(agent, inv, workers=args.workers)
                for resp in report.responses:
                    sims = [s for _, s in resp.fired]
                    max_sim = max(sims) if sims else 0.0
                    aligned = None
                    # Only score alignment on target-dim items.
                    if resp.item.dimension == dim:
                        aligned = trait_aligned(
                            resp.scored, pol, scale_mid, args.margin
                        )
                    records.append({
                        "dim": dim,
                        "pol": pol,
                        "run": r,
                        "item_id": resp.item.id,
                        "item_dim": resp.item.dimension,
                        "keyed": resp.item.keyed,
                        "rating": resp.rating,
                        "scored": resp.scored,
                        "max_sim": round(max_sim, 4),
                        "top_atom_ids": [aid for aid, _ in resp.fired],
                        "aligned": aligned,
                    })

    # Aggregate: for each persona (dim, pol), bin target-dim items by
    # max_sim and report aligned-rate.
    print("\n=== RQ-G summary (target-dim items only, bin-by-max_sim) ===")
    print("Question: does CAPS pull *unseen* items in the personality direction?\n")
    bins = DEFAULT_BINS
    summary: dict = {}
    for dim in args.dimensions:
        for pol in args.polarities:
            subset = [r for r in records
                      if r["dim"] == dim and r["pol"] == pol
                      and r["item_dim"] == dim and r["aligned"] is not None]
            if not subset:
                continue
            print(f"--- {pol}_{dim}  (n={len(subset)} target-dim observations across runs) ---")
            print(f"{'sim_bin':<14} {'n':>4} {'aligned_rate':>14}  {'mean_sim':>9}  example_items")
            persona_summary = []
            for lo, hi in bins:
                bucket = [r for r in subset if lo <= r["max_sim"] < hi]
                if not bucket:
                    print(f"  [{lo:.2f},{hi:.2f})   0           --        --   --")
                    persona_summary.append({
                        "lo": lo, "hi": hi, "n": 0,
                        "aligned_rate": None, "mean_sim": None,
                    })
                    continue
                rate = sum(1 for b in bucket if b["aligned"]) / len(bucket)
                msim = sum(b["max_sim"] for b in bucket) / len(bucket)
                examples = ",".join(b["item_id"] for b in bucket[:3])
                print(f"  [{lo:.2f},{hi:.2f}) {len(bucket):>4} "
                      f"{rate*100:>12.1f}%  {msim:>9.3f}  {examples}")
                persona_summary.append({
                    "lo": lo, "hi": hi, "n": len(bucket),
                    "aligned_rate": rate, "mean_sim": msim,
                })
            summary[f"{pol}_{dim}"] = persona_summary
            print()

    # Pooled across personas: same binning, all (dim, pol) combined.
    print("--- POOLED across all personas ---")
    pooled = [r for r in records if r["item_dim"] == r["dim"] and r["aligned"] is not None]
    pooled_summary = []
    for lo, hi in bins:
        bucket = [r for r in pooled if lo <= r["max_sim"] < hi]
        if not bucket:
            pooled_summary.append({"lo": lo, "hi": hi, "n": 0,
                                   "aligned_rate": None, "mean_sim": None})
            continue
        rate = sum(1 for b in bucket if b["aligned"]) / len(bucket)
        msim = sum(b["max_sim"] for b in bucket) / len(bucket)
        print(f"  [{lo:.2f},{hi:.2f}) n={len(bucket):>3}  aligned_rate={rate*100:.1f}%  mean_sim={msim:.3f}")
        pooled_summary.append({"lo": lo, "hi": hi, "n": len(bucket),
                               "aligned_rate": rate, "mean_sim": msim})

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "backend": args.backend,
                "temperature": args.temperature,
                "runs": args.runs,
                "dimensions": args.dimensions,
                "polarities": args.polarities,
                "corpus_dir": str(args.corpus_dir),
                "margin": args.margin,
                "bins": bins,
            },
            "records": records,
            "per_persona_summary": summary,
            "pooled_summary": pooled_summary,
        }
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[rqg] wrote -> {args.out}")


if __name__ == "__main__":
    main()
