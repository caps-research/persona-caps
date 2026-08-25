"""Experiment 03 — RQ3: Targeted compositional editing.

This is the killer demo against IRiS / persona-vector approaches: with
persona-caps you can move *one* Big Five dimension by adding atoms to
the corpus, leaving the other dimensions undisturbed. With a steering
vector or a monolithic persona prompt, equivalent edits either require
re-fitting the vector / rewriting the prompt or have unpredictable
side effects on other dimensions.

Setup
-----
For a chosen target (e.g. neuroticism, high), we start from the
*opposite-polarity* corpus (e.g. low-neuroticism) — the hardest
starting point — and incrementally add high-neuroticism atoms in
configurable step sizes. After each step we re-administer the
Mini-IPIP inventory.

Predictions
-----------
1. The target dimension score moves monotonically toward the new
   polarity as atoms are added.
2. Non-target dimensions stay within sampling noise of their initial
   values.

Usage
-----
    python experiments/exp03_editing.py --backend minimax \\
        --dimension neuroticism --target-polarity high \\
        --steps 0 5 10 15 20 \\
        --out runs/exp03_neuroticism_minimax.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from persona_caps.agent import CapsAgent  # noqa: E402
from persona_caps.corpus import IfThenAtom, IfThenCorpus  # noqa: E402
from persona_caps.eval import Inventory, administer  # noqa: E402
from persona_caps.llm import get_client  # noqa: E402

INVENTORY = ROOT / "data" / "inventories" / "ipip_short.json"
DEFAULT_CORPUS_DIR = ROOT / "data" / "if_then"
DIMENSIONS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")


def build_edited_corpus(
    base_atoms: list[IfThenAtom],
    addition_atoms: list[IfThenAtom],
    n_added: int,
) -> IfThenCorpus:
    """Base corpus + first ``n_added`` atoms from ``addition_atoms``."""
    n = max(0, min(n_added, len(addition_atoms)))
    return IfThenCorpus(base_atoms + addition_atoms[:n])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="minimax")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--dimension", required=True, choices=list(DIMENSIONS),
        help="Big Five dimension to edit",
    )
    parser.add_argument(
        "--target-polarity", required=True, choices=["high", "low"],
        help="Polarity we are trying to move toward",
    )
    parser.add_argument(
        "--steps", nargs="+", type=int, default=[0, 5, 10, 15, 20],
        help="Atom counts at which to re-administer the inventory",
    )
    parser.add_argument(
        "--base-corpus", default=None,
        help=(
            "Override the base corpus path. By default we start from the "
            "OPPOSITE-polarity corpus for the same dimension (hardest edit). "
            "Pass a path (e.g. data/if_then/noise_demos.jsonl) to test the "
            "easier edit-from-neutral case."
        ),
    )
    parser.add_argument("--runs", type=int, default=2,
                        help="independent inventory runs per edit step")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel LLM calls per inventory administration")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR,
                        help="Directory containing {polarity}_{dim}.jsonl. "
                             "Default = legacy hand-authored seeds. For paper-track "
                             "use data/if_then/paper/beck_2022_v2/")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    base_polarity = "low" if args.target_polarity == "high" else "high"
    if args.base_corpus is not None:
        base_path = Path(args.base_corpus)
        base_label = base_path.stem
    else:
        base_path = args.corpus_dir / f"{base_polarity}_{args.dimension}.jsonl"
        base_label = f"{base_polarity}_{args.dimension}"
    target_path = args.corpus_dir / f"{args.target_polarity}_{args.dimension}.jsonl"
    if not base_path.exists() or not target_path.exists():
        print(
            f"[exp03] missing corpora. Need both:\n  {base_path}\n  {target_path}\n"
            "Run scripts/seed_corpus.py to generate."
        )
        sys.exit(1)

    base_atoms = list(IfThenCorpus.from_jsonl(base_path).atoms)
    target_atoms = list(IfThenCorpus.from_jsonl(target_path).atoms)
    print(
        f"[exp03] dimension={args.dimension}  start={base_label}({len(base_atoms)} atoms)  "
        f"adding={args.target_polarity}({len(target_atoms)} atoms available)\n"
        f"[exp03] steps={args.steps}  runs/step={args.runs}"
    )

    inventory = Inventory.load(INVENTORY)
    llm = get_client(args.backend, temperature=args.temperature)

    per_step: list[dict] = []
    for step in args.steps:
        edited = build_edited_corpus(base_atoms, target_atoms, step)
        print(f"\n[exp03] step n_added={step}  total_atoms={len(edited)}")
        runs: list[dict] = []
        for r in range(args.runs):
            agent = CapsAgent(corpus=edited, llm=llm, top_k=4)
            report = administer(agent, inventory, workers=args.workers)
            runs.append({
                "dimension_scores": report.dimension_scores,
                "parse_failures": report.parse_failures,
            })
            print(
                f"  run {r + 1}/{args.runs}: target {args.dimension}="
                f"{report.dimension_scores.get(args.dimension, float('nan')):.2f}"
            )
        agg = {d: mean(r["dimension_scores"][d] for r in runs if d in r["dimension_scores"])
               for d in DIMENSIONS}
        per_step.append({"n_added": step, "n_total": len(edited), "runs": runs, "aggregate": agg})

    print("\n=== RQ3 summary ===")
    print(f"target: move {args.dimension} from {base_polarity} -> {args.target_polarity}\n")
    print("| n_added | " + " | ".join(DIMENSIONS) + " |")
    print("| " + " | ".join(["---"] * (len(DIMENSIONS) + 1)) + " |")
    for s in per_step:
        cells = [str(s["n_added"])]
        for d in DIMENSIONS:
            v = s["aggregate"].get(d)
            marker = "*" if d == args.dimension else " "
            cells.append(f"{marker}{v:.2f}" if v is not None else "—")
        print("| " + " | ".join(cells) + " |")

    # Headline: did target move? did off-targets stay put?
    if len(per_step) >= 2:
        first, last = per_step[0]["aggregate"], per_step[-1]["aggregate"]
        target_delta = last.get(args.dimension, 0) - first.get(args.dimension, 0)
        off_target_deltas = {
            d: last.get(d, 0) - first.get(d, 0) for d in DIMENSIONS if d != args.dimension
        }
        print(f"\n  target Δ ({args.dimension}): {target_delta:+.2f}")
        print(f"  max |off-target Δ|: {max(abs(v) for v in off_target_deltas.values()):.2f}")
        for d, v in off_target_deltas.items():
            print(f"    {d}: {v:+.2f}")
        print(
            "\n  Pass criterion: |target Δ| > 2 * max |off-target Δ|  "
            "(target moves at least 2x more than the noisiest off-target)"
        )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "backend": args.backend,
                "temperature": args.temperature,
                "dimension": args.dimension,
                "target_polarity": args.target_polarity,
                "base_polarity": base_polarity,
                "steps": args.steps,
                "runs": args.runs,
                "corpus_dir": str(args.corpus_dir),
            },
            "per_step": per_step,
        }
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[exp03] wrote -> {out}")


if __name__ == "__main__":
    main()
