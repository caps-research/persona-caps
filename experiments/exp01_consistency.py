"""Experiment 01 — RQ1: Does an if-then corpus produce stable, measurable personality?
And: does it match or beat the trait-prompt and persona-description baselines?

For each (dimension, polarity) pair we compare three persona-construction methods:

    A. CAPS-RAG          — our method, conditioning on retrieved if-then atoms
    B. Trait label       — short status-quo prompt: "You are high in neuroticism."
    C. Persona desc.     — long SOUL.md-style persona description

All three answer the same Mini-IPIP. We measure:
  - Target-dimension score: did we hit the intended dimension?
  - Off-target movement: did we leak into other Big Five dimensions? (lower = better)
  - Run-to-run variance: how stable is each method?

Usage:
    python experiments/exp01_consistency.py --backend minimax --runs 3 \\
        --dimensions neuroticism --out runs/exp01_minimax.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from persona_caps.agent import (  # noqa: E402
    CapsAgent,
    PersonaDescriptionAgent,
    TraitPromptAgent,
)
from persona_caps.corpus import IfThenCorpus  # noqa: E402
from persona_caps.eval import Inventory, administer  # noqa: E402
from persona_caps.llm import get_client  # noqa: E402

INVENTORY = ROOT / "data" / "inventories" / "ipip_short.json"
DEFAULT_CORPUS_DIR = ROOT / "data" / "if_then"
BASELINES = json.loads((ROOT / "data" / "baselines" / "trait_prompts.json").read_text())

# Methods we compare. Each builds an agent from a (dimension, polarity) spec.
#
# Two ablations sit alongside the persona-as-string baselines:
#
# - ``random_demos``  : same retrieval scaffolding as ``caps`` but the injected
#   atoms are personality-irrelevant chitchat. If random_demos scores like
#   CAPS, the effect is just demonstration count, not persona content.
# - ``wrong_trait``   : same retrieval scaffolding but the corpus is for a
#   DIFFERENT, orthogonal Big Five dimension at the same polarity. If
#   wrong_trait still moves the target dimension, then ANY persona-like
#   content moves any dimension (a strong null) — which would weaken the
#   claim that retrieval is content-specific. By default we pair as:
#   neuroticism <-> conscientiousness, extraversion <-> agreeableness, and
#   openness uses conscientiousness as the foil.
METHODS = ("caps", "trait_label", "persona_description", "random_demos", "wrong_trait")
NOISE_CORPUS_PATH = ROOT / "data" / "if_then" / "noise_demos.jsonl"

DIMENSIONS = ("neuroticism", "extraversion", "conscientiousness", "agreeableness", "openness")
POLARITIES = ("high", "low")

WRONG_TRAIT_PAIRS = {
    "neuroticism": "conscientiousness",
    "conscientiousness": "neuroticism",
    "extraversion": "agreeableness",
    "agreeableness": "extraversion",
    "openness": "conscientiousness",
}


def _build_agent(method: str, dimension: str, polarity: str, llm, corpus_dir: Path) -> object:
    key = f"{polarity}_{dimension}"
    if method == "caps":
        path = corpus_dir / f"{key}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"corpus missing: {path}. Run scripts/seed_corpus.py or one of the "
                "scripts/build_*_atoms_*.py builders to generate it."
            )
        return CapsAgent(corpus=IfThenCorpus.from_jsonl(path), llm=llm, top_k=4)
    if method == "trait_label":
        return TraitPromptAgent(BASELINES[key]["trait_label"], llm=llm)
    if method == "persona_description":
        return PersonaDescriptionAgent(BASELINES[key]["persona_description"], llm=llm)
    if method == "random_demos":
        if not NOISE_CORPUS_PATH.exists():
            raise FileNotFoundError(f"noise corpus missing: {NOISE_CORPUS_PATH}")
        return CapsAgent(corpus=IfThenCorpus.from_jsonl(NOISE_CORPUS_PATH), llm=llm, top_k=4)
    if method == "wrong_trait":
        wrong_dim = WRONG_TRAIT_PAIRS.get(dimension)
        if wrong_dim is None:
            raise ValueError(f"no wrong-trait pairing defined for {dimension}")
        wrong_key = f"{polarity}_{wrong_dim}"
        path = corpus_dir / f"{wrong_key}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"wrong-trait corpus missing: {path}. (target={dimension}, foil={wrong_dim})"
            )
        return CapsAgent(corpus=IfThenCorpus.from_jsonl(path), llm=llm, top_k=4)
    raise ValueError(f"unknown method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="minimax")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--dimensions", nargs="+", default=["neuroticism"], choices=list(DIMENSIONS),
    )
    parser.add_argument(
        "--methods", nargs="+", default=list(METHODS), choices=list(METHODS),
    )
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel LLM calls per inventory administration")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR,
                        help="Directory containing {polarity}_{dim}.jsonl corpus files. "
                             "Default = legacy hand-authored seeds at data/if_then/. "
                             "For paper-track, use data/if_then/paper/beck_2022/")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    inventory = Inventory.load(INVENTORY)
    llm = get_client(args.backend, temperature=args.temperature)

    # results[dim][polarity][method] = list[run_dict]
    results: dict = {dim: {pol: {m: [] for m in args.methods} for pol in POLARITIES} for dim in args.dimensions}

    for dim in args.dimensions:
        for pol in POLARITIES:
            for method in args.methods:
                for r in range(args.runs):
                    print(f"[exp01] {method:22s} {pol:5s} {dim:18s} run {r + 1}/{args.runs}")
                    try:
                        agent = _build_agent(method, dim, pol, llm, args.corpus_dir)
                    except FileNotFoundError as e:
                        print(f"  SKIP: {e}")
                        continue
                    report = administer(agent, inventory, workers=args.workers)
                    results[dim][pol][method].append({
                        "dimension_scores": report.dimension_scores,
                        "parse_failures": report.parse_failures,
                    })

    print("\n=== RQ1 summary ===")
    print(f"backend: {args.backend}  runs: {args.runs}\n")

    # Build comparison tables, one per dimension.
    for dim in args.dimensions:
        print(f"--- target dimension: {dim} ---")
        # Header
        cols = ["method", "polarity"] + list(DIMENSIONS)
        print("| " + " | ".join(cols) + " |")
        print("| " + " | ".join(["---"] * len(cols)) + " |")
        for method in args.methods:
            for pol in POLARITIES:
                runs = results[dim][pol][method]
                if not runs:
                    print(f"| {method} | {pol} | " + " | ".join(["—"] * len(DIMENSIONS)) + " |")
                    continue
                row = [method, pol]
                for d in DIMENSIONS:
                    vals = [r["dimension_scores"].get(d) for r in runs if d in r["dimension_scores"]]
                    if not vals:
                        row.append("—")
                        continue
                    m = mean(vals)
                    s = pstdev(vals) if len(vals) > 1 else 0.0
                    marker = "*" if d == dim else " "
                    row.append(f"{marker}{m:.2f}±{s:.2f}")
                print("| " + " | ".join(row) + " |")

        # Headline metric: high - low spread on the target dimension. CAPS should
        # produce as large or larger a spread as the strongest baseline.
        print(f"\n  spread on target dim ({dim}): high - low")
        for method in args.methods:
            highs = [r["dimension_scores"].get(dim) for r in results[dim]["high"][method] if dim in r["dimension_scores"]]
            lows = [r["dimension_scores"].get(dim) for r in results[dim]["low"][method] if dim in r["dimension_scores"]]
            if not highs or not lows:
                print(f"    {method}: insufficient data")
                continue
            print(f"    {method:22s}: {mean(highs):.2f} - {mean(lows):.2f} = {mean(highs) - mean(lows):+.2f}")
        print()

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "backend": args.backend,
                "temperature": args.temperature,
                "runs": args.runs,
                "dimensions": args.dimensions,
                "methods": args.methods,
                "corpus_dir": str(args.corpus_dir),
            },
            "results": results,
        }
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[exp01] wrote -> {out}")


if __name__ == "__main__":
    main()
