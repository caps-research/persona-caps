"""Administer a personality inventory to a CAPS-RAG agent.

Usage:
    python scripts/run_inventory.py \
        --corpus data/if_then/high_neuroticism.jsonl \
        --inventory data/inventories/ipip_short.json \
        --backend minimax \
        --out runs/high_neuroticism__minimax.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from persona_caps.agent import CapsAgent  # noqa: E402
from persona_caps.corpus import IfThenCorpus  # noqa: E402
from persona_caps.eval import Inventory, administer  # noqa: E402
from persona_caps.llm import get_client  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--backend", default="minimax")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    corpus = IfThenCorpus.from_jsonl(args.corpus)
    inventory = Inventory.load(args.inventory)
    print(f"[run] corpus={args.corpus} ({len(corpus)} atoms)")
    print(f"[run] inventory={inventory.name} ({len(inventory.items)} items)")
    print(f"[run] backend={args.backend} top_k={args.top_k}")

    llm = get_client(args.backend, temperature=args.temperature)
    agent = CapsAgent(corpus=corpus, llm=llm, top_k=args.top_k)
    report = administer(agent, inventory)
    print()
    print(report.summary())

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "corpus": args.corpus,
            "inventory": args.inventory,
            "backend": args.backend,
            "top_k": args.top_k,
            "temperature": args.temperature,
            "dimension_scores": report.dimension_scores,
            "dimension_n": report.dimension_n,
            "parse_failures": report.parse_failures,
            "responses": [
                {
                    "item_id": r.item.id,
                    "dimension": r.item.dimension,
                    "keyed": r.item.keyed,
                    "rating": r.rating,
                    "scored": r.scored,
                    "raw_text": r.raw_text,
                }
                for r in report.responses
            ],
        }
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[run] wrote -> {out}")


if __name__ == "__main__":
    main()
