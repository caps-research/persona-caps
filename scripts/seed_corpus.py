"""Seed an if-then corpus from a target trait + polarity using an LLM.

This produces the *first* of the three corpus-construction strategies
described in NOVELTY.md (LLM-generated from inventory items). The other
two — human-authored and dialogue-extracted — will be implemented
separately, and the paper will compare all three.

Usage:
    python scripts/seed_corpus.py --trait neuroticism --level high --n 30 \
        --backend minimax --out data/if_then/high_neuroticism.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path

# Make src/ importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from persona_caps.corpus import IfThenAtom, IfThenCorpus  # noqa: E402
from persona_caps.llm import ChatMessage, get_client  # noqa: E402

PROMPT_TMPL = """You are a personality psychology research assistant grounded in \
Mischel & Shoda's Cognitive-Affective Personality System (CAPS, 1995). \
CAPS holds that personality is expressed as stable IF-THEN behavioral \
signatures: a person is characterized not by an abstract trait, but by \
their distinctive pattern of "when situation X happens, I do Y".

Generate {n} diverse, concrete IF-THEN behavioral contingencies that \
together describe someone who is {level} on the Big Five dimension of \
{trait}. The contingencies must:

1. Each have a SPECIFIC, observable situation (not vague like "in social settings").
2. Each have a SPECIFIC, observable behavior (not vague like "feels anxious").
3. Cover diverse domains: work, family, strangers, conflict, success, failure, \
   surprise, boredom, time pressure, technology, money, etc.
4. Be psychologically realistic — a real {level}-{trait} person could plausibly \
   show this pattern.
5. NEVER mention the trait label ("{trait}") in the situation or behavior text.

Output STRICT JSON, an array of objects with EXACTLY these keys:
  "situation": short string
  "behavior":  short string
  "affect":    short string (the felt emotion, optional, may be "")
  "goal":      short string (the activated goal, optional, may be "")

Output ONLY the JSON array, no prose, no code fences.
"""


def _extract_json_array(text: str) -> list[dict]:
    """Best-effort extraction of a JSON array from LLM output."""
    text = text.strip()
    # Strip code fences if the model added them despite instructions.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    # Find the outermost [...] block.
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON array found in:\n{text[:400]}")
    return json.loads(text[start : end + 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trait", required=True, choices=[
        "openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism",
    ])
    parser.add_argument("--level", required=True, choices=["high", "low"])
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--backend", default="minimax", help="preset or backend:model")
    parser.add_argument("--out", required=True)
    parser.add_argument("--temperature", type=float, default=0.9)
    args = parser.parse_args()

    client = get_client(args.backend, temperature=args.temperature, max_tokens=4096)
    prompt = PROMPT_TMPL.format(n=args.n, trait=args.trait, level=args.level)
    print(f"[seed] requesting {args.n} atoms ({args.level} {args.trait}) from {args.backend} ...")
    raw = client.chat(messages=[ChatMessage(role="user", content=prompt)])
    items = _extract_json_array(raw)
    print(f"[seed] parsed {len(items)} items")

    corpus = IfThenCorpus()
    for it in items:
        atom = IfThenAtom(
            id=f"{args.level}_{args.trait}_{uuid.uuid4().hex[:8]}",
            situation=str(it.get("situation", "")).strip(),
            behavior=str(it.get("behavior", "")).strip(),
            affect=(str(it["affect"]).strip() or None) if it.get("affect") else None,
            goal=(str(it["goal"]).strip() or None) if it.get("goal") else None,
            target_dimension=args.trait,
            polarity=args.level,
        )
        if atom.situation and atom.behavior:
            corpus.add(atom)

    corpus.to_jsonl(args.out)
    print(f"[seed] wrote {len(corpus)} atoms -> {args.out}")


if __name__ == "__main__":
    main()
