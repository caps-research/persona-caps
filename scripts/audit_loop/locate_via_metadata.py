"""Locator (b): scheduled corpus-metadata-content audit.

Sweeps the *whole* polluted corpus once with a small classifier prompt
asking GPT-4o to label each atom's behavior text with its Big Five
dimension and polarity, then flags atoms whose classification
disagrees with the stored ``polarity`` field. This is the second of
the two audit modes contrasted in §4.7 — the "scheduled" complement
to Locator (a)'s incident-driven workflow.

Unlike Locator (a), this audit:

* does not depend on the inventory or on any subject-side response;
* sees every atom, so recall is not capped by retrieval coverage;
* costs one LLM call per atom (cheap, batchable in principle); and
* surfaces "label rot" that nothing has yet caused a user-visible
  failure — useful for preventive audits.

The classifier is deliberately *generic*: it does not know the
ground-truth flipped IDs and is not told what cell the file
purports to be. It judges each behavior on its own merits and then
this script does the metadata-comparison locally.

The output structure mirrors ``locate_via_responses.py`` so downstream
fix-and-measure scripts can consume either.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from persona_caps.corpus import IfThenCorpus  # noqa: E402
from persona_caps.llm import ChatMessage, get_client  # noqa: E402


CLASSIFY_PROMPT_VERSION = "v1"

CLASSIFY_SYSTEM = """You are a careful Big Five personality coder. \
Given one If-Then behavioral contingency (a situation + a behavior), \
classify which Big Five dimension the BEHAVIOR most strongly expresses \
and whether it expresses the HIGH or LOW pole of that dimension. \
Use only the standard five: extraversion, agreeableness, conscientiousness, \
neuroticism, openness. If the behavior is genuinely ambiguous between two \
dimensions, pick the dominant one. Output STRICT JSON only, with schema: \
{"dimension": "<one of: extraversion, agreeableness, conscientiousness, neuroticism, openness>", \
"polarity": "<high or low>", \
"confidence": <0.0 to 1.0>, \
"rationale": "<one short clause>"}. \
No prose outside the JSON, no markdown fences."""


CLASSIFY_USER_TEMPLATE = """Situation: {situation}
Behavior: {behavior}

Classify the BEHAVIOR's Big Five dimension and polarity.
"""


def _classify_atom(llm, atom, *, max_attempts: int) -> dict | None:
    msg = CLASSIFY_USER_TEMPLATE.format(
        situation=atom.situation.strip(),
        behavior=atom.behavior.strip(),
    )
    for _ in range(max_attempts):
        raw = llm.chat(
            messages=[ChatMessage(role="user", content=msg)],
            system=CLASSIFY_SYSTEM,
        )
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].lstrip()
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            continue
        dim = (obj.get("dimension") or "").strip().lower()
        pol = (obj.get("polarity") or "").strip().lower()
        if dim and pol in ("high", "low"):
            obj["dimension"] = dim
            obj["polarity"] = pol
            return obj
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--polluted-corpus", required=True)
    p.add_argument("--injection-log", required=True)
    p.add_argument("--classifier", default="gpt-4o")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    corpus = IfThenCorpus.from_jsonl(args.polluted_corpus)
    inj = json.loads(Path(args.injection_log).read_text(encoding="utf-8"))
    ground_truth: set[str] = set(inj["flipped_ids"])

    llm = get_client(args.classifier, temperature=args.temperature)
    print(
        f"[locate-b] sweeping {len(corpus)} atoms via {args.classifier} "
        f"(T={args.temperature}, max_attempts={args.max_attempts})",
        flush=True,
    )

    rows: list[dict] = []
    flagged: list[str] = []
    parse_failures = 0
    for i, atom in enumerate(corpus.atoms, 1):
        cls = _classify_atom(llm, atom, max_attempts=args.max_attempts)
        if cls is None:
            parse_failures += 1
            rows.append({
                "atom_id": atom.id,
                "stored_dimension": atom.target_dimension,
                "stored_polarity": atom.polarity,
                "classified": None,
                "flagged": False,
                "in_ground_truth": atom.id in ground_truth,
            })
            print(f"  [{i}/{len(corpus)}] {atom.id} -> CLASSIFY FAILED", flush=True)
            continue
        # An atom is flagged if its classified polarity disagrees with
        # the stored polarity on the SAME dimension. If the classifier
        # picks a different dimension entirely, we treat that as also
        # disagreeing (the behavior text does not match the metadata).
        same_dim = cls["dimension"] == (atom.target_dimension or "").lower()
        flag = (not same_dim) or (cls["polarity"] != (atom.polarity or "").lower())
        if flag:
            flagged.append(atom.id)
        rows.append({
            "atom_id": atom.id,
            "stored_dimension": atom.target_dimension,
            "stored_polarity": atom.polarity,
            "classified": cls,
            "flagged": flag,
            "in_ground_truth": atom.id in ground_truth,
        })
        marker = "FLAG" if flag else "ok  "
        gt = "TP" if (flag and atom.id in ground_truth) else (
            "FN" if (not flag and atom.id in ground_truth) else (
                "FP" if (flag and atom.id not in ground_truth) else "TN"
            )
        )
        print(
            f"  [{i}/{len(corpus)}] {marker} ({gt}) {atom.id} -> "
            f"{cls['dimension']}/{cls['polarity']} (conf={cls.get('confidence', '?')})",
            flush=True,
        )

    flagged_set = set(flagged)
    tp = len(flagged_set & ground_truth)
    fp = len(flagged_set - ground_truth)
    fn = len(ground_truth - flagged_set)
    tn = len(corpus.atoms) - tp - fp - fn
    precision = tp / max(len(flagged_set), 1)
    recall = tp / max(len(ground_truth), 1)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    payload = {
        "locator": "metadata",
        "polluted_corpus": args.polluted_corpus,
        "injection_log": args.injection_log,
        "classifier": args.classifier,
        "temperature": args.temperature,
        "max_attempts": args.max_attempts,
        "classify_prompt_version": CLASSIFY_PROMPT_VERSION,
        "n_atoms": len(corpus.atoms),
        "n_flagged": len(flagged_set),
        "n_ground_truth": len(ground_truth),
        "parse_failures": parse_failures,
        "metrics": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
        },
        "flagged_ids": sorted(flagged_set),
        "ground_truth": sorted(ground_truth),
        "per_atom": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[locate-b] wrote {out}  P={precision:.2f} R={recall:.2f} F1={f1:.2f} "
        f"(TP={tp} FP={fp} FN={fn} TN={tn}; parse_fail={parse_failures})"
    )


if __name__ == "__main__":
    main()
