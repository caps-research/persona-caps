"""Extract an if-then corpus from a dialogue transcript.

This is the third corpus-construction method described in NOVELTY.md and
PLAN.md (the first two: hand-authored seed corpora, and LLM-generated
from a trait+polarity spec). It addresses the "personality is *learned*
through experience" extension — given an agent's dialogue history,
recover the if-then signatures that history exhibits.

Input formats supported (auto-detected, JSONL or JSON-array):

    1. {"speaker": "Alice", "text": "..."}
    2. {"role": "user"|"assistant", "content": "..."}
    3. {"name": "Alice", "message": "..."}

You specify which speaker is the *target* (the one whose personality
we are reconstructing). The script slides a window over the transcript
and, for each consecutive (other-speaker, target-speaker) pair, asks
the LLM to express the pair as a clean if-then atom (situation = what
prompted the response, behavior = how the target responded, both
abstracted away from the specific topic). It then deduplicates atoms
that are too semantically similar.

Usage:
    python scripts/extract_from_dialogue.py \\
        --input mahm/logs/agent_dialogue.jsonl \\
        --target-speaker "assistant" \\
        --max-atoms 50 \\
        --backend minimax \\
        --out data/if_then/extracted_assistant.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from persona_caps.corpus import IfThenAtom, IfThenCorpus  # noqa: E402
from persona_caps.llm import ChatMessage, get_client  # noqa: E402

EXTRACT_PROMPT = """You are a personality psychology research assistant. \
Below is one exchange from a dialogue: a prompt-turn from someone else, \
followed by a response-turn from the speaker whose personality we are \
trying to characterize.

Your job is to express this exchange as ONE if-then behavioral \
contingency — the kind Mischel & Shoda (1995) call a *situation-behavior \
signature*. The contingency must:

1. ABSTRACT the situation away from the specific topic. Not "when asked \
   about Q3 budget", but "when pressed for a concrete number". Not "when \
   they mention their cat", but "when they share something personal".
2. ABSTRACT the behavior the same way. Not the literal words, but the \
   *kind of move* the speaker made.
3. Be psychologically meaningful — describe a disposition, not just a \
   single act. If the exchange isn't psychologically informative (e.g. \
   pure factual exchange, scheduling, greetings) output the JSON value \
   ``null`` instead.

PROMPT TURN (other speaker):
"{prompt_turn}"

RESPONSE TURN (target speaker):
"{response_turn}"

Output STRICT JSON, ONE object with these keys (or the literal value \
``null`` if not informative):
  "situation": short abstracted string starting with "when ..."
  "behavior":  short abstracted string starting with "I ..." or "they ..."
  "affect":    short string for likely felt emotion ("" if none)
  "goal":      short string for likely activated goal ("" if none)

Output ONLY the JSON object or the literal ``null``, no prose, no fences.
"""


def _normalize_turn(rec: dict) -> tuple[str, str] | None:
    """Return (speaker, text) from one of the supported record schemas."""
    speaker = rec.get("speaker") or rec.get("role") or rec.get("name")
    text = rec.get("text") or rec.get("content") or rec.get("message")
    if speaker is None or text is None:
        return None
    speaker = str(speaker).strip()
    text = str(text).strip()
    if not speaker or not text:
        return None
    return speaker, text


def load_turns(path: Path) -> list[tuple[str, str]]:
    """Auto-detect JSONL vs JSON-array; normalize to a flat list of (speaker, text)."""
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    records: list[dict] = []
    if raw.startswith("["):
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                records = [r for r in data if isinstance(r, dict)]
        except json.JSONDecodeError:
            pass
    if not records:
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if isinstance(rec, dict):
                    records.append(rec)
            except json.JSONDecodeError:
                continue
    turns: list[tuple[str, str]] = []
    for rec in records:
        norm = _normalize_turn(rec)
        if norm:
            turns.append(norm)
    return turns


def make_pairs(
    turns: list[tuple[str, str]],
    target: str,
    *,
    max_pairs: int | None = None,
) -> list[tuple[str, str]]:
    """Take all (other_turn, target_turn) consecutive pairs."""
    pairs: list[tuple[str, str]] = []
    for i in range(1, len(turns)):
        prev_speaker, prev_text = turns[i - 1]
        cur_speaker, cur_text = turns[i]
        if cur_speaker == target and prev_speaker != target:
            pairs.append((prev_text, cur_text))
            if max_pairs and len(pairs) >= max_pairs:
                break
    return pairs


def _extract_one(client, prompt_turn: str, response_turn: str) -> dict | None:
    text = client.chat(messages=[ChatMessage(
        role="user",
        content=EXTRACT_PROMPT.format(
            prompt_turn=prompt_turn[:1500], response_turn=response_turn[:1500],
        ),
    )])
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.MULTILINE).strip()
    if s.lower() == "null":
        return None
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    if not obj.get("situation") or not obj.get("behavior"):
        return None
    return obj


def _dedupe_by_similarity(atoms: list[IfThenAtom], threshold: float = 0.92) -> list[IfThenAtom]:
    """Drop atoms whose situation embedding is too similar to a previously kept atom.

    Uses the same encoder as the retriever so dedup distance and retrieval
    distance live in the same space.
    """
    if len(atoms) <= 1:
        return atoms
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embs = model.encode([a.situation for a in atoms], normalize_embeddings=True, show_progress_bar=False)
    embs = np.asarray(embs, dtype=np.float32)
    keep_idx: list[int] = []
    for i in range(len(atoms)):
        if any(float(embs[i] @ embs[j]) >= threshold for j in keep_idx):
            continue
        keep_idx.append(i)
    return [atoms[i] for i in keep_idx]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSONL or JSON-array dialogue file")
    parser.add_argument("--target-speaker", required=True,
                        help="speaker whose personality we are extracting")
    parser.add_argument("--max-atoms", type=int, default=50,
                        help="cap on extracted atoms before dedup")
    parser.add_argument("--backend", default="minimax")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--dedupe-threshold", type=float, default=0.92,
                        help="cosine similarity threshold for dedup (1.0 disables)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[extract] input not found: {in_path}")
        sys.exit(1)
    turns = load_turns(in_path)
    print(f"[extract] loaded {len(turns)} turns; speakers seen: "
          f"{sorted({s for s, _ in turns})}")
    pairs = make_pairs(turns, args.target_speaker, max_pairs=args.max_atoms)
    print(f"[extract] {len(pairs)} prompt->target pairs (target={args.target_speaker!r})")
    if not pairs:
        print("[extract] no pairs; check --target-speaker matches a speaker in the transcript.")
        sys.exit(1)

    client = get_client(args.backend, temperature=args.temperature, max_tokens=512)
    atoms: list[IfThenAtom] = []
    for i, (prompt_turn, response_turn) in enumerate(pairs):
        obj = _extract_one(client, prompt_turn, response_turn)
        if obj is None:
            print(f"  [{i + 1}/{len(pairs)}] skipped (uninformative)")
            continue
        atom = IfThenAtom(
            id=f"extracted_{uuid.uuid4().hex[:8]}",
            situation=str(obj.get("situation", "")).strip(),
            behavior=str(obj.get("behavior", "")).strip(),
            affect=(str(obj["affect"]).strip() or None) if obj.get("affect") else None,
            goal=(str(obj["goal"]).strip() or None) if obj.get("goal") else None,
            target_dimension=None,
            polarity=None,
        )
        if atom.situation and atom.behavior:
            atoms.append(atom)
            print(f"  [{i + 1}/{len(pairs)}] {atom.situation[:70]} -> {atom.behavior[:70]}")

    print(f"[extract] {len(atoms)} raw atoms before dedup")
    if args.dedupe_threshold < 1.0 and atoms:
        atoms = _dedupe_by_similarity(atoms, threshold=args.dedupe_threshold)
    print(f"[extract] {len(atoms)} atoms after dedup")

    corpus = IfThenCorpus(atoms)
    corpus.to_jsonl(args.out)
    print(f"[extract] wrote -> {args.out}")


if __name__ == "__main__":
    main()
