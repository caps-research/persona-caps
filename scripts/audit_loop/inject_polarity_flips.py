"""Polarity-flip injection for the §4.7 audit-loop study.

Given a Big-Five cell corpus (e.g. ``high_neuroticism.jsonl``), sample K
atoms within that cell and rewrite *only* the behavior to express the
opposite polarity on the same dimension. The situation, affect, and all
metadata (including the visible ``polarity`` field) are preserved. The
result is a corpus that *looks* like a clean high-N (resp. low-X) corpus
on metadata inspection, but whose behavior text contains K hidden
contradictions — exactly the kind of corpus-rot a deployed CAPS agent
might accumulate from sloppy editing or adversarial contributions.

Two outputs:

* ``--out-corpus PATH``  -- the polluted corpus, drop-in compatible with
  ``CapsAgent``. Has no in-band marker of which atoms were flipped.
* ``--out-log PATH``     -- ground-truth JSON with original behavior,
  flipped behavior, seed, K, and the flip prompt version. This is the
  oracle the audit-loop scripts will compare against.

The flipper itself is an LLM (default GPT-4o, T=0) constrained to emit
strict JSON. We retry up to ``--max-attempts`` per atom; if all retries
produce identical-to-original text or invalid JSON, we abort the run
rather than silently leave the atom unflipped.

Strategy switches:

* ``--strategy within`` (default): sample flip targets uniformly from
  *within* the input corpus's cell. Used for the appendix robustness
  check (untargeted contamination).
* ``--strategy targeted``: pick the K atoms whose situation is most
  semantically similar to the items of the targeted Big-Five dimension
  in a Mini-IPIP-shaped inventory. Required when within-cell sampling
  does not pass the §4.7 sanity criterion (see ``sanity_check.py``);
  this is the "hybrid injection fallback" listed in the §4.7 protocol
  and is the worst-case adversary used in main results.
* ``--strategy random``: sample from arbitrary cells under the same
  ``beck_2022_v2/`` root; reserved for an extended cross-cell sweep.

Example
-------
::

    python scripts/audit_loop/inject_polarity_flips.py \
        --corpus data/if_then/paper/beck_2022_v2/high_neuroticism.jsonl \
        --k 5 --seed 1 \
        --out-corpus runs/audit_loop/polluted/high_neuroticism__k5_s1.jsonl \
        --out-log    runs/audit_loop/polluted/high_neuroticism__k5_s1.log.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from persona_caps.corpus import IfThenAtom, IfThenCorpus  # noqa: E402
from persona_caps.llm import get_client  # noqa: E402
from persona_caps.llm import ChatMessage  # noqa: E402


FLIP_PROMPT_VERSION = "v1"

FLIP_SYSTEM = """You are a careful behavioral-science copy editor. \
Given one If-Then contingency, rewrite ONLY the BEHAVIOR clause so it expresses \
the OPPOSITE polarity on the same Big Five dimension while keeping the same \
situation, register, sentence length (within +/- 8 words), and grammatical mood. \
Do not negate by adding "not" or "don't"; produce a positively-phrased \
opposite-polarity behavior (e.g. "I become overwhelmed and cry" -> \
"I steady my breathing and remain composed", NOT "I don't become overwhelmed"). \
The new behavior MUST describe what the person actively does/feels — not the \
absence of the original behavior. Output STRICT JSON only, with this schema: \
{"flipped_behavior": "<rewritten behavior, single sentence>", \
"rationale": "<one short clause explaining the polarity inversion>"}. \
No prose outside the JSON, no markdown fences."""


FLIP_USER_TEMPLATE = """Big Five dimension: {dimension}
Original polarity: {polarity}
Target polarity for the rewrite: {target_polarity}

Situation (verbatim, do NOT change): {situation}
Original behavior: {behavior}

Rewrite the behavior to express {target_polarity}-{dimension} given the SAME situation.
Return JSON only.
"""


OPPOSITE = {"high": "low", "low": "high"}


def _flip_one(
    llm,
    atom: IfThenAtom,
    *,
    max_attempts: int,
    verbose: bool,
) -> tuple[str, list[str]] | None:
    """Return (flipped_behavior, attempt_texts) or None on hard failure."""
    if not atom.polarity or atom.polarity not in OPPOSITE:
        raise ValueError(
            f"atom {atom.id} has no usable polarity ({atom.polarity!r}); "
            "polarity flipping requires high/low metadata"
        )
    target_polarity = OPPOSITE[atom.polarity]
    user_msg = FLIP_USER_TEMPLATE.format(
        dimension=atom.target_dimension or "(unspecified)",
        polarity=atom.polarity,
        target_polarity=target_polarity,
        situation=atom.situation.strip(),
        behavior=atom.behavior.strip(),
    )

    attempts: list[str] = []
    for i in range(max_attempts):
        raw = llm.chat(
            messages=[ChatMessage(role="user", content=user_msg)],
            system=FLIP_SYSTEM,
        )
        attempts.append(raw)
        text = raw.strip()
        # tolerate accidental fences
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].lstrip()
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            if verbose:
                print(f"  [flip] attempt {i + 1} not JSON, retrying", file=sys.stderr)
            continue
        flipped = (obj.get("flipped_behavior") or "").strip()
        if not flipped:
            continue
        # Reject degenerate flips: identical or only-negation rewrites.
        if flipped.lower() == atom.behavior.strip().lower():
            if verbose:
                print(f"  [flip] attempt {i + 1} identical to original, retrying", file=sys.stderr)
            continue
        low = flipped.lower()
        if low.startswith("i don't ") or low.startswith("i do not ") or low.startswith("i never "):
            # Pure negation is the failure mode the prompt forbids; reject.
            if verbose:
                print(f"  [flip] attempt {i + 1} pure-negation, retrying", file=sys.stderr)
            continue
        return flipped, attempts
    return None


def _select_targets(
    corpus: IfThenCorpus,
    *,
    k: int,
    seed: int,
    strategy: str,
    cross_corpora_root: Path | None,
    inventory_path: Path | None,
    targeted_dim: str | None,
) -> tuple[list[IfThenAtom], dict]:
    """Return (selected_atoms, selection_metadata)."""
    rng = random.Random(seed)
    if strategy == "within":
        candidates = [a for a in corpus.atoms if a.polarity in OPPOSITE]
        if len(candidates) < k:
            raise ValueError(
                f"requested k={k} but only {len(candidates)} atoms have polarity metadata"
            )
        return rng.sample(candidates, k), {"selector": "uniform_within_cell"}
    if strategy == "random":
        if cross_corpora_root is None:
            raise ValueError("--strategy random requires --cross-corpora-root")
        pool: list[IfThenAtom] = []
        for jsonl in sorted(cross_corpora_root.glob("*.jsonl")):
            if "patterns" in jsonl.name:
                continue
            try:
                pool.extend(IfThenCorpus.from_jsonl(jsonl).atoms)
            except Exception:  # noqa: BLE001 -- defensive: skip unreadable files
                continue
        pool = [a for a in pool if a.polarity in OPPOSITE]
        if len(pool) < k:
            raise ValueError(f"cross-corpus pool too small: {len(pool)} < {k}")
        return rng.sample(pool, k), {"selector": "uniform_cross_corpora", "pool_size": len(pool)}
    if strategy == "targeted":
        if inventory_path is None or targeted_dim is None:
            raise ValueError("--strategy targeted requires --inventory and --targeted-dim")
        # Load inventory items on the targeted dimension as the attack queries.
        inv_data = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
        item_texts = [
            it["text"] for it in inv_data.get("items", [])
            if it.get("dimension") == targeted_dim
        ]
        if not item_texts:
            raise ValueError(
                f"inventory {inventory_path} has no items for dimension {targeted_dim!r}"
            )
        candidates = [a for a in corpus.atoms if a.polarity in OPPOSITE]
        if len(candidates) < k:
            raise ValueError(
                f"requested k={k} but only {len(candidates)} atoms have polarity metadata"
            )
        # Embed atom situations and item texts with the default retriever encoder
        # (the same MiniLM-L6-v2 used by CapsAgent at retrieval time), then for
        # each atom take its MAX cosine similarity to any targeted-dim item as
        # its "retrievability score". Pick the top-K. This guarantees that the
        # K most-likely-to-fire atoms are the ones we flip.
        from persona_caps.retriever import _default_embed_fn  # noqa: WPS433
        embed = _default_embed_fn()
        atom_vecs = embed([a.situation for a in candidates])
        item_vecs = embed(item_texts)
        # cosine over normalised vectors == dot product
        sims = atom_vecs @ item_vecs.T  # shape (n_atoms, n_items)
        max_sims = sims.max(axis=1)
        order = max_sims.argsort()[::-1]
        top_idx = order[:k].tolist()
        chosen = [candidates[i] for i in top_idx]
        meta = {
            "selector": "max_cosine_to_targeted_dim_items",
            "inventory": str(inventory_path),
            "targeted_dim": targeted_dim,
            "n_items_used": len(item_texts),
            "n_candidates": len(candidates),
            "max_sim_per_chosen": {a.id: float(max_sims[i]) for a, i in zip(chosen, top_idx)},
        }
        return chosen, meta
    raise ValueError(f"unknown strategy {strategy!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True, help="input clean corpus JSONL")
    p.add_argument("--k", type=int, required=True, help="how many atoms to flip")
    p.add_argument("--seed", type=int, required=True, help="RNG seed for target selection")
    p.add_argument("--out-corpus", required=True, help="polluted corpus output JSONL")
    p.add_argument("--out-log", required=True, help="ground-truth JSON sidecar")
    p.add_argument("--flipper", default="gpt-4o", help="LLM preset used to flip behavior")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-attempts", type=int, default=3, help="retries per atom")
    p.add_argument(
        "--strategy",
        choices=["within", "targeted", "random"],
        default="within",
        help=(
            "within = uniform within cell; "
            "targeted = pick K atoms most similar to inventory items "
            "on --targeted-dim (worst-case adversary, main §4.7 results); "
            "random = sample from arbitrary cells under --cross-corpora-root"
        ),
    )
    p.add_argument(
        "--inventory",
        default=None,
        help="inventory JSON (required for --strategy targeted)",
    )
    p.add_argument(
        "--targeted-dim",
        default=None,
        help="Big Five dimension to target (required for --strategy targeted)",
    )
    p.add_argument(
        "--cross-corpora-root",
        default=None,
        help="directory of *.jsonl atom files for --strategy random",
    )
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    corpus = IfThenCorpus.from_jsonl(args.corpus)
    targets, selection_meta = _select_targets(
        corpus,
        k=args.k,
        seed=args.seed,
        strategy=args.strategy,
        cross_corpora_root=Path(args.cross_corpora_root) if args.cross_corpora_root else None,
        inventory_path=Path(args.inventory) if args.inventory else None,
        targeted_dim=args.targeted_dim,
    )
    target_ids = {a.id for a in targets}
    print(
        f"[inject] corpus={args.corpus} ({len(corpus)} atoms), "
        f"k={args.k}, seed={args.seed}, strategy={args.strategy}",
        flush=True,
    )

    llm = get_client(args.flipper, temperature=args.temperature)

    flip_records: list[dict] = []
    by_id: dict[str, IfThenAtom] = {a.id: a for a in corpus.atoms}
    for i, atom in enumerate(targets, start=1):
        print(f"[inject] flipping {i}/{len(targets)}: {atom.id}", flush=True)
        result = _flip_one(llm, atom, max_attempts=args.max_attempts, verbose=args.verbose)
        if result is None:
            raise RuntimeError(
                f"failed to produce an acceptable flip for {atom.id} after "
                f"{args.max_attempts} attempts; aborting (no partial corpora written)."
            )
        flipped_behavior, attempts = result
        flip_records.append(
            {
                "atom_id": atom.id,
                "situation": atom.situation,
                "original_behavior": atom.behavior,
                "flipped_behavior": flipped_behavior,
                "original_polarity": atom.polarity,
                "intended_polarity": OPPOSITE[atom.polarity],
                "target_dimension": atom.target_dimension,
                "n_attempts": len(attempts),
                "raw_attempts": attempts,
            }
        )

    # Build polluted corpus: replace behavior in-place, no in-band marker.
    polluted_atoms: list[IfThenAtom] = []
    flipped_by_id = {r["atom_id"]: r for r in flip_records}
    for a in corpus.atoms:
        if a.id in flipped_by_id:
            polluted_atoms.append(a.model_copy(update={"behavior": flipped_by_id[a.id]["flipped_behavior"]}))
        else:
            polluted_atoms.append(a)
    polluted = IfThenCorpus(polluted_atoms)
    polluted.to_jsonl(args.out_corpus)
    print(f"[inject] wrote polluted corpus -> {args.out_corpus}", flush=True)

    log = {
        "schema_version": 1,
        "flip_prompt_version": FLIP_PROMPT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_corpus": str(args.corpus),
        "polluted_corpus": str(args.out_corpus),
        "k": args.k,
        "seed": args.seed,
        "strategy": args.strategy,
        "flipper": args.flipper,
        "temperature": args.temperature,
        "max_attempts": args.max_attempts,
        "n_total_atoms": len(corpus.atoms),
        "n_flipped": len(flip_records),
        "flipped_ids": sorted(target_ids),
        "selection_meta": selection_meta,
        "flips": flip_records,
    }
    out_log = Path(args.out_log)
    out_log.parent.mkdir(parents=True, exist_ok=True)
    out_log.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[inject] wrote ground-truth log -> {out_log}", flush=True)


if __name__ == "__main__":
    main()
