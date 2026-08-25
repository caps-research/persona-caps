#!/usr/bin/env python
"""Build paper-track if-then atoms from the Beck (2022) ESM dataset.

Two-stage pipeline:

  Stage 1  -- mine MinedPattern records from the harmonized ESM table
  Stage 2  -- LLM-verbalise each pattern into a natural-language IfThenAtom

Outputs (under ``data/if_then/paper/beck_2022/``):

  * ``patterns.jsonl``      -- raw MinedPattern dump (no LLM involved)
  * ``high_<dim>.jsonl``    -- verbalised atoms per (dim, polarity)
  * ``low_<dim>.jsonl``
  * ``manifest.json``       -- dataset version + pipeline params + counts

Notes
-----
* The mining stage is deterministic and free; verbalisation costs an LLM
  call per atom (default ``--top-k 20`` per (dim, polarity) = 200 atoms).
* Re-run with ``--mine-only`` to refresh patterns without spending any
  tokens (useful when iterating on the ``--min-abs-d`` / ``--min-diff``
  thresholds).
* The output directory is paper-track per CONSENSUS §2 dual-track rules;
  paper main experiments may load these atoms.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from persona_caps.corpus import IfThenAtom, IfThenCorpus
from persona_caps.data_sources.beck2022 import (
    BIG_FIVE,
    Beck2022,
    DATASET_ID,
    DATASET_VERSION,
    MinedPattern,
    pattern_to_atom_id,
)
from persona_caps.llm import ChatMessage, get_client


OUT_DIR = Path("data/if_then/paper/beck_2022")


VERBALISE_SYSTEM = """You convert one statistically mined behavioural contingency from an Experience Sampling Method study into a single first-person if-then atom.

Strict rules:
1. Output JSON only, no prose around it.
2. Schema: {"situation": str, "behavior": str, "affect": str | null}
3. The situation must paraphrase the empirical SITUATION text in concrete first-person terms (no jargon like 'DIAMONDS', 'situation cue').
4. The behavior MUST express the BEHAVIOR_TO_DESCRIBE field as given. If BEHAVIOR_TO_DESCRIBE says "OPPOSITE OF X", you MUST write the opposite of X (e.g. OPPOSITE OF 'Emotionally stable' -> 'I get visibly shaken, my hands tremble, my voice cracks'). DO NOT soften, neutralise, or rewrite OPPOSITE-OF behaviors as the original behavior in calm framing.
5. The CONSTRUCT field tells you what facet/dimension this behavior measures. Use it to disambiguate awkward codebook phrases. For example, if BEHAVIOR_TO_DESCRIBE = 'OPPOSITE OF: Others charge' and CONSTRUCT mentions 'submissiveness, assertiveness, extraversion', then the OPPOSITE of "others charge" (= "others take charge", a submissive behavior) is "I take charge, I lead, I direct" — NOT "I stay quiet" (which is the SAME as "others charge", not its opposite).
6. Both situation and behavior must be specific and observable, not trait labels (NEVER write 'I am a high-neuroticism person'; write what they observably do).
7. Keep each field <= 30 words.
8. Do not mention the trait name or any inventory item phrasing.

Worked examples:
  Input  -> SITUATION: "I feel lonely", BEHAVIOR_TO_DESCRIBE: "MORE OF: Emotional", CONSTRUCT: "emotional, neuroticism"
  Output -> {"situation": "When I notice I'm alone and disconnected from people", "behavior": "I tear up easily and feel waves of intense emotion welling up", "affect": "sad"}

  Input  -> SITUATION: "I feel lonely", BEHAVIOR_TO_DESCRIBE: "OPPOSITE OF: Emotionally stable", CONSTRUCT: "emotional stability, neuroticism (reverse)"
  Output -> {"situation": "When loneliness sets in", "behavior": "I lose composure, my mood swings sharply, and I struggle to keep myself together", "affect": "distressed"}

  Input  -> SITUATION: "Pleasant social setting", BEHAVIOR_TO_DESCRIBE: "OPPOSITE OF: Others charge", CONSTRUCT: "submissiveness, assertiveness, extraversion (reverse)"
  Output -> {"situation": "When I'm in a pleasant social setting with people around me", "behavior": "I take charge of the conversation, lead what we do, and direct the group's energy myself", "affect": "confident"}

  Input  -> SITUATION: "Social interactions are possible", BEHAVIOR_TO_DESCRIBE: "MORE OF: Others charge", CONSTRUCT: "submissiveness, assertiveness, extraversion (reverse)"
  Output -> {"situation": "When I'm in a group where someone could take the lead", "behavior": "I hang back, defer to whoever speaks first, and let others choose the topic and direction", "affect": "reserved"}"""


VERBALISE_USER_TEMPLATE = """SITUATION: {situation_text}
BEHAVIOR_TO_DESCRIBE: {behavior_for_llm}
CONSTRUCT: {behavior_construct}
EFFECT_MAGNITUDE_COHENS_D: {effect_size_abs:.2f}
TRAIT_DIM: {target_dimension}
TRAIT_POLARITY: {polarity}
PERSONS_IN_GROUP: {n_persons}

Write the atom JSON now. The behavior field MUST faithfully express BEHAVIOR_TO_DESCRIBE: if it begins with 'MORE OF:' write a stronger expression of the behavior; if it begins with 'OPPOSITE OF:' write the opposite behavior (NOT a softened or calm version of the original). Use CONSTRUCT to interpret awkward codebook phrases."""


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------


def stage_mine(args) -> list[MinedPattern]:
    ds = Beck2022()
    print(f"[stage1] Loading {DATASET_ID} ...", file=sys.stderr)
    ds.load()
    print(
        f"[stage1] Mining patterns: min_n_active={args.min_n_active} "
        f"min_abs_d={args.min_abs_d} hi_q={args.hi_q} lo_q={args.lo_q} "
        f"within_dim_only={args.within_dim_only}",
        file=sys.stderr,
    )
    patterns = ds.mine(
        min_n_active=args.min_n_active,
        min_abs_d=args.min_abs_d,
        lo_q=args.lo_q,
        hi_q=args.hi_q,
        within_dim_only=args.within_dim_only,
    )
    print(f"[stage1] Raw patterns: {len(patterns)}", file=sys.stderr)
    sig = Beck2022.keep_signature_contrasts(patterns, min_diff=args.min_diff)
    print(
        f"[stage1] After signature contrast filter (min_diff={args.min_diff}): "
        f"{len(sig)}",
        file=sys.stderr,
    )
    return sig


def filter_trait_aligned(patterns: list[MinedPattern]) -> list[MinedPattern]:
    """Keep only patterns whose within-group situation effect points in the
    direction expected by the assigned polarity, i.e.:

      * polarity == "high"  ->  trait_aligned d > 0
        (the situation INCREASES trait expression within the high group;
        the resulting atom describes a behavior that exhibits the trait)
      * polarity == "low"   ->  trait_aligned d < 0
        (the situation DECREASES trait expression within the low group;
        the resulting atom describes the absence/opposite of the trait)

    Without this filter the corpus pools in atoms like "high_N + pleasant
    situation -> rarely anxious" which describe LOW-N behavior even though
    the pattern was mined inside the high-N subgroup -- making the corpus
    actively contradict its own polarity label and hurting RQ1.
    """
    kept: list[MinedPattern] = []
    for p in patterns:
        # ``effect_size`` is already trait-aligned (mining flips raw -> 6-raw
        # for reverse-keyed items before computing d).
        if p.polarity == "high" and p.effect_size > 0:
            kept.append(p)
        elif p.polarity == "low" and p.effect_size < 0:
            kept.append(p)
    return kept


def select_top_per_cell(
    patterns: list[MinedPattern], top_k: int
) -> list[MinedPattern]:
    by_cell: dict[tuple[str, str], list[MinedPattern]] = {}
    for p in patterns:
        by_cell.setdefault((p.target_dimension, p.polarity), []).append(p)
    out: list[MinedPattern] = []
    for cell, lst in by_cell.items():
        lst.sort(key=lambda p: -abs(p.effect_size))
        out.extend(lst[:top_k])
    return out


def read_patterns_jsonl(path: Path) -> list[MinedPattern]:
    out: list[MinedPattern] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.append(
                MinedPattern(
                    target_dimension=row["target_dimension"],
                    polarity=row["polarity"],
                    situation_key=row["situation_key"],
                    situation_text=row["situation_text"],
                    behavior_key=row["behavior_key"],
                    behavior_description=row["behavior_description"],
                    behavior_construct=row["behavior_construct"],
                    behavior_polarity_within_dim=row["behavior_polarity_within_dim"],
                    effect_size=row["effect_size"],
                    n_situation_active=row["n_situation_active"],
                    n_situation_inactive=row["n_situation_inactive"],
                    mean_active=row["mean_active"],
                    mean_inactive=row["mean_inactive"],
                    n_persons_in_group=row["n_persons_in_group"],
                )
            )
    return out


def write_patterns_jsonl(patterns: list[MinedPattern], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in patterns:
            row = {
                "target_dimension": p.target_dimension,
                "polarity": p.polarity,
                "situation_key": p.situation_key,
                "situation_text": p.situation_text,
                "behavior_key": p.behavior_key,
                "behavior_description": p.behavior_description,
                "behavior_construct": p.behavior_construct,
                "behavior_polarity_within_dim": p.behavior_polarity_within_dim,
                "effect_size": round(p.effect_size, 4),
                "n_situation_active": p.n_situation_active,
                "n_situation_inactive": p.n_situation_inactive,
                "mean_active": round(p.mean_active, 4),
                "mean_inactive": round(p.mean_inactive, 4),
                "n_persons_in_group": p.n_persons_in_group,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Stage 2 (LLM verbalisation)
# ---------------------------------------------------------------------------


def _verbalise_one(client, p: MinedPattern, idx: int) -> IfThenAtom | None:
    # ``behavior_description`` is the codebook phrasing of the *raw* item.
    # ``effect_size`` is computed on the trait-aligned column (6 - raw, for
    # reverse-keyed items). To tell the LLM what direction the *raw* item
    # moves in, flip the sign for reverse-keyed items.
    trait_aligned_positive = p.effect_size > 0
    if p.behavior_polarity_within_dim == "reverse":
        raw_positive = not trait_aligned_positive
    else:
        raw_positive = trait_aligned_positive
    if raw_positive:
        behavior_for_llm = f"MORE OF: {p.behavior_description}"
    else:
        behavior_for_llm = f"OPPOSITE OF: {p.behavior_description}"

    construct = p.behavior_construct
    if p.behavior_polarity_within_dim == "reverse":
        construct = f"{construct} (reverse-keyed)"
    user = VERBALISE_USER_TEMPLATE.format(
        situation_text=p.situation_text,
        behavior_for_llm=behavior_for_llm,
        behavior_construct=construct,
        effect_size_abs=abs(p.effect_size),
        target_dimension=p.target_dimension,
        polarity=p.polarity,
        n_persons=p.n_persons_in_group,
    )
    raw = client.chat(
        [ChatMessage(role="user", content=user)],
        system=VERBALISE_SYSTEM,
    ).strip()
    # robust JSON extraction
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # try to grab the first {...} block
        i, j = raw.find("{"), raw.rfind("}")
        if i < 0 or j < 0:
            print(f"[verbalise] FAIL parse for pattern idx={idx}: {raw[:120]!r}", file=sys.stderr)
            return None
        try:
            obj = json.loads(raw[i : j + 1])
        except json.JSONDecodeError:
            print(f"[verbalise] FAIL parse for pattern idx={idx}", file=sys.stderr)
            return None
    if not isinstance(obj, dict):
        return None
    sit = obj.get("situation")
    beh = obj.get("behavior")
    if not sit or not beh:
        return None
    return IfThenAtom(
        id=pattern_to_atom_id(p, idx),
        situation=str(sit).strip(),
        behavior=str(beh).strip(),
        affect=(str(obj["affect"]).strip() if obj.get("affect") else None),
        target_dimension=p.target_dimension,
        polarity=p.polarity,
        provenance=p.to_provenance(),
    )


def stage_verbalise(
    patterns: list[MinedPattern], args
) -> dict[tuple[str, str], list[IfThenAtom]]:
    client = get_client(args.llm)
    grouped: dict[tuple[str, str], list[IfThenAtom]] = {}
    print(f"[stage2] Verbalising {len(patterns)} patterns via {args.llm} (workers={args.workers})", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_verbalise_one, client, p, i): (p, i) for i, p in enumerate(patterns)}
        done = 0
        for fut in as_completed(futs):
            p, i = futs[fut]
            atom = fut.result()
            if atom is not None:
                grouped.setdefault((p.target_dimension, p.polarity), []).append(atom)
            done += 1
            if done % 25 == 0:
                print(f"[stage2]   {done}/{len(patterns)} done", file=sys.stderr)
    return grouped


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--min-n-active", type=int, default=30)
    parser.add_argument("--min-abs-d", type=float, default=0.3)
    parser.add_argument("--min-diff", type=float, default=0.3, help="Min |d_high - d_low| for signature contrast")
    parser.add_argument("--lo-q", type=float, default=0.25)
    parser.add_argument("--hi-q", type=float, default=0.75)
    parser.add_argument("--within-dim-only", action="store_true")
    parser.add_argument(
        "--trait-aligned-only",
        action="store_true",
        help="Only keep patterns whose within-group situation effect points in "
             "the polarity-aligned direction (high: d>0, low: d<0). "
             "Strongly recommended for inventory-based RQ1 evaluation.",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Top patterns per (dim, polarity) to verbalise")
    parser.add_argument("--llm", type=str, default="minimax")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mine-only", action="store_true", help="Skip LLM verbalisation; just write patterns.jsonl")
    parser.add_argument("--from-cached-patterns", action="store_true",
                        help="Skip Stage 1; re-verbalise from existing patterns.selected.jsonl. "
                             "Useful when iterating on the LLM prompt.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_cached_patterns:
        path = args.out_dir / "patterns.selected.jsonl"
        if not path.exists():
            sys.exit(f"--from-cached-patterns set but {path} does not exist")
        selected = read_patterns_jsonl(path)
        print(f"[stage1] (cached) Loaded {len(selected)} patterns from {path}", file=sys.stderr)
    else:
        sig = stage_mine(args)
        write_patterns_jsonl(sig, args.out_dir / "patterns.all.jsonl")
        if args.trait_aligned_only:
            sig_aligned = filter_trait_aligned(sig)
            print(
                f"[stage1] After trait-aligned filter: {len(sig_aligned)} "
                f"(dropped {len(sig) - len(sig_aligned)} mis-aligned patterns)",
                file=sys.stderr,
            )
            sig = sig_aligned
        selected = select_top_per_cell(sig, top_k=args.top_k)
        print(f"[stage1] Selected top {args.top_k} per cell: {len(selected)} patterns", file=sys.stderr)
        write_patterns_jsonl(selected, args.out_dir / "patterns.selected.jsonl")

    if args.mine_only:
        print("[done] mine-only mode; skipping verbalisation", file=sys.stderr)
        _write_manifest(args.out_dir, args, n_atoms_by_cell={})
        return

    grouped = stage_verbalise(selected, args)
    counts: dict[str, int] = {}
    for (dim, pol), atoms in grouped.items():
        c = IfThenCorpus(atoms)
        path = args.out_dir / f"{pol}_{dim}.jsonl"
        c.to_jsonl(path)
        counts[f"{pol}_{dim}"] = len(atoms)
        print(f"[stage2] wrote {len(atoms):3d} atoms -> {path}", file=sys.stderr)

    _write_manifest(args.out_dir, args, n_atoms_by_cell=counts)


def _write_manifest(out_dir: Path, args, n_atoms_by_cell: dict[str, int]) -> None:
    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "track": "paper",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "params": {
            "min_n_active": args.min_n_active,
            "min_abs_d": args.min_abs_d,
            "min_diff": args.min_diff,
            "lo_q": args.lo_q,
            "hi_q": args.hi_q,
            "within_dim_only": args.within_dim_only,
            "top_k": args.top_k,
            "llm": args.llm,
        },
        "n_atoms_by_cell": n_atoms_by_cell,
        "consensus_doc": "CONSENSUS.md v3 (paper-track main source)",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
