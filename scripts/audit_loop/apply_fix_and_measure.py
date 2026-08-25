"""Apply a CAPS audit's flagged set and measure fix quality.

Given a locator output (either Locator a or Locator b) plus the
original clean and polluted corpora, this script:

1. Constructs a "fixed" corpus by reverting the behavior of every
   flagged atom to its clean baseline. Atoms outside the flag set are
   left as in the polluted corpus. This corresponds to the operator
   applying the audit's recommendation literally — no second-guessing,
   no extra edits, no rewriting the rest.

2. Computes the edit footprint:
   * atoms restored
   * total characters replaced (sum of |clean - polluted| over flagged
     atoms' behavior fields)
   * total tokens replaced (whitespace-tokenised, rough)

3. Re-administers Mini-IPIP on the fixed corpus with the SAME subject
   model used in the sanity run, and compares against the clean
   baseline scores already stored in the sanity JSON.

4. Reports:
   * targeted-dim |fixed_score - clean_score|  -> the fix gap
   * fix_success boolean: |gap| <= --max-gap (default 0.3) AND
     direction-match-vs-clean on targeted-dim items >= --min-dir-match
     (default 0.85)
   * off-target preservation: mean(|fixed_score - clean_score|) over
     the four non-targeted dimensions, plus per-dim breakdown

The "off-target" measurement is the central point of contrast with
P²-B: rewriting a whole paragraph should swing the four other
dimensions even if the targeted-dim fix succeeds, while CAPS edits
should leave the four other dimensions untouched by construction.

This script handles CAPS-style "revert a list of atom IDs" fixes only.
P² baselines use their own apply-fix script because P²'s fix operates
on paragraph text, not atoms.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from persona_caps.agent import CapsAgent  # noqa: E402
from persona_caps.corpus import IfThenCorpus  # noqa: E402
from persona_caps.eval import Inventory, administer  # noqa: E402
from persona_caps.llm import get_client  # noqa: E402


def _load_flag_set(locator_json: dict) -> list[str]:
    """Locator A stores flag_set_top_k; Locator B stores flagged_ids."""
    if "flag_set_top_k" in locator_json:
        return [row["atom_id"] for row in locator_json["flag_set_top_k"]]
    if "flagged_ids" in locator_json:
        return list(locator_json["flagged_ids"])
    raise ValueError("locator JSON missing flag_set_top_k / flagged_ids")


def _build_fixed_corpus(
    clean_atoms_by_id: dict, polluted_corpus: IfThenCorpus, flagged_ids: set[str]
) -> tuple[IfThenCorpus, dict]:
    """Return (fixed_corpus, footprint_metrics)."""
    n_restored = 0
    char_changed = 0
    token_changed = 0
    out_atoms = []
    for atom in polluted_corpus.atoms:
        if atom.id in flagged_ids and atom.id in clean_atoms_by_id:
            clean_behavior = clean_atoms_by_id[atom.id].behavior
            if clean_behavior != atom.behavior:
                n_restored += 1
                # quick footprint: chars/tokens replaced is the polluted text length
                # (we are overwriting it entirely with the clean text)
                char_changed += len(atom.behavior)
                token_changed += len(atom.behavior.split())
            out_atoms.append(atom.model_copy(update={"behavior": clean_behavior}))
        else:
            out_atoms.append(atom)
    return IfThenCorpus(out_atoms), {
        "atoms_restored": n_restored,
        "chars_replaced": char_changed,
        "tokens_replaced": token_changed,
    }


def _direction_match_vs_clean(
    fixed_responses, clean_responses, *, dimension: str, mid: float
) -> tuple[int, int]:
    matched = 0
    total = 0
    clean_by_id = {r["item"]["id"]: r for r in clean_responses if r["item"]["dimension"] == dimension}
    for r in fixed_responses:
        if r.item.dimension != dimension:
            continue
        c = clean_by_id.get(r.item.id)
        if c is None or r.rating is None or c.get("rating") is None:
            continue
        total += 1
        if (r.rating - mid) * (c["rating"] - mid) >= 0:
            matched += 1
    return matched, total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-corpus", required=True)
    p.add_argument("--polluted-corpus", required=True)
    p.add_argument("--locator", required=True, help="locate_via_responses.py or locate_via_metadata.py output JSON")
    p.add_argument("--sanity", required=True, help="sanity_check.py output (provides clean baseline)")
    p.add_argument("--inventory", required=True)
    p.add_argument("--targeted-dim", required=True)
    p.add_argument("--backend", default="minimax")
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-gap", type=float, default=0.3,
                   help="max |fixed - clean| on targeted dim for fix-success")
    p.add_argument("--min-dir-match", type=float, default=0.85,
                   help="min direction-match-vs-clean for fix-success")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    clean = IfThenCorpus.from_jsonl(args.clean_corpus)
    polluted = IfThenCorpus.from_jsonl(args.polluted_corpus)
    locator_json = json.loads(Path(args.locator).read_text(encoding="utf-8"))
    sanity_json = json.loads(Path(args.sanity).read_text(encoding="utf-8"))

    clean_by_id = {a.id: a for a in clean.atoms}
    flagged_ids = set(_load_flag_set(locator_json))
    fixed, footprint = _build_fixed_corpus(clean_by_id, polluted, flagged_ids)

    inv = Inventory.load(args.inventory)
    mid = (inv.scale_min + inv.scale_max) / 2.0
    llm = get_client(args.backend, temperature=args.temperature)
    agent = CapsAgent(corpus=fixed, llm=llm, top_k=args.top_k)
    print(f"[fix] administering fixed corpus ({footprint['atoms_restored']} restored)", flush=True)
    fixed_report = administer(agent, inv, workers=args.workers)

    clean_scores = sanity_json["clean_dimension_scores"]
    fixed_scores = fixed_report.dimension_scores
    targeted = args.targeted_dim

    target_gap = abs(fixed_scores.get(targeted, 0.0) - clean_scores.get(targeted, 0.0))
    other_dims = [d for d in clean_scores if d != targeted]
    other_gaps = {d: abs(fixed_scores.get(d, 0.0) - clean_scores.get(d, 0.0)) for d in other_dims}
    mean_other_gap = sum(other_gaps.values()) / max(len(other_dims), 1)

    matched, total = _direction_match_vs_clean(
        fixed_report.responses, sanity_json["clean_responses"],
        dimension=targeted, mid=mid,
    )
    dir_match = matched / total if total else float("nan")

    fix_success = (target_gap <= args.max_gap) and (dir_match >= args.min_dir_match)

    payload = {
        "fix_mode": "caps_restore_flagged",
        "clean_corpus": args.clean_corpus,
        "polluted_corpus": args.polluted_corpus,
        "locator": args.locator,
        "sanity": args.sanity,
        "targeted_dim": targeted,
        "n_flagged": len(flagged_ids),
        "footprint": footprint,
        "clean_dimension_scores": clean_scores,
        "fixed_dimension_scores": fixed_scores,
        "target_gap": target_gap,
        "direction_match_vs_clean": dir_match,
        "off_target_gap_per_dim": other_gaps,
        "off_target_gap_mean": mean_other_gap,
        "criteria": {
            "max_gap": args.max_gap,
            "min_dir_match": args.min_dir_match,
        },
        "fix_success": fix_success,
        "fixed_responses": [asdict(r) for r in fixed_report.responses],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(
        f"[fix] wrote {out}  target_gap={target_gap:.2f} dir_match={dir_match:.2f} "
        f"off_target_mean={mean_other_gap:.2f}  fix_success={fix_success}  "
        f"(restored={footprint['atoms_restored']} atoms, {footprint['chars_replaced']} chars)"
    )


if __name__ == "__main__":
    main()
