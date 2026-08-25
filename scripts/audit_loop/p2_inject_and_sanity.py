"""P² baseline: render a corpus as a single paragraph persona, inject
the same polarity-flipped behaviors used by the CAPS injection, and
sanity-check that the polluted paragraph actually breaks the agent's
Mini-IPIP score on the targeted dimension.

This is the persona-as-text baseline that §4.7 compares against. To
keep the comparison strict, the P² persona text is constructed by
rendering each if-then atom as a single sentence
("WHEN <situation> -> <behavior>"), in corpus order, separated by
periods. The polluted paragraph is built the same way but from the
already-polluted corpus produced by ``inject_polarity_flips.py``, so
the K sentence-level "edits" exactly track the K atom-level flips
that CAPS sees. This means an audit comparing CAPS vs P² is comparing
two delivery formats over the *same* underlying persona content.

The agent used is ``PersonaDescriptionAgent`` (no retrieval; full
paragraph in the system prompt), and we re-use the same
``InventoryReport``/``administer`` pipeline as CAPS so all sanity
metrics are comparable across the two methods.

Sanity criteria are identical to the CAPS sanity check
(``|delta_score| >= 1.0`` AND ``direction_match <= 0.70``); this
script writes its sanity JSON in the same shape as ``sanity_check.py``
so downstream tools (``apply_fix_and_measure.py``, etc.) can consume
either.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from persona_caps.agent import PersonaDescriptionAgent  # noqa: E402
from persona_caps.corpus import IfThenAtom, IfThenCorpus  # noqa: E402
from persona_caps.eval import Inventory, administer  # noqa: E402
from persona_caps.llm import get_client  # noqa: E402


def render_paragraph(corpus: IfThenCorpus) -> str:
    """Render each atom as a single sentence; concat in corpus order."""
    sentences: list[str] = []
    for atom in corpus.atoms:
        # Strip terminal punctuation so we can re-attach a period uniformly.
        sit = atom.situation.strip().rstrip(".")
        beh = atom.behavior.strip().rstrip(".")
        sentences.append(f"When {sit[0].lower() + sit[1:]}, {beh[0].lower() + beh[1:]}.")
    return " ".join(sentences)


def _direction_match(
    clean_responses, polluted_responses, *, dimension: str, mid: float
) -> tuple[int, int]:
    matched = 0
    total = 0
    clean_by_id = {r.item.id: r for r in clean_responses if r.item.dimension == dimension}
    poll_by_id = {r.item.id: r for r in polluted_responses if r.item.dimension == dimension}
    for iid, cr in clean_by_id.items():
        pr = poll_by_id.get(iid)
        if pr is None or cr.rating is None or pr.rating is None:
            continue
        total += 1
        if (cr.rating - mid) * (pr.rating - mid) >= 0:
            matched += 1
    return matched, total


def _administer(persona_text: str, *, inv: Inventory, llm, workers: int):
    agent = PersonaDescriptionAgent(persona_text, llm=llm)
    return administer(agent, inv, workers=workers)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-corpus", required=True)
    p.add_argument("--polluted-corpus", required=True)
    p.add_argument("--inventory", required=True)
    p.add_argument("--backend", default="minimax")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--targeted-dim", required=True)
    p.add_argument("--min-score-shift", type=float, default=1.0)
    p.add_argument("--max-direction-match", type=float, default=0.70)
    p.add_argument("--out-paragraphs", required=True,
                   help="JSON with the clean & polluted paragraph text (for fix scripts)")
    p.add_argument("--out-sanity", required=True,
                   help="sanity JSON parallel to CAPS sanity_check.py output")
    args = p.parse_args()

    clean = IfThenCorpus.from_jsonl(args.clean_corpus)
    polluted = IfThenCorpus.from_jsonl(args.polluted_corpus)
    if len(clean.atoms) != len(polluted.atoms):
        raise RuntimeError(
            "clean / polluted corpora differ in length; rendering would be unaligned"
        )

    clean_text = render_paragraph(clean)
    poll_text = render_paragraph(polluted)
    inv = Inventory.load(args.inventory)
    mid = (inv.scale_min + inv.scale_max) / 2.0
    llm = get_client(args.backend, temperature=args.temperature)

    print(f"[p2-sanity] clean paragraph: {len(clean_text)} chars, {len(clean.atoms)} sentences", flush=True)
    clean_report = _administer(clean_text, inv=inv, llm=llm, workers=args.workers)
    print(f"[p2-sanity] polluted paragraph: {len(poll_text)} chars", flush=True)
    poll_report = _administer(poll_text, inv=inv, llm=llm, workers=args.workers)

    clean_score = clean_report.dimension_scores.get(args.targeted_dim)
    poll_score = poll_report.dimension_scores.get(args.targeted_dim)
    if clean_score is None or poll_score is None:
        raise RuntimeError("missing dimension scores after administration")
    clean_spread = abs(clean_score - mid)
    poll_spread = abs(poll_score - mid)
    spread_drop = clean_spread - poll_spread
    score_shift = abs(clean_score - poll_score)
    matched, total = _direction_match(
        clean_report.responses, poll_report.responses,
        dimension=args.targeted_dim, mid=mid,
    )
    direction_match = matched / total if total else float("nan")
    pass_shift = score_shift >= args.min_score_shift
    pass_direction = direction_match <= args.max_direction_match
    overall = pass_shift and pass_direction

    print()
    print(f"=== P² Sanity check on {args.targeted_dim} ===")
    print(f"  clean : {clean_report.dimension_scores}")
    print(f"  poll  : {poll_report.dimension_scores}")
    print(f"  score shift  : {score_shift:.3f}  (need >= {args.min_score_shift})  "
          f"{'PASS' if pass_shift else 'FAIL'}")
    print(f"  dir match    : {direction_match:.3f}  (need <= {args.max_direction_match})  "
          f"{'PASS' if pass_direction else 'FAIL'}")
    print(f"  OVERALL      : {'PASS' if overall else 'FAIL'}")

    out_para = Path(args.out_paragraphs)
    out_para.parent.mkdir(parents=True, exist_ok=True)
    out_para.write_text(json.dumps({
        "clean_corpus": args.clean_corpus,
        "polluted_corpus": args.polluted_corpus,
        "render": "When <situation>, <behavior>. ... (corpus order, period-joined)",
        "clean_paragraph": clean_text,
        "polluted_paragraph": poll_text,
        "n_sentences": len(clean.atoms),
        "atom_id_order": [a.id for a in clean.atoms],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    out_sanity = Path(args.out_sanity)
    out_sanity.parent.mkdir(parents=True, exist_ok=True)
    out_sanity.write_text(json.dumps({
        "audit_method": "p2",
        "clean_corpus": args.clean_corpus,
        "polluted_corpus": args.polluted_corpus,
        "inventory": args.inventory,
        "backend": args.backend,
        "temperature": args.temperature,
        "workers": args.workers,
        "targeted_dim": args.targeted_dim,
        "scale_mid": mid,
        "clean_dimension_scores": clean_report.dimension_scores,
        "poll_dimension_scores": poll_report.dimension_scores,
        "clean_spread": clean_spread,
        "poll_spread": poll_spread,
        "spread_drop": spread_drop,
        "score_shift": score_shift,
        "direction_match_n_matched": matched,
        "direction_match_n_total": total,
        "direction_match_rate": direction_match,
        "criteria": {
            "min_score_shift": args.min_score_shift,
            "max_direction_match": args.max_direction_match,
        },
        "pass_score_shift": pass_shift,
        "pass_direction": pass_direction,
        "pass_overall": overall,
        "clean_responses": [asdict(r) for r in clean_report.responses],
        "poll_responses": [asdict(r) for r in poll_report.responses],
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"[p2-sanity] wrote -> {out_sanity}")
    print(f"[p2-sanity] paragraphs -> {out_para}")


if __name__ == "__main__":
    main()
