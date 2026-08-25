"""Sanity check for the polarity-flip injection.

A polluted corpus is only useful for the §4.7 audit-loop experiment if
the polarity flips actually *break* the agent's measured personality on
that dimension — otherwise we are auditing nothing. This script runs
Mini-IPIP on both the clean and polluted corpora and reports:

* per-dimension mean ratings on each
* drop in the targeted dimension's spread (|target_dim_score - 3|)
* item-level direction-match rate vs the clean run on the targeted dim
  (an item is "direction-matched" if both runs land on the same side
  of the scale midpoint; we expect this to fall sharply on the
  polluted run if the flips actually flipped the agent's behavior)

Acceptance criteria (per agreed §4.7 protocol, refined to handle
"full-flip" cases that the original spread-only formulation missed):

* targeted-dim absolute score movement |clean - polluted| on the
  targeted dim is >= ``--min-score-shift`` (default 1.0 Likert points)
  AND
* targeted-dim direction-match rate <= ``--max-direction-match``
  (default 0.70).

Item is "direction-matched" iff
``(clean_rating - mid) * (polluted_rating - mid) >= 0``, evaluated on
the *raw* rating (pre-keyed-reversal) so the test measures the agent's
literal response, not the inventory's re-scoring.

Spread (|score - mid|) is also reported for diagnostics — a spread
*drop* indicates "personality became less polarized"; a spread that
stays large with direction-match=0 indicates "personality flipped to
the opposite pole". Both qualify as PASS under |Δscore| >= 1.0.

Exit codes:
* 0 = both criteria pass (sanity OK, the polluted corpus is usable)
* 2 = at least one criterion fails (flip prompt or K is too weak)

Example
-------
::

    python scripts/audit_loop/sanity_check.py \
        --clean-corpus    data/if_then/paper/beck_2022_v2/high_neuroticism.jsonl \
        --polluted-corpus runs/audit_loop/polluted/high_neuroticism__k10_s1.jsonl \
        --inventory       data/inventories/ipip_short.json \
        --backend         minimax \
        --targeted-dim    neuroticism \
        --workers         8 \
        --out             runs/audit_loop/sanity/high_neuroticism__k10_s1.json
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


def _direction_match(
    clean_responses, polluted_responses, *, dimension: str, mid: float
) -> tuple[int, int]:
    """Return (n_matched, n_total) on the targeted dimension."""
    matched = 0
    total = 0
    clean_by_id = {r.item.id: r for r in clean_responses if r.item.dimension == dimension}
    poll_by_id = {r.item.id: r for r in polluted_responses if r.item.dimension == dimension}
    for iid, cr in clean_by_id.items():
        pr = poll_by_id.get(iid)
        if pr is None or cr.rating is None or pr.rating is None:
            continue
        total += 1
        # Direction match: both on the same side of the midpoint.
        if (cr.rating - mid) * (pr.rating - mid) >= 0:
            matched += 1
    return matched, total


def _run(args, corpus_path: str, *, label: str):
    print(f"[sanity:{label}] loading {corpus_path}", flush=True)
    corpus = IfThenCorpus.from_jsonl(corpus_path)
    llm = get_client(args.backend, temperature=args.temperature)
    agent = CapsAgent(corpus=corpus, llm=llm, top_k=args.top_k)
    inventory = Inventory.load(args.inventory)
    print(
        f"[sanity:{label}] administering {inventory.name} "
        f"({len(inventory.items)} items, workers={args.workers})",
        flush=True,
    )
    return inventory, administer(agent, inventory, workers=args.workers)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-corpus", required=True)
    p.add_argument("--polluted-corpus", required=True)
    p.add_argument("--inventory", required=True)
    p.add_argument("--backend", default="minimax")
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--targeted-dim",
        required=True,
        help="dimension whose corpus was polluted (e.g. neuroticism, extraversion)",
    )
    p.add_argument(
        "--min-score-shift",
        type=float,
        default=1.0,
        help="minimum |clean - polluted| on the targeted dim's mean score",
    )
    p.add_argument(
        "--max-direction-match",
        type=float,
        default=0.70,
        help="maximum direction-match rate (lower = more items flipped)",
    )
    p.add_argument("--out", default=None, help="JSON path for full report")
    args = p.parse_args()

    inv, clean_report = _run(args, args.clean_corpus, label="clean")
    _, poll_report = _run(args, args.polluted_corpus, label="polluted")

    mid = (inv.scale_min + inv.scale_max) / 2.0
    clean_score = clean_report.dimension_scores.get(args.targeted_dim)
    poll_score = poll_report.dimension_scores.get(args.targeted_dim)
    if clean_score is None or poll_score is None:
        raise RuntimeError(
            f"missing dimension score for {args.targeted_dim}; "
            f"clean={clean_report.dimension_scores}, poll={poll_report.dimension_scores}"
        )
    clean_spread = abs(clean_score - mid)
    poll_spread = abs(poll_score - mid)
    spread_drop = clean_spread - poll_spread
    score_shift = abs(clean_score - poll_score)

    matched, total = _direction_match(
        clean_report.responses,
        poll_report.responses,
        dimension=args.targeted_dim,
        mid=mid,
    )
    direction_match = matched / total if total else float("nan")

    pass_shift = score_shift >= args.min_score_shift
    pass_direction = direction_match <= args.max_direction_match
    overall = pass_shift and pass_direction

    print()
    print(f"=== Sanity check on {args.targeted_dim} ===")
    print(f"  clean  scores: {clean_report.dimension_scores}")
    print(f"  poll   scores: {poll_report.dimension_scores}")
    print(f"  clean  spread (|score-mid|): {clean_spread:.3f}  [diagnostic]")
    print(f"  poll   spread (|score-mid|): {poll_spread:.3f}  [diagnostic]")
    print(f"  spread drop                : {spread_drop:+.3f}  [diagnostic only]")
    print(f"  absolute score shift       : {score_shift:.3f}  (need >= {args.min_score_shift})  "
          f"{'PASS' if pass_shift else 'FAIL'}")
    print(f"  direction match rate       : {direction_match:.3f}  (need <= {args.max_direction_match})  "
          f"{'PASS' if pass_direction else 'FAIL'}")
    print(f"  OVERALL                    : {'PASS' if overall else 'FAIL'}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "clean_corpus": args.clean_corpus,
            "polluted_corpus": args.polluted_corpus,
            "inventory": args.inventory,
            "backend": args.backend,
            "top_k": args.top_k,
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
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"  wrote -> {out}")

    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
