"""Locator (a): incident-driven per-response attribution.

Given a sanity-check run that has stored ``clean_responses`` and
``poll_responses`` (each carrying the fired-atom log for the
inventory items), find the atoms most likely responsible for the
personality drift on the targeted dimension.

The intuition: the fired-atom log is what CAPS gives you for free as
its "governance interface". When the persona shifts, the operator does
not have to read the whole corpus — they can read only the atoms that
were *in the room* when the personality misbehaved on a particular
inventory item, weighted by how badly that item disagrees with the
clean baseline.

Algorithm:

1.  ``disagreement(item) = |clean_rating - polluted_rating|`` on the
    *targeted dimension only*. Items on other dimensions are ignored
    because, by construction, the contamination should only show on
    the polluted cell.
2.  For every atom that fires on any item, accumulate
    ``score(atom) = sum_{items where atom fired} disagreement(item) * sim(atom, item)``.
    ``sim`` is the cosine similarity already recorded in the fired-atom
    tuple (it is identical for clean and polluted because situation
    fields are unchanged by the pollution).
3.  Rank atoms by descending score and emit the top-K as the flag set.

We deliberately do *not* re-query the LLM here. Locator (a) is the
"cheap" workflow whose entire input is artefacts CAPS already
produces in normal operation — that is the comparison we want to make
against P²-D (which has no fired-atom log and must therefore re-prompt
an LLM to detect suspects).

Outputs a JSON with:

* ``flag_set`` -- ranked list ``[{atom_id, score}, ...]`` truncated to
  ``--top-k``
* ``ground_truth`` -- the K injected atom IDs (from the injection log)
* ``metrics`` -- precision@K, recall@K, F1@K, top-1 hit, P@5/P@10

Example
-------
::

    python scripts/audit_loop/locate_via_responses.py \
        --sanity runs/audit_loop/sanity/high_neuroticism__k15_s1_targeted.json \
        --injection-log runs/audit_loop/polluted/high_neuroticism__k15_s1_targeted.log.json \
        --targeted-dim neuroticism \
        --top-k 15 \
        --out runs/audit_loop/locate_a/high_neuroticism__k15_s1_targeted.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def _load_sanity(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_injection(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _accumulate_scores(
    poll_responses: list[dict],
    clean_responses: list[dict],
    *,
    targeted_dim: str,
) -> tuple[dict[str, float], dict[str, int], dict[str, int]]:
    """Return (atom_score, atom_appearance_count, n_targeted_items_total)."""
    clean_by_iid = {r["item"]["id"]: r for r in clean_responses}
    atom_scores: dict[str, float] = defaultdict(float)
    atom_appearances: dict[str, int] = defaultdict(int)
    n_targeted = 0
    for p in poll_responses:
        item = p["item"]
        if item["dimension"] != targeted_dim:
            continue
        n_targeted += 1
        c = clean_by_iid.get(item["id"])
        # If we can't find a paired clean response or either rating is None,
        # we skip the item — it contributes no attribution signal.
        if c is None or p.get("rating") is None or c.get("rating") is None:
            continue
        disagreement = abs(p["rating"] - c["rating"])
        for atom_id, sim in p.get("fired", []):
            atom_appearances[atom_id] += 1
            if disagreement > 0:
                # weight by both disagreement magnitude and the atom's
                # situation-relevance to *this* particular item
                atom_scores[atom_id] += disagreement * float(sim)
    return atom_scores, atom_appearances, n_targeted


def _precision_recall(flag_set: list[str], ground_truth: set[str], *, k: int) -> dict:
    flagged_at_k = flag_set[:k]
    tp = sum(1 for a in flagged_at_k if a in ground_truth)
    fp = len(flagged_at_k) - tp
    fn = len(ground_truth) - tp
    p = tp / max(len(flagged_at_k), 1)
    r = tp / max(len(ground_truth), 1)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"k": k, "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sanity", required=True, help="sanity_check.py output JSON")
    p.add_argument("--injection-log", required=True, help="inject_polarity_flips.py log JSON")
    p.add_argument("--targeted-dim", required=True)
    p.add_argument("--top-k", type=int, default=None,
                   help="size of the flag set (default = K injected, i.e. the oracle K)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    sanity = _load_sanity(Path(args.sanity))
    inj = _load_injection(Path(args.injection_log))
    ground_truth: set[str] = set(inj["flipped_ids"])
    k_oracle = int(inj["k"])

    atom_scores, atom_appearances, n_targeted = _accumulate_scores(
        sanity["poll_responses"],
        sanity["clean_responses"],
        targeted_dim=args.targeted_dim,
    )

    ranked = sorted(atom_scores.items(), key=lambda kv: (-kv[1], kv[0]))
    flag_set_full = [aid for aid, _ in ranked]
    top_k = args.top_k if args.top_k is not None else k_oracle
    flag_set = flag_set_full[:top_k]

    pr_at_k = _precision_recall(flag_set_full, ground_truth, k=top_k)
    pr_at_5 = _precision_recall(flag_set_full, ground_truth, k=5)
    pr_at_10 = _precision_recall(flag_set_full, ground_truth, k=10)
    top1_hit = 1 if flag_set_full and flag_set_full[0] in ground_truth else 0

    payload = {
        "locator": "responses",
        "sanity_input": args.sanity,
        "injection_log": args.injection_log,
        "targeted_dim": args.targeted_dim,
        "k_oracle": k_oracle,
        "top_k_used": top_k,
        "n_atoms_with_signal": len(ranked),
        "n_atoms_with_any_fire": len(atom_appearances),
        "n_targeted_inventory_items": n_targeted,
        "flag_set_top_k": [
            {"atom_id": aid, "score": atom_scores[aid], "in_ground_truth": aid in ground_truth}
            for aid in flag_set
        ],
        "flag_set_full_ranked": [
            {"atom_id": aid, "score": s, "in_ground_truth": aid in ground_truth}
            for aid, s in ranked
        ],
        "ground_truth": sorted(ground_truth),
        "metrics_at_oracle_k": pr_at_k,
        "metrics_at_5": pr_at_5,
        "metrics_at_10": pr_at_10,
        "top1_hit": top1_hit,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[locate-a] wrote {out}")
    print(
        f"[locate-a] dim={args.targeted_dim} oracle_K={k_oracle} "
        f"P@K={pr_at_k['precision']:.2f} R@K={pr_at_k['recall']:.2f} "
        f"F1@K={pr_at_k['f1']:.2f} top1_hit={top1_hit}"
    )


if __name__ == "__main__":
    main()
