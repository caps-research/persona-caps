"""P² baseline audit workflows: oracle / blind-rewrite / response-detect.

Given a polluted P² paragraph and ground-truth flip log (the same one
the CAPS scripts consume), this script runs ONE of the three P² audit
strategies enumerated in §4.7.3 and reports its localization +
edit-footprint + fix-success + off-target metrics in the same shape as
``apply_fix_and_measure.py`` so the two methods can be aggregated by
the same downstream code.

Strategies
----------

* ``oracle`` (P²-A) -- the auditor is *handed* the K ground-truth
  flipped sentence indices and reverts exactly those sentences to
  their clean text. No LLM calls. Localization is trivially perfect
  by construction; this measures the best-case fix and the minimal
  edit footprint a paragraph-level fix can achieve.

* ``blind`` (P²-B) -- the auditor sees only "the persona is mis-
  behaving on dimension D, target polarity P". They re-generate the
  entire paragraph from a trait-target prompt; the original paragraph
  is discarded. One LLM call (default GPT-4o, T=0.0). The auditor has
  no atom-level access; this measures the realistic floor when the
  persona was originally written as a single paragraph and the
  operator just wants to "fix the personality" without surgical
  edits.

* ``detect`` (P²-D) -- the auditor sees the polluted paragraph AND
  the signal that the persona is mis-behaving on dimension D. They
  ask an LLM (GPT-4o, T=0.0) to flag up to K_target sentences whose
  behavior content contradicts the intended polarity. The flagged
  sentences are then replaced with the clean baseline text (we treat
  the LLM's flag as a localizer, not a rewriter, so that the only
  thing being measured is detection quality — not LLM rewriting
  quality, which would confound the comparison with ``oracle``).

The fix step in all three strategies produces a "fixed paragraph",
which is fed to a fresh ``PersonaDescriptionAgent`` and re-measured
on Mini-IPIP. Fix-success and off-target gap are computed against
the P² clean baseline already stored in the P² sanity JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from persona_caps.agent import PersonaDescriptionAgent  # noqa: E402
from persona_caps.corpus import IfThenCorpus  # noqa: E402
from persona_caps.eval import Inventory, administer  # noqa: E402
from persona_caps.llm import ChatMessage, get_client  # noqa: E402

# Local helper from p2_inject_and_sanity.py (re-implemented here to
# avoid sibling import gymnastics for top-level scripts).
def render_paragraph(corpus: IfThenCorpus) -> str:
    sentences: list[str] = []
    for atom in corpus.atoms:
        sit = atom.situation.strip().rstrip(".")
        beh = atom.behavior.strip().rstrip(".")
        sentences.append(f"When {sit[0].lower() + sit[1:]}, {beh[0].lower() + beh[1:]}.")
    return " ".join(sentences)


def _render_single(atom) -> str:
    sit = atom.situation.strip().rstrip(".")
    beh = atom.behavior.strip().rstrip(".")
    return f"When {sit[0].lower() + sit[1:]}, {beh[0].lower() + beh[1:]}."


# ----------------------------- ORACLE (A) -----------------------------

def _apply_oracle(clean: IfThenCorpus, polluted: IfThenCorpus, flipped_ids: set[str]) -> tuple[str, list[str], dict]:
    """Revert each flipped atom to its clean text; render."""
    clean_by_id = {a.id: a for a in clean.atoms}
    fixed_atoms = []
    n_restored = 0
    chars_replaced = 0
    for a in polluted.atoms:
        if a.id in flipped_ids and a.id in clean_by_id:
            ca = clean_by_id[a.id]
            if ca.behavior != a.behavior:
                n_restored += 1
                chars_replaced += len(a.behavior)
            fixed_atoms.append(a.model_copy(update={"behavior": ca.behavior}))
        else:
            fixed_atoms.append(a)
    fixed_corpus = IfThenCorpus(fixed_atoms)
    return (
        render_paragraph(fixed_corpus),
        sorted(flipped_ids),
        {"atoms_restored": n_restored, "chars_replaced": chars_replaced},
    )


# ------------------------- BLIND REWRITE (B) -------------------------

BLIND_SYSTEM = """You are writing the first-person system prompt for an \
agent that should exhibit {polarity} {dimension} on the Big Five. \
Produce ONE paragraph composed of 5-10 short "When <situation>, I <behavior>" \
contingency sentences, in the agent's own voice, describing how this \
agent reacts across everyday situations to clearly express {polarity} \
{dimension}. Match this exact sentence template; do NOT meta-comment about \
personality or use third person. Output the paragraph and nothing else."""

def _apply_blind(targeted_dim: str, polarity: str, *, llm) -> tuple[str, list[str], dict]:
    """Regenerate the persona paragraph from a trait prompt; everything else discarded."""
    sys_prompt = BLIND_SYSTEM.format(polarity=polarity, dimension=targeted_dim)
    text = llm.chat(
        messages=[ChatMessage(role="user", content="Write the personality description now.")],
        system=sys_prompt,
    )
    text = text.strip()
    return (
        text,
        [],
        {"atoms_restored": 0, "chars_replaced": -1, "blind_regen_chars": len(text)},
    )


# ------------------------- RESPONSE-DETECT (D) -------------------------

DETECT_SYSTEM = """You are reviewing a Big-Five personality description \
that should express {polarity} {dimension}, but the persona-driven agent \
is now exhibiting the OPPOSITE polarity on that dimension. Each sentence \
in the paragraph is one behavioral contingency of the form "When X, Y." \
Identify which sentences (by index, 0-based) most likely contradict the \
intended {polarity}-{dimension} polarity. Flag at most {k_target} \
sentences. Output STRICT JSON only: \
{{"flagged_indices": [<int>, ...], "rationale": "<one short clause>"}}. \
No prose outside the JSON, no markdown fences."""

DETECT_USER_TEMPLATE = """Intended dimension: {dimension}
Intended polarity: {polarity}

The paragraph (sentence indices in brackets):
{indexed_paragraph}

Flag the sentence indices whose content contradicts {polarity}-{dimension}.
"""


def _apply_detect(
    clean: IfThenCorpus,
    polluted: IfThenCorpus,
    targeted_dim: str,
    polarity: str,
    *,
    k_target: int,
    llm,
    max_attempts: int = 3,
) -> tuple[str, list[str], dict]:
    indexed_lines = []
    for i, atom in enumerate(polluted.atoms):
        indexed_lines.append(f"[{i}] {_render_single(atom)}")
    indexed_paragraph = "\n".join(indexed_lines)
    sys_prompt = DETECT_SYSTEM.format(
        polarity=polarity, dimension=targeted_dim, k_target=k_target,
    )
    user_msg = DETECT_USER_TEMPLATE.format(
        dimension=targeted_dim, polarity=polarity, indexed_paragraph=indexed_paragraph,
    )
    flagged_indices: list[int] = []
    raw_attempts = []
    for _ in range(max_attempts):
        raw = llm.chat(
            messages=[ChatMessage(role="user", content=user_msg)],
            system=sys_prompt,
        )
        raw_attempts.append(raw)
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].lstrip()
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            continue
        idxs = obj.get("flagged_indices")
        if isinstance(idxs, list) and all(isinstance(i, int) for i in idxs):
            flagged_indices = sorted({int(i) for i in idxs if 0 <= i < len(polluted.atoms)})
            break

    clean_by_id = {a.id: a for a in clean.atoms}
    fixed_atoms = list(polluted.atoms)
    flagged_ids = []
    chars_replaced = 0
    n_restored = 0
    for i in flagged_indices:
        atom = polluted.atoms[i]
        flagged_ids.append(atom.id)
        clean_atom = clean_by_id.get(atom.id)
        if clean_atom is not None and clean_atom.behavior != atom.behavior:
            n_restored += 1
            chars_replaced += len(atom.behavior)
            fixed_atoms[i] = atom.model_copy(update={"behavior": clean_atom.behavior})
    fixed_corpus = IfThenCorpus(fixed_atoms)
    return (
        render_paragraph(fixed_corpus),
        flagged_ids,
        {
            "atoms_restored": n_restored,
            "chars_replaced": chars_replaced,
            "n_flagged": len(flagged_ids),
            "raw_detect_attempts": len(raw_attempts),
        },
    )


# ----------------------------- DRIVER -----------------------------

def _direction_match_vs_clean(fixed_resp, clean_resp, *, dim: str, mid: float):
    matched = 0
    total = 0
    clean_by_id = {r["item"]["id"]: r for r in clean_resp if r["item"]["dimension"] == dim}
    for r in fixed_resp:
        if r.item.dimension != dim:
            continue
        c = clean_by_id.get(r.item.id)
        if c is None or r.rating is None or c.get("rating") is None:
            continue
        total += 1
        if (r.rating - mid) * (c["rating"] - mid) >= 0:
            matched += 1
    return matched, total


def _precision_recall(flagged: set[str], gt: set[str]) -> dict:
    tp = len(flagged & gt)
    fp = len(flagged - gt)
    fn = len(gt - flagged)
    p = tp / max(len(flagged), 1) if flagged else 0.0
    r = tp / max(len(gt), 1)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", choices=["oracle", "blind", "detect"], required=True)
    p.add_argument("--clean-corpus", required=True)
    p.add_argument("--polluted-corpus", required=True)
    p.add_argument("--injection-log", required=True)
    p.add_argument("--p2-sanity", required=True,
                   help="p2_inject_and_sanity.py output for the clean baseline scores")
    p.add_argument("--inventory", required=True)
    p.add_argument("--targeted-dim", required=True)
    p.add_argument("--polarity", required=True, choices=["high", "low"],
                   help="intended polarity of the targeted dim (for blind / detect)")
    p.add_argument("--backend", default="minimax", help="subject LLM (for fixed paragraph)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--auditor", default="gpt-4o", help="LLM used for blind / detect")
    p.add_argument("--max-gap", type=float, default=0.3)
    p.add_argument("--min-dir-match", type=float, default=0.85)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    clean = IfThenCorpus.from_jsonl(args.clean_corpus)
    polluted = IfThenCorpus.from_jsonl(args.polluted_corpus)
    inj = json.loads(Path(args.injection_log).read_text(encoding="utf-8"))
    p2_sanity = json.loads(Path(args.p2_sanity).read_text(encoding="utf-8"))
    gt_set = set(inj["flipped_ids"])
    k_oracle = int(inj["k"])

    if args.strategy in ("blind", "detect"):
        auditor_llm = get_client(args.auditor, temperature=0.0)
    else:
        auditor_llm = None

    if args.strategy == "oracle":
        fixed_text, flagged_ids, footprint = _apply_oracle(clean, polluted, gt_set)
    elif args.strategy == "blind":
        fixed_text, flagged_ids, footprint = _apply_blind(
            args.targeted_dim, args.polarity, llm=auditor_llm,
        )
    else:
        fixed_text, flagged_ids, footprint = _apply_detect(
            clean, polluted, args.targeted_dim, args.polarity,
            k_target=k_oracle, llm=auditor_llm,
        )

    inv = Inventory.load(args.inventory)
    mid = (inv.scale_min + inv.scale_max) / 2.0
    subject_llm = get_client(args.backend, temperature=args.temperature)
    print(f"[p2-fix:{args.strategy}] administering fixed paragraph", flush=True)
    agent = PersonaDescriptionAgent(fixed_text, llm=subject_llm)
    fixed_report = administer(agent, inv, workers=args.workers)

    clean_scores = p2_sanity["clean_dimension_scores"]
    fixed_scores = fixed_report.dimension_scores
    targeted = args.targeted_dim
    target_gap = abs(fixed_scores.get(targeted, 0.0) - clean_scores.get(targeted, 0.0))
    other_dims = [d for d in clean_scores if d != targeted]
    other_gaps = {d: abs(fixed_scores.get(d, 0.0) - clean_scores.get(d, 0.0)) for d in other_dims}
    mean_other_gap = sum(other_gaps.values()) / max(len(other_dims), 1)

    matched, total = _direction_match_vs_clean(
        fixed_report.responses, p2_sanity["clean_responses"], dim=targeted, mid=mid,
    )
    dir_match = matched / total if total else float("nan")
    fix_success = (target_gap <= args.max_gap) and (dir_match >= args.min_dir_match)

    # localization metrics (oracle: perfect by construction; blind: undefined)
    if args.strategy == "blind":
        loc_metrics = {"note": "blind rewrite has no localization step"}
    else:
        loc_metrics = _precision_recall(set(flagged_ids), gt_set)

    payload = {
        "fix_mode": f"p2_{args.strategy}",
        "clean_corpus": args.clean_corpus,
        "polluted_corpus": args.polluted_corpus,
        "injection_log": args.injection_log,
        "p2_sanity": args.p2_sanity,
        "targeted_dim": targeted,
        "polarity": args.polarity,
        "k_oracle": k_oracle,
        "flagged_ids": flagged_ids,
        "footprint": footprint,
        "localization_metrics": loc_metrics,
        "clean_dimension_scores": clean_scores,
        "fixed_dimension_scores": fixed_scores,
        "target_gap": target_gap,
        "direction_match_vs_clean": dir_match,
        "off_target_gap_per_dim": other_gaps,
        "off_target_gap_mean": mean_other_gap,
        "criteria": {"max_gap": args.max_gap, "min_dir_match": args.min_dir_match},
        "fix_success": fix_success,
        "fixed_paragraph": fixed_text,
        "fixed_responses": [asdict(r) for r in fixed_report.responses],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(
        f"[p2-fix:{args.strategy}] wrote {out}  target_gap={target_gap:.2f} "
        f"dir_match={dir_match:.2f} off_target={mean_other_gap:.2f} success={fix_success} "
        f"loc={loc_metrics}"
    )


if __name__ == "__main__":
    main()
