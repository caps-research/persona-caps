#!/usr/bin/env python
"""Build agent-track if-then atoms from the IPIP-NEO-300 item bank.

This is the agent-experiments pipeline (CONSENSUS §3 'Agent-track 主线').
The output is **forbidden in paper main results** -- only the downstream
agent toolkit may load these atoms.

For each of the 300 IPIP-NEO items the LLM produces ONE concrete
if-then atom:

  * Item polarity '+' (item asserts the trait behaviour) -> atom in
    ``high_<domain>.jsonl`` (so loading high_neuroticism.jsonl gives a
    set of behaviours typical of a high-N person).
  * Item polarity '-' (item asserts the absence of the trait behaviour)
    -> atom in ``low_<domain>.jsonl``.

Every atom carries provenance back to the source item id, facet code,
facet name and inventory polarity, so the downstream agent can be
audited (e.g. "this agent's anxiety facet is built from items 1, 31, 61, ...").

Output dir: ``data/if_then/agent/ipip_neo_300/``
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from persona_caps.corpus import IfThenAtom, IfThenCorpus
from persona_caps.data_sources.ipip_neo import IpipItem, load_items
from persona_caps.llm import ChatMessage, get_client


OUT_DIR = Path("data/if_then/agent/ipip_neo_300")


VERBALISE_SYSTEM = """You convert one IPIP-NEO personality test item into a single, concrete first-person if-then behavioural contingency.

Strict rules:
1. Output JSON only, no prose around it.
2. Schema: {"situation": str, "behavior": str, "affect": str | null}
3. The atom must describe what someone who answers 'Very Accurate' to the test item would actually DO in a specific everyday situation. The situation must be concrete (a workplace, a conversation, a household event), NOT a generic 'when I'm anxious' or 'in general'.
4. Do NOT echo the test item verbatim or use the trait name. Translate the disposition into observable behaviour.
5. The atom must be unambiguous about which pole of the facet it expresses (the pole given as TARGET_POLE below).
6. Each field <= 35 words.
7. Vary situations across calls -- don't reuse the same context (e.g. don't always pick 'work meeting')."""


VERBALISE_USER_TEMPLATE = """IPIP item: "{item_text}"
Domain: {domain} (Big Five)
Facet: {facet_name} (code {facet_code})
Item polarity in inventory: {item_polarity}   # '+' = item asserts the high-pole behaviour, '-' = item asserts the low-pole behaviour
TARGET_POLE: {target_pole}                    # write an atom that someone scoring HIGH on this facet (if TARGET_POLE='high') or LOW (if 'low') would naturally show

Write the atom JSON now."""


def _verbalise_one(client, item: IpipItem) -> IfThenAtom | None:
    target_pole = "high" if item.polarity == "+" else "low"
    user = VERBALISE_USER_TEMPLATE.format(
        item_text=item.text,
        domain=item.domain,
        facet_name=item.facet_name,
        facet_code=item.facet_code,
        item_polarity=item.polarity,
        target_pole=target_pole,
    )
    raw = client.chat(
        [ChatMessage(role="user", content=user)],
        system=VERBALISE_SYSTEM,
    ).strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        i, j = raw.find("{"), raw.rfind("}")
        if i < 0 or j < 0:
            print(f"[verbalise] FAIL parse for item id={item.item_id}: {raw[:120]!r}", file=sys.stderr)
            return None
        try:
            obj = json.loads(raw[i : j + 1])
        except json.JSONDecodeError:
            print(f"[verbalise] FAIL parse for item id={item.item_id}", file=sys.stderr)
            return None
    if not isinstance(obj, dict):
        return None
    sit, beh = obj.get("situation"), obj.get("behavior")
    if not sit or not beh:
        return None
    return IfThenAtom(
        id=f"ipip_neo300_{item.facet_code.lower()}_{('plus' if item.polarity == '+' else 'minus')}_{item.item_id:03d}",
        situation=str(sit).strip(),
        behavior=str(beh).strip(),
        affect=(str(obj["affect"]).strip() if obj.get("affect") else None),
        target_dimension=item.domain,
        polarity=target_pole,
        provenance=item.to_provenance(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--llm", default="minimax")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only verbalise the first N items (smoke test).")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    items = load_items()
    if args.limit:
        items = items[: args.limit]
    print(f"[stage1] Loaded {len(items)} IPIP items", file=sys.stderr)

    client = get_client(args.llm)
    grouped: dict[tuple[str, str], list[IfThenAtom]] = {}
    print(f"[stage2] Verbalising {len(items)} items via {args.llm} (workers={args.workers})", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_verbalise_one, client, it): it for it in items}
        done = 0
        for fut in as_completed(futs):
            it = futs[fut]
            atom = fut.result()
            if atom is not None:
                target_pole = "high" if it.polarity == "+" else "low"
                grouped.setdefault((it.domain, target_pole), []).append(atom)
            done += 1
            if done % 25 == 0:
                print(f"[stage2]   {done}/{len(items)} done", file=sys.stderr)

    counts: dict[str, int] = {}
    for (dim, pol), atoms in grouped.items():
        atoms.sort(key=lambda a: int(a.provenance["item_id"]))
        c = IfThenCorpus(atoms)
        path = args.out_dir / f"{pol}_{dim}.jsonl"
        c.to_jsonl(path)
        counts[f"{pol}_{dim}"] = len(atoms)
        print(f"[stage2] wrote {len(atoms):3d} atoms -> {path}", file=sys.stderr)

    manifest = {
        "source": "IPIP-NEO-300",
        "track": "agent",
        "license": "public_domain (https://ipip.ori.org/newPermission.htm)",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "params": {"llm": args.llm, "limit": args.limit},
        "n_atoms_by_cell": counts,
        "consensus_doc": "CONSENSUS.md v3 (agent-track; forbidden in paper main results)",
        "warning": "Agent-track corpus. Do NOT load in scripts that produce paper main results.",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
