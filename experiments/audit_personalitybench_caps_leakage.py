#!/usr/bin/env python3
"""PersonalityBench dual-eval helper: CAPS/SP generate + CAPS prompt leak audit.

CAPS path must show zero hard trait/π leakage in reconstructed system prompts.
SP path is an intentional-leak baseline (reported separately).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from persona_caps.agent import CAPS_SYSTEM_PROMPT  # noqa: E402
from persona_caps.corpus import load_corpus_paper_only  # noqa: E402

TRAIT_DIR = {
    "Extraversion": "extraversion",
    "Openness": "openness",
    "Conscientiousness": "conscientiousness",
    "Agreeableness": "agreeableness",
    "Neuroticism": "neuroticism",
}

HARD_PATTERNS = [
    ("explicit_trait_phrase", re.compile(r"You tend toward\s+(low|moderate|high)\s+\w+", re.I)),
    ("low_high_trait", re.compile(
        r"\b(low|high|moderate)\s+(extraversion|agreeableness|openness|"
        r"conscientiousness|neuroticism)\b",
        re.I,
    )),
    ("pi_assignment", re.compile(r"\bpi(?:_vector)?\s*[:=]\s*|\bπ\s*[:=]", re.I)),
    ("engagement_dial", re.compile(r"Social engagement intensity", re.I)),
    ("imagine_trait_adj", re.compile(
        r"Imagine you are (an?|a)\s+"
        r"(extraverted|introverted|agreeable|disagreeable|open|closed|"
        r"conscientious|unconscientious|neurotic|calm)\b",
        re.I,
    )),
    ("speech_control_block", re.compile(r"SPEECH CONTROL\s*\(", re.I)),
]


def reconstruct_caps_prompt(
    *,
    fired_atom_ids: list[str],
    corpus_dir: Path,
    trait: str,
    pol: str,
) -> str:
    pole = corpus_dir / f"{pol}_{TRAIT_DIR[trait]}.jsonl"
    corpus = load_corpus_paper_only(pole)
    by_id = {a.id: a for a in corpus.atoms}
    lines = []
    for aid in fired_atom_ids:
        atom = by_id.get(aid)
        if atom is None:
            lines.append(f"- <MISSING {aid}>")
        else:
            lines.append(f"- {atom.render()}")
    return CAPS_SYSTEM_PROMPT.format(contingencies="\n".join(lines))


def audit_caps_dir(gen_dir: Path, corpus_dir: Path) -> dict:
    hits: list[dict] = []
    n = 0
    for path in sorted(gen_dir.glob("*_*.jsonl")):
        # Extraversion_high.jsonl
        stem = path.stem
        if "_" not in stem:
            continue
        trait, pol = stem.rsplit("_", 1)
        if trait not in TRAIT_DIR or pol not in {"high", "low"}:
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            n += 1
            ids = row.get("fired_atom_ids") or []
            prompt = reconstruct_caps_prompt(
                fired_atom_ids=ids,
                corpus_dir=corpus_dir,
                trait=trait,
                pol=pol,
            )
            blob = prompt + "\n" + str(row.get("question") or "")
            for name, pat in HARD_PATTERNS:
                m = pat.search(blob)
                if m:
                    hits.append(
                        {
                            "file": path.name,
                            "index": i,
                            "pattern": name,
                            "match": m.group(0),
                        }
                    )
                    break
    return {
        "path": str(gen_dir),
        "n_rows": n,
        "n_hits": len(hits),
        "status": "PASS" if len(hits) == 0 else "FAIL",
        "hits_sample": hits[:20],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--caps-dir", type=Path, required=True)
    p.add_argument(
        "--corpus-dir",
        type=Path,
        default=ROOT / "data/if_then/paper/beck_2022_v2",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    report = audit_caps_dir(args.caps_dir, args.corpus_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
