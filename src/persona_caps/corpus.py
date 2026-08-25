"""If-then contingency corpus — the symbolic representation of personality.

An ``IfThenAtom`` is one (situation, behavior) contingency, optionally
extended with the ``encoding`` / ``affect`` / ``goal`` slots from the full
Cognitive-Affective Unit (CAU) of Mischel & Shoda (1995). The minimal
required fields are ``situation`` and ``behavior``; the rest are reserved
for the extended CAU experiment in Phase 6 of the plan.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, Field


class IfThenAtom(BaseModel):
    id: str
    situation: str = Field(..., description="The triggering situation (the 'if').")
    behavior: str = Field(..., description="The dispositional response (the 'then').")
    # Optional extensions toward full CAUs:
    encoding: str | None = Field(default=None, description="How the situation is construed.")
    affect: str | None = Field(default=None, description="Affective response.")
    goal: str | None = Field(default=None, description="Activated goal.")
    # Bookkeeping
    target_dimension: str | None = Field(
        default=None,
        description="Big Five / HEXACO dimension this atom is intended to express.",
    )
    polarity: str | None = Field(
        default=None,
        description="'high' or 'low' on the target dimension, if any.",
    )
    provenance: dict | None = Field(
        default=None,
        description=(
            "Where this atom came from: dataset id, version, the empirical "
            "evidence (e.g. effect_size, n_obs), and any pipeline params. "
            "Mandatory for paper-track atoms; None allowed for hand-authored seeds."
        ),
    )

    def render(self, *, strip_affect: bool = False) -> str:
        """Render the atom as a single human-readable sentence used in the prompt.

        When ``strip_affect=True``, the optional ``(felt: ...)`` and
        ``(goal: ...)`` parentheticals are dropped. This is the "metadata-free"
        rendering used in the L8B ablation (paper §A.4.1): early-period Llama-3-8B
        sometimes regurgitated these parentheticals verbatim into its replies,
        leaking persona structure to the judge. Stripping them isolates the
        situation→behavior contingency from the optional CAU annotations.
        """
        parts = [f"WHEN {self.situation.strip()} → {self.behavior.strip()}"]
        if not strip_affect:
            if self.affect:
                parts.append(f"(felt: {self.affect.strip()})")
            if self.goal:
                parts.append(f"(goal: {self.goal.strip()})")
        return " ".join(parts)


class IfThenCorpus:
    """An ordered, JSONL-backed collection of if-then atoms."""

    def __init__(self, atoms: list[IfThenAtom] | None = None) -> None:
        self.atoms: list[IfThenAtom] = list(atoms or [])

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "IfThenCorpus":
        path = Path(path)
        atoms: list[IfThenAtom] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                atoms.append(IfThenAtom.model_validate_json(line))
        return cls(atoms)

    def to_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for atom in self.atoms:
                f.write(atom.model_dump_json(exclude_none=True) + "\n")

    def add(self, atom: IfThenAtom) -> None:
        self.atoms.append(atom)

    def remove(self, atom_id: str) -> bool:
        before = len(self.atoms)
        self.atoms = [a for a in self.atoms if a.id != atom_id]
        return len(self.atoms) < before

    def __len__(self) -> int:
        return len(self.atoms)

    def __iter__(self) -> Iterator[IfThenAtom]:
        return iter(self.atoms)

    def situations(self) -> list[str]:
        return [a.situation for a in self.atoms]

    @classmethod
    def merge(cls, *corpora: "IfThenCorpus") -> "IfThenCorpus":
        seen: set[str] = set()
        merged: list[IfThenAtom] = []
        for c in corpora:
            for a in c.atoms:
                if a.id in seen:
                    continue
                seen.add(a.id)
                merged.append(a)
        return cls(merged)


def load_corpus(path: str | Path) -> IfThenCorpus:
    return IfThenCorpus.from_jsonl(path)


def load_corpus_paper_only(path: str | Path) -> IfThenCorpus:
    """Like :func:`load_corpus` but refuses any path under
    ``data/if_then/agent/``. Use this from scripts producing
    paper main results to enforce the dual-track isolation rule
    (CONSENSUS §2 D4 / dual-track section).
    """
    p = Path(path).resolve()
    parts = p.parts
    if "agent" in parts:
        idx = parts.index("agent")
        if idx >= 1 and parts[idx - 1] == "if_then":
            raise PermissionError(
                f"Refusing to load agent-track corpus into paper-track pipeline: {p}. "
                "Per CONSENSUS.md §2, paper main results may not use IPIP-derived "
                "or otherwise circular corpora. If this load is intentional and "
                "the result is for a side experiment (not paper main results), "
                "use load_corpus() instead."
            )
    return IfThenCorpus.from_jsonl(p)


def dump_corpus(corpus: IfThenCorpus, path: str | Path) -> None:
    corpus.to_jsonl(path)


# Lightweight helper for ad-hoc construction in scripts/notebooks.
def make_atom(
    *,
    id: str,
    situation: str,
    behavior: str,
    target_dimension: str | None = None,
    polarity: str | None = None,
    affect: str | None = None,
    goal: str | None = None,
    provenance: dict | None = None,
) -> IfThenAtom:
    return IfThenAtom(
        id=id,
        situation=situation,
        behavior=behavior,
        target_dimension=target_dimension,
        polarity=polarity,
        affect=affect,
        goal=goal,
        provenance=provenance,
    )


def _read_jsonl(path: Path) -> list[dict]:
    """Internal helper kept for tests; not part of the public API."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
