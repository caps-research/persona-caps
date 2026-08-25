"""Personality inventory administration and scoring.

We model an inventory as a list of items, each with a target dimension
and a polarity (``"+"`` for keyed-positive, ``"-"`` for keyed-negative /
reverse-scored). The agent answers each item on a Likert scale; we
extract the integer rating, reverse-score where needed, and aggregate by
dimension.

Inventory file format (JSON)::

    {
      "name": "ipip_neo_short",
      "scale": {"min": 1, "max": 5, "labels": ["Strongly disagree", ..., "Strongly agree"]},
      "items": [
        {"id": "ipip_001", "text": "I am the life of the party.",
         "dimension": "extraversion", "keyed": "+"},
        ...
      ]
    }
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev

from .agent import CapsAgent

LIKERT_PROMPT = """Below is a personality questionnaire item. \
Rate how well it describes you on the given scale, then briefly justify in one sentence.

Scale: {scale_text}

Item: "{item_text}"

Respond in EXACTLY this format:
RATING: <integer between {scale_min} and {scale_max}>
WHY: <one sentence>
"""

_RATING_RE = re.compile(r"RATING\s*:\s*(-?\d+)", re.IGNORECASE)


@dataclass
class InventoryItem:
    id: str
    text: str
    dimension: str
    keyed: str  # "+" or "-"


@dataclass
class Inventory:
    name: str
    scale_min: int
    scale_max: int
    scale_labels: list[str]
    items: list[InventoryItem]

    @classmethod
    def load(cls, path: str | Path) -> "Inventory":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        scale = data["scale"]
        items = [InventoryItem(**i) for i in data["items"]]
        return cls(
            name=data["name"],
            scale_min=int(scale["min"]),
            scale_max=int(scale["max"]),
            scale_labels=list(scale["labels"]),
            items=items,
        )

    def scale_text(self) -> str:
        return ", ".join(f"{i + self.scale_min}={lbl}" for i, lbl in enumerate(self.scale_labels))


@dataclass
class ItemResponse:
    item: InventoryItem
    raw_text: str
    rating: int | None
    scored: float | None  # reverse-scored if needed; None if parsing failed
    # Audit trail of which atoms fired for this item, in retrieval order.
    # Empty for non-retrieval agents (TraitPromptAgent, PersonaDescriptionAgent).
    # Each entry: (atom_id, cosine_similarity). Atom id is preferred over the
    # full atom object so the result is JSON-serialisable for downstream
    # robustness analysis (e.g. trait-aligned hit-rate vs. similarity).
    fired: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class InventoryReport:
    inventory_name: str
    responses: list[ItemResponse]
    dimension_scores: dict[str, float] = field(default_factory=dict)
    dimension_n: dict[str, int] = field(default_factory=dict)
    parse_failures: int = 0

    def summary(self) -> str:
        lines = [f"Inventory: {self.inventory_name}", f"Parse failures: {self.parse_failures}"]
        for dim, score in sorted(self.dimension_scores.items()):
            lines.append(f"  {dim:20s}  mean={score:.2f}  n={self.dimension_n[dim]}")
        return "\n".join(lines)


def _parse_rating(text: str, scale_min: int, scale_max: int) -> int | None:
    m = _RATING_RE.search(text)
    if not m:
        return None
    val = int(m.group(1))
    if val < scale_min or val > scale_max:
        return None
    return val


def administer(
    agent: CapsAgent,
    inventory: Inventory,
    *,
    workers: int = 1,
) -> InventoryReport:
    """Administer the full inventory; one LLM call per item.

    Set ``workers`` > 1 to fan items out concurrently (a thread pool over
    the synchronous LLM client). Item order in the returned report still
    matches inventory order.
    """
    responses: list[ItemResponse | None] = [None] * len(inventory.items)

    def _ask(idx: int, item: InventoryItem) -> tuple[int, ItemResponse]:
        prompt = LIKERT_PROMPT.format(
            scale_text=inventory.scale_text(),
            item_text=item.text,
            scale_min=inventory.scale_min,
            scale_max=inventory.scale_max,
        )
        # Note: we pass the questionnaire item through the agent so retrieval still fires.
        # This is intentional: we want the persona's dispositions to influence its self-report.
        resp = agent.respond(prompt)
        rating = _parse_rating(resp.text, inventory.scale_min, inventory.scale_max)
        if rating is None:
            scored: float | None = None
        elif item.keyed == "-":
            scored = float(inventory.scale_max + inventory.scale_min - rating)
        else:
            scored = float(rating)
        fired = [(a.id, float(s)) for a, s in getattr(resp, "fired_atoms", [])]
        return idx, ItemResponse(
            item=item, raw_text=resp.text, rating=rating, scored=scored, fired=fired,
        )

    if workers <= 1:
        for i, item in enumerate(inventory.items):
            responses[i] = _ask(i, item)[1]
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_ask, i, item) for i, item in enumerate(inventory.items)]
            for f in as_completed(futures):
                i, ir = f.result()
                responses[i] = ir

    by_dim: dict[str, list[float]] = defaultdict(list)
    parse_failures = 0
    for r in responses:
        assert r is not None
        if r.scored is None:
            parse_failures += 1
        else:
            by_dim[r.item.dimension].append(r.scored)

    dim_scores = {d: mean(vs) for d, vs in by_dim.items() if vs}
    dim_n = {d: len(vs) for d, vs in by_dim.items()}
    return InventoryReport(
        inventory_name=inventory.name,
        responses=[r for r in responses if r is not None],
        dimension_scores=dim_scores,
        dimension_n=dim_n,
        parse_failures=parse_failures,
    )


def cronbach_alpha(item_score_matrix: list[list[float]]) -> float:
    """Cronbach's alpha over an N_subjects x K_items matrix.

    Used in RQ1: internal consistency of each Big Five subscale across
    multiple agent runs. With a single agent, run it K times with
    temperature>0 to get the subject dimension.
    """
    import numpy as np

    X = np.asarray(item_score_matrix, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("expected 2D matrix [subjects x items]")
    n_subjects, k = X.shape
    if k < 2 or n_subjects < 2:
        return float("nan")
    item_var = X.var(axis=0, ddof=1).sum()
    total_var = X.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float("nan")
    return float((k / (k - 1)) * (1 - item_var / total_var))
