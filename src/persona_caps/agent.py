"""Agents.

Three agent classes share a uniform ``respond(situation) -> AgentResponse``
interface so that experiments can swap them without code changes. They
correspond to the three persona-construction paradigms compared in the
paper:

- ``TraitPromptAgent`` — minimal trait-label system prompt
  (e.g. "You are someone who is high in neuroticism."). The status-quo
  baseline used implicitly by every "give the model a personality" hack.
- ``PersonaDescriptionAgent`` — long natural-language persona description
  (the SOUL.md / character-card style). Strong baseline for the symbolic
  persona-as-text paradigm.
- ``CapsAgent`` — our method. Personality is the if-then corpus; retrieval
  injects only situation-relevant contingencies; system prompt contains
  **no trait labels**.

The CAPS agent system prompt deliberately avoids any trait-level
abstraction. This is the central empirical claim: behavior conditioned
on situation-relevant if-then atoms produces a coherent, measurable
personality without naming any trait.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .corpus import IfThenAtom, IfThenCorpus
from .llm import ChatMessage, LLMClient
from .retriever import Retriever, RetrievalMode

CAPS_SYSTEM_PROMPT = """You are an agent whose dispositions are described below as a set of \
if-then behavioral contingencies, each of the form WHEN <situation> -> <behavior>. \
The contingencies have been retrieved because they are relevant to the current input. \
Respond *as the kind of person these contingencies describe would respond*. \
Do not list the contingencies, do not mention them, do not meta-comment about them. \
Just respond in character. Be concise unless the input invites elaboration.

YOUR DISPOSITIONS (most relevant first):
{contingencies}
"""


@dataclass
class AgentResponse:
    """The agent's reply plus an audit trail of which atoms fired (empty for baselines)."""

    text: str
    fired_atoms: list[tuple[IfThenAtom, float]]
    situation: str


class Agent(Protocol):
    """Common interface for all three agent types."""

    def respond(self, situation: str) -> AgentResponse: ...


class CapsAgent:
    def __init__(
        self,
        corpus: IfThenCorpus,
        llm: LLMClient,
        *,
        retriever: Retriever | None = None,
        top_k: int = 4,
        strip_affect_metadata: bool = False,
    ) -> None:
        self.corpus = corpus
        self.llm = llm
        self.retriever = retriever or Retriever(corpus)
        self.top_k = top_k
        self.strip_affect_metadata = strip_affect_metadata

    def respond(self, situation: str) -> AgentResponse:
        fired = self.retriever.retrieve(situation, k=self.top_k)
        rendered = "\n".join(
            f"- {atom.render(strip_affect=self.strip_affect_metadata)}"
            for atom, _ in fired
        )
        system = CAPS_SYSTEM_PROMPT.format(contingencies=rendered)
        text = self.llm.chat(
            messages=[ChatMessage(role="user", content=situation)],
            system=system,
        )
        return AgentResponse(text=text.strip(), fired_atoms=fired, situation=situation)


class TraitPromptAgent:
    """Baseline: persona via short trait-label system prompt."""

    def __init__(self, trait_prompt: str, llm: LLMClient) -> None:
        # We deliberately keep the same response-style scaffolding so that
        # any difference vs CapsAgent is attributable to the *content* of
        # the persona, not to surface formatting differences.
        self.system = (
            trait_prompt.strip()
            + "\n\nRespond in character. Do not meta-comment about your personality. "
              "Be concise unless the input invites elaboration."
        )
        self.llm = llm

    def respond(self, situation: str) -> AgentResponse:
        text = self.llm.chat(
            messages=[ChatMessage(role="user", content=situation)],
            system=self.system,
        )
        return AgentResponse(text=text.strip(), fired_atoms=[], situation=situation)


class PersonaDescriptionAgent(TraitPromptAgent):
    """Baseline: persona via a longer natural-language description (SOUL.md style).

    Functionally identical to ``TraitPromptAgent`` — the difference is the
    expected length and richness of the input prompt. We keep them as
    distinct classes for experimental clarity in result tables.
    """

    pass


class ConditionalCapsAgent:
    """A single agent whose if-then dispositions depend on the *current context*.

    Background. Mischel's CAPS theory predicts that personality is *not* a
    constant trait expressed identically across all situations: the same person
    is reliably more conscientious at work than with family, more agreeable
    with strangers than with siblings. This intra-individual but cross-
    situational coherence (the "behavioral signature") is what CAPS calls
    personality. Persona-as-string and persona-as-vector approaches struggle
    to express this — a single global string or activation cannot encode
    "be high-C in context A, low-C in context B" without contradiction.

    With if-then atoms, the same idea is mechanical: keep one corpus per
    context, route the situation to the matching corpus, then retrieve and
    inject as usual. The agent's "personality" is therefore literally a
    *family of corpora* indexed by context, and switching contexts is just
    file routing.

    Parameters
    ----------
    contexts : dict[str, IfThenCorpus]
        Map from context name (e.g. ``"work"``, ``"family"``) to the
        corpus that should drive behavior in that context.
    llm : LLMClient
    top_k : int
        Atoms retrieved per ``respond`` call.

    Notes
    -----
    For the first pass we expect callers to pass the context label
    explicitly to ``respond``. An LLM-based context classifier is a
    natural extension but is intentionally left out of v1: hand-labeled
    contexts give a cleaner test of the *expression* mechanism, decoupled
    from the *recognition* mechanism.
    """

    def __init__(
        self,
        contexts: dict[str, IfThenCorpus],
        llm: LLMClient,
        *,
        top_k: int = 4,
    ) -> None:
        if not contexts:
            raise ValueError("ConditionalCapsAgent requires at least one context corpus")
        self.contexts = contexts
        self.llm = llm
        self.top_k = top_k
        self._retrievers = {name: Retriever(corpus) for name, corpus in contexts.items()}

    def respond(self, situation: str, context: str | None = None) -> AgentResponse:
        if context is None:
            raise ValueError(
                "ConditionalCapsAgent.respond requires an explicit context label "
                f"(one of: {sorted(self.contexts)})"
            )
        if context not in self._retrievers:
            raise KeyError(
                f"unknown context {context!r}; available: {sorted(self._retrievers)}"
            )
        fired = self._retrievers[context].retrieve(situation, k=self.top_k)
        rendered = "\n".join(f"- {atom.render()}" for atom, _ in fired)
        system = CAPS_SYSTEM_PROMPT.format(contingencies=rendered)
        text = self.llm.chat(
            messages=[ChatMessage(role="user", content=situation)],
            system=system,
        )
        return AgentResponse(text=text.strip(), fired_atoms=fired, situation=situation)


class JointCapsAgent:
    """Joint multi-trait CAPS agent with per-dimension conditional retrieval.

    Where ``CapsAgent`` corresponds to a *single*-cell persona (e.g. the
    "high neuroticism" agent retrieves only from ``high_neuroticism.jsonl``),
    ``JointCapsAgent`` lets a single agent express an arbitrary Big Five
    *profile* by maintaining one independent retriever per active
    (dimension, polarity) cell and pulling atoms from each cell on every
    query — a direct, deterministic operationalisation of the
    ``P(behavior | situation, persona) = sum_d P(d | persona) * P(atom | situation, d)``
    decomposition discussed in §3.5.

    Parameters
    ----------
    cells : dict[str, IfThenCorpus]
        Map from cell label (e.g. ``"high_neuroticism"``, ``"low_extraversion"``)
        to that cell's atom corpus. Cell labels are exposed as audit-trail
        keys; their string content has no effect on retrieval.
    llm : LLMClient
    dim_weights : dict[str, float] | None
        Per-cell weight ``pi_d`` interpreted as a relative probability mass.
        Auto-normalised to sum to 1. Defaults to a uniform mixture across
        all cells. Passed-in weights need not sum to 1.
    top_k_per_dim : int
        How many atoms to retrieve from each active cell per query. The
        injected prompt therefore contains
        ``(num cells) x top_k_per_dim`` atoms. Set to 1 for parity with
        ``CapsAgent(top_k=4)`` on a 5-cell persona (5 atoms total).
    mode : "argmax" | "softmax"
        Retrieval mode applied to *each* per-cell retriever. Use
        ``"softmax"`` together with ``tau`` to make per-cell retrieval
        genuinely probabilistic.
    tau : float
        Softmax temperature for ``mode="softmax"``.
    rng : numpy.random.Generator | None
        Numpy generator used (a) to seed each per-cell retriever's sampler
        in softmax mode and (b) to do mixture-weighted cell sub-sampling
        when ``sample_cells=True``.
    sample_cells : bool
        If ``False`` (default), retrieve from *every* cell on every query
        — equivalent to a full mixture expectation. If ``True``, sample
        ``cells_per_step`` cells from ``dim_weights`` each query and only
        retrieve from those — a per-step Monte-Carlo realisation of the
        mixture model. Sampling adds extra variance and should be reserved
        for explicit "is the mixture point estimate enough?" sweeps.
    cells_per_step : int
        Number of cells to sample per query when ``sample_cells=True``.

    Notes
    -----
    The system prompt is reused unchanged from ``CapsAgent``: only the
    retrieval policy differs. Atoms from different cells are sorted by
    cosine similarity in the rendered prompt so the LLM sees the most
    situation-relevant ones first regardless of which cell they came from.
    """

    def __init__(
        self,
        cells: dict[str, IfThenCorpus],
        llm: LLMClient,
        *,
        dim_weights: dict[str, float] | None = None,
        top_k_per_dim: int = 1,
        mode: RetrievalMode = "argmax",
        tau: float = 0.1,
        rng: np.random.Generator | None = None,
        sample_cells: bool = False,
        cells_per_step: int = 5,
    ) -> None:
        if not cells:
            raise ValueError("JointCapsAgent requires at least one cell")
        if top_k_per_dim < 1:
            raise ValueError(f"top_k_per_dim must be >=1 (got {top_k_per_dim})")
        self.cells = cells
        self.llm = llm
        self.top_k_per_dim = top_k_per_dim
        self.mode: RetrievalMode = mode
        self.tau = float(tau)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.sample_cells = sample_cells
        self.cells_per_step = min(cells_per_step, len(cells))

        # weight resolution: default uniform, otherwise normalise the given
        # weights and zero-fill any unmentioned cells (cell with weight 0
        # is dropped from the mixture but kept available to swap in later)
        cell_names = list(cells.keys())
        if dim_weights is None:
            w = {name: 1.0 / len(cell_names) for name in cell_names}
        else:
            for name in dim_weights:
                if name not in cells:
                    raise KeyError(f"dim_weights references unknown cell {name!r}")
            total = sum(max(0.0, v) for v in dim_weights.values())
            if total <= 0:
                raise ValueError("dim_weights must sum to a positive total")
            w = {name: max(0.0, dim_weights.get(name, 0.0)) / total for name in cell_names}
        self.dim_weights = w

        # one retriever per cell, each seeded from the parent rng so two
        # JointCapsAgents seeded the same way produce the same retrievals
        self._retrievers: dict[str, Retriever] = {}
        ss = np.random.SeedSequence(int(self.rng.integers(0, 2**31 - 1)))
        sub_seeds = ss.spawn(len(cell_names))
        for name, sub in zip(cell_names, sub_seeds):
            self._retrievers[name] = Retriever(
                cells[name],
                mode=mode,
                tau=tau,
                rng=np.random.default_rng(sub),
            )

    def _select_active_cells(self) -> list[str]:
        if not self.sample_cells:
            return [n for n, w in self.dim_weights.items() if w > 0]
        names = [n for n, w in self.dim_weights.items() if w > 0]
        weights = np.array([self.dim_weights[n] for n in names], dtype=np.float64)
        weights /= weights.sum()
        chosen = self.rng.choice(
            len(names),
            size=min(self.cells_per_step, len(names)),
            replace=False,
            p=weights,
        )
        return [names[int(i)] for i in chosen]

    def respond(self, situation: str) -> AgentResponse:
        active = self._select_active_cells()
        fired_per_cell: list[tuple[IfThenAtom, float]] = []
        for name in active:
            cell_fired = self._retrievers[name].retrieve(situation, k=self.top_k_per_dim)
            fired_per_cell.extend(cell_fired)
        # global re-sort by similarity so the most situation-relevant atoms
        # appear first in the prompt regardless of which cell they came from
        fired = sorted(fired_per_cell, key=lambda pair: -pair[1])
        rendered = "\n".join(f"- {atom.render()}" for atom, _ in fired)
        system = CAPS_SYSTEM_PROMPT.format(contingencies=rendered)
        text = self.llm.chat(
            messages=[ChatMessage(role="user", content=situation)],
            system=system,
        )
        return AgentResponse(text=text.strip(), fired_atoms=fired, situation=situation)
