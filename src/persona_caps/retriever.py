"""Situation-similarity retriever over an if-then corpus.

Default backend uses ``sentence-transformers`` (all-MiniLM-L6-v2, 384-dim)
because it is small, CPU-friendly, and sufficient for short-string
situation matching. Swap via the ``embed_fn`` argument to use anything
else (OpenAI embeddings, BGE, instruction-tuned, etc.).

The retriever indexes the *situation* field of each atom only — that is
the "if" half of the contingency, which is what the runtime situation
should match. The "then" half is what we want to inject; we do not want
to bias retrieval on it.

Two retrieval modes are supported:

- ``"argmax"`` (default, used in the §4.1 / §4.4 headline experiments):
  deterministic top-k by cosine similarity. This is a MAP / point estimate
  of the conditional distribution P(atom | query).
- ``"softmax"`` (used in the App. C conditional-retrieval ablation): genuine
  sampling from the temperature-softmaxed similarity distribution
      P(atom | query) = softmax(sim(query, atom) / tau).
  This is the operational implementation of CAPS-as-conditional-probability;
  ``tau -> 0`` recovers ``argmax``, ``tau -> inf`` recovers uniform sampling
  (i.e. the ``random_demos`` ablation), and intermediate ``tau`` interpolates.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

from .corpus import IfThenAtom, IfThenCorpus

EmbedFn = Callable[[list[str]], np.ndarray]
RetrievalMode = Literal["argmax", "softmax"]


def _default_embed_fn() -> EmbedFn:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed(texts: list[str]) -> np.ndarray:
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    return embed


class Retriever:
    def __init__(
        self,
        corpus: IfThenCorpus,
        embed_fn: EmbedFn | None = None,
        *,
        mode: RetrievalMode = "argmax",
        tau: float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Build a retriever over an if-then corpus.

        Parameters
        ----------
        corpus
            The atom corpus to retrieve from.
        embed_fn
            Optional embedding function override.
        mode
            ``"argmax"`` (default): deterministic top-k by cosine similarity.
            ``"softmax"``: sample atoms from softmax(sim/tau) over the full
            corpus; this is the conditional-probability variant.
        tau
            Softmax temperature for ``mode="softmax"``. Only used in that
            mode. Smaller tau -> sharper distribution -> closer to argmax;
            larger tau -> flatter -> closer to uniform.
        rng
            Numpy random generator for ``mode="softmax"``. If ``None``, a
            fresh generator is created (non-deterministic). Pass an
            explicitly-seeded generator from experiment scripts when
            reproducibility matters.
        """
        if len(corpus) == 0:
            raise ValueError("cannot build retriever over empty corpus")
        if mode not in ("argmax", "softmax"):
            raise ValueError(f"unknown mode={mode!r}; expected 'argmax' or 'softmax'")
        if tau <= 0:
            raise ValueError(f"tau must be positive (got {tau}); use mode='argmax' for tau->0 limit")
        self.corpus = corpus
        self.embed_fn = embed_fn or _default_embed_fn()
        self._index: np.ndarray = self.embed_fn(corpus.situations())
        self.mode: RetrievalMode = mode
        self.tau: float = float(tau)
        self.rng = rng if rng is not None else np.random.default_rng()

    def retrieve(
        self,
        situation: str,
        *,
        k: int = 4,
        mode: RetrievalMode | None = None,
        tau: float | None = None,
    ) -> list[tuple[IfThenAtom, float]]:
        """Return ``k`` atoms scored by cosine similarity to ``situation``.

        ``mode`` and ``tau`` default to the values set at ``__init__``;
        passing them per-call lets a single retriever be reused under
            different sampling regimes (used by the App. C sweeps).

        Returns
        -------
        list of ``(atom, similarity)`` tuples. The returned ``similarity``
        field is the *cosine similarity* (not the softmax probability),
        so audit logs are comparable across modes.
        """
        active_mode: RetrievalMode = mode if mode is not None else self.mode
        active_tau: float = tau if tau is not None else self.tau

        q = self.embed_fn([situation])[0]
        # Suppress spurious FP-exception flags: sentence-transformers / Accelerate
        # can leave divide/overflow/invalid bits set even when both operands are
        # clean unit-norm float32. We verified the inputs (no NaN/Inf, norms=1.0)
        # so the warnings carry no information about the actual cosine values.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = self._index @ q
        k = min(k, len(self.corpus))

        if active_mode == "argmax":
            idx = np.argpartition(-sims, k - 1)[:k]
            idx = idx[np.argsort(-sims[idx])]
        else:
            # softmax sampling without replacement
            logits = sims / active_tau
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            idx = self.rng.choice(len(sims), size=k, replace=False, p=probs)
            # sort returned atoms by similarity for stable downstream ordering
            idx = np.array(sorted(idx, key=lambda i: -sims[int(i)]))

        return [(self.corpus.atoms[int(i)], float(sims[int(i)])) for i in idx]
