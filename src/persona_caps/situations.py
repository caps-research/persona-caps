"""Cross-situational consistency dataset + similarity metrics.

Used by experiment 02 (RQ2). The dataset is a list of "base situations",
each with several paraphrases. The CAPS-faithful prediction is:

    within-paraphrase mean similarity  >>  across-base mean similarity

Trait theory predicts both should be high (cross-situational consistency).
A flat persona prompt that ignores situation specifics tends to look more
like the trait-theory prediction; a CAPS agent should show stronger
discrimination between situations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Paraphrase:
    base_id: str
    domain: str
    text: str
    in_corpus_for: tuple[str, ...] = ()


@dataclass
class CrossSituationalSet:
    base_situations: list[dict]

    @classmethod
    def load(cls, path: str | Path) -> "CrossSituationalSet":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(base_situations=data["base_situations"])

    def all_paraphrases(self) -> list[Paraphrase]:
        out: list[Paraphrase] = []
        for base in self.base_situations:
            in_corpus = tuple(base.get("in_corpus_for", []))
            for text in base["paraphrases"]:
                out.append(Paraphrase(
                    base_id=base["id"],
                    domain=base.get("domain", ""),
                    text=text,
                    in_corpus_for=in_corpus,
                ))
        return out

    def held_out_for(self, dimension: str) -> set[str]:
        """Base ids whose corpus-overlap annotation does NOT include `dimension`."""
        return {
            b["id"] for b in self.base_situations
            if dimension not in b.get("in_corpus_for", [])
        }

    def in_corpus_for(self, dimension: str) -> set[str]:
        return {
            b["id"] for b in self.base_situations
            if dimension in b.get("in_corpus_for", [])
        }


@dataclass
class SignatureMetrics:
    """Within / across cosine similarity statistics over response embeddings."""

    within_paraphrase_mean: float
    within_paraphrase_std: float
    across_base_mean: float
    across_base_std: float
    signature_strength: float  # within - across, the CAPS effect size

    def as_dict(self) -> dict[str, float]:
        return {
            "within_paraphrase_mean": self.within_paraphrase_mean,
            "within_paraphrase_std": self.within_paraphrase_std,
            "across_base_mean": self.across_base_mean,
            "across_base_std": self.across_base_std,
            "signature_strength": self.signature_strength,
        }


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-9, None)


def compute_signature(
    paraphrases: list[Paraphrase],
    embeddings: np.ndarray,
    *,
    base_filter: set[str] | None = None,
) -> SignatureMetrics:
    """Compute within-paraphrase vs across-base cosine similarity stats.

    ``paraphrases[i]`` corresponds to row ``embeddings[i]``. Embeddings need
    not be normalized; this function normalizes a copy. If ``base_filter`` is
    given, only paraphrases whose ``base_id`` is in that set contribute (used
    to compute in-corpus vs held-out subset metrics for honest evaluation).
    """
    if len(paraphrases) != embeddings.shape[0]:
        raise ValueError("paraphrases and embeddings length mismatch")

    emb = _normalize(np.asarray(embeddings, dtype=np.float32))
    by_base: dict[str, list[int]] = {}
    for idx, p in enumerate(paraphrases):
        if base_filter is not None and p.base_id not in base_filter:
            continue
        by_base.setdefault(p.base_id, []).append(idx)

    within_sims: list[float] = []
    for base_id, idxs in by_base.items():
        if len(idxs) < 2:
            continue
        for i, j in combinations(idxs, 2):
            within_sims.append(float(emb[i] @ emb[j]))

    across_sims: list[float] = []
    base_ids = list(by_base.keys())
    for i in range(len(base_ids)):
        for j in range(i + 1, len(base_ids)):
            for ai in by_base[base_ids[i]]:
                for bj in by_base[base_ids[j]]:
                    across_sims.append(float(emb[ai] @ emb[bj]))

    w_arr = np.asarray(within_sims) if within_sims else np.asarray([0.0])
    a_arr = np.asarray(across_sims) if across_sims else np.asarray([0.0])
    w_mean, w_std = float(w_arr.mean()), float(w_arr.std(ddof=0))
    a_mean, a_std = float(a_arr.mean()), float(a_arr.std(ddof=0))
    return SignatureMetrics(
        within_paraphrase_mean=w_mean,
        within_paraphrase_std=w_std,
        across_base_mean=a_mean,
        across_base_std=a_std,
        signature_strength=w_mean - a_mean,
    )
