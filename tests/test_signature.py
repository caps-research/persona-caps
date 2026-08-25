"""Unit tests for the cross-situational signature metric."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from persona_caps.situations import (  # noqa: E402
    CrossSituationalSet,
    Paraphrase,
    compute_signature,
)


def test_dataset_loads_and_has_expected_shape() -> None:
    cs = CrossSituationalSet.load(ROOT / "data" / "situations" / "cross_situational.json")
    paraphrases = cs.all_paraphrases()
    # 12 base situations, ~3 paraphrases each
    assert len(cs.base_situations) >= 10
    assert len(paraphrases) >= 30
    by_base: dict[str, int] = {}
    for p in paraphrases:
        by_base[p.base_id] = by_base.get(p.base_id, 0) + 1
    assert all(n >= 2 for n in by_base.values()), "each base must have >=2 paraphrases"


def test_signature_perfect_within_perfect_across_distinct() -> None:
    """If paraphrases of the same base are identical and bases are orthogonal,
    W should be ~1 and A should be ~0, so signature_strength ~ 1."""
    paraphrases = [
        Paraphrase(base_id="a", domain="x", text=""),
        Paraphrase(base_id="a", domain="x", text=""),
        Paraphrase(base_id="b", domain="x", text=""),
        Paraphrase(base_id="b", domain="x", text=""),
    ]
    embeddings = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    m = compute_signature(paraphrases, embeddings)
    assert abs(m.within_paraphrase_mean - 1.0) < 1e-5
    assert abs(m.across_base_mean - 0.0) < 1e-5
    assert abs(m.signature_strength - 1.0) < 1e-5


def test_signature_uniform_responses_zero_signature() -> None:
    """If every response is identical, W = A = 1 and signature_strength = 0.
    This is the trait-theory / flat-persona prediction."""
    paraphrases = [
        Paraphrase(base_id="a", domain="x", text=""),
        Paraphrase(base_id="a", domain="x", text=""),
        Paraphrase(base_id="b", domain="x", text=""),
        Paraphrase(base_id="b", domain="x", text=""),
    ]
    embeddings = np.tile(np.array([1.0, 0.0], dtype=np.float32), (4, 1))
    m = compute_signature(paraphrases, embeddings)
    assert abs(m.within_paraphrase_mean - 1.0) < 1e-5
    assert abs(m.across_base_mean - 1.0) < 1e-5
    assert abs(m.signature_strength) < 1e-5


def test_signature_paraphrases_inconsistent_negative_signature() -> None:
    """If responses to paraphrases of the same base are unrelated (W low) but
    cross-base responses happen to align (A high), signature is negative.
    This is the pathological case we want to detect."""
    paraphrases = [
        Paraphrase(base_id="a", domain="x", text=""),
        Paraphrase(base_id="a", domain="x", text=""),
        Paraphrase(base_id="b", domain="x", text=""),
        Paraphrase(base_id="b", domain="x", text=""),
    ]
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    m = compute_signature(paraphrases, embeddings)
    assert m.signature_strength < 0
