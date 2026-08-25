"""Smoke tests that don't require any API keys."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from persona_caps.corpus import IfThenAtom, IfThenCorpus, make_atom  # noqa: E402


def test_atom_render_minimal() -> None:
    a = make_atom(
        id="t1",
        situation="someone interrupts me mid-sentence",
        behavior="I pause, smile, and pick up where I left off",
    )
    rendered = a.render()
    assert rendered.startswith("WHEN someone interrupts me mid-sentence")
    assert "I pause" in rendered


def test_atom_render_with_affect_and_goal() -> None:
    a = make_atom(
        id="t2",
        situation="my boss criticizes my work",
        behavior="I write down each point and ask clarifying questions",
        affect="composed",
        goal="learn what to improve",
    )
    rendered = a.render()
    assert "(felt: composed)" in rendered
    assert "(goal: learn what to improve)" in rendered


def test_corpus_roundtrip(tmp_path: Path) -> None:
    c = IfThenCorpus(
        [
            make_atom(id=f"x{i}", situation=f"situation {i}", behavior=f"behavior {i}")
            for i in range(5)
        ]
    )
    p = tmp_path / "c.jsonl"
    c.to_jsonl(p)
    loaded = IfThenCorpus.from_jsonl(p)
    assert len(loaded) == 5
    assert loaded.atoms[2].situation == "situation 2"


def test_corpus_remove() -> None:
    c = IfThenCorpus(
        [make_atom(id="a", situation="s", behavior="b"), make_atom(id="b", situation="s2", behavior="b2")]
    )
    assert c.remove("a") is True
    assert c.remove("nope") is False
    assert len(c) == 1


def test_seed_jsonl_files_load() -> None:
    """The shipped sample corpora must be valid."""
    for name in ("high_neuroticism.jsonl", "low_neuroticism.jsonl"):
        path = ROOT / "data" / "if_then" / name
        c = IfThenCorpus.from_jsonl(path)
        assert len(c) >= 8, f"{name} should ship with at least 8 atoms"
        for a in c:
            assert isinstance(a, IfThenAtom)
            assert a.situation and a.behavior


def test_baseline_prompts_present_for_all_dim_polarity() -> None:
    """Every Big Five x {high, low} pair must have both baseline prompts."""
    import json

    data = json.loads((ROOT / "data" / "baselines" / "trait_prompts.json").read_text())
    expected_keys = {
        f"{pol}_{dim}"
        for dim in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")
        for pol in ("high", "low")
    }
    actual = set(data.keys()) - {"_doc"}
    assert actual == expected_keys
    for k in expected_keys:
        assert "trait_label" in data[k]
        assert "persona_description" in data[k]
        # Persona descriptions should be substantially longer than trait labels.
        assert len(data[k]["persona_description"]) > 4 * len(data[k]["trait_label"])
