"""Tests for the dialogue loader / pair maker (no API required)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# extract_from_dialogue.py is a script; import its helpers directly
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "extract_from_dialogue", ROOT / "scripts" / "extract_from_dialogue.py"
)
mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
spec.loader.exec_module(mod)  # type: ignore[union-attr]


def test_load_jsonl_speaker_text(tmp_path: Path) -> None:
    p = tmp_path / "d.jsonl"
    p.write_text(
        "\n".join(
            json.dumps(r) for r in [
                {"speaker": "A", "text": "hi"},
                {"speaker": "B", "text": "hello"},
                {"speaker": "A", "text": "how are you"},
            ]
        )
    )
    turns = mod.load_turns(p)
    assert turns == [("A", "hi"), ("B", "hello"), ("A", "how are you")]


def test_load_json_array_role_content(tmp_path: Path) -> None:
    p = tmp_path / "d.json"
    p.write_text(json.dumps([
        {"role": "user", "content": "what's up"},
        {"role": "assistant", "content": "not much"},
    ]))
    turns = mod.load_turns(p)
    assert turns == [("user", "what's up"), ("assistant", "not much")]


def test_load_jsonl_name_message(tmp_path: Path) -> None:
    p = tmp_path / "d.jsonl"
    p.write_text(
        "\n".join(json.dumps(r) for r in [
            {"name": "Bob", "message": "x"},
            {"name": "Alice", "message": "y"},
        ])
    )
    turns = mod.load_turns(p)
    assert ("Bob", "x") in turns and ("Alice", "y") in turns


def test_make_pairs_only_other_then_target() -> None:
    turns = [
        ("Alice", "1"),
        ("Bob", "2"),    # target turn after non-target -> pair
        ("Bob", "3"),    # target after target -> NOT a pair
        ("Alice", "4"),
        ("Bob", "5"),    # target after non-target -> pair
    ]
    pairs = mod.make_pairs(turns, target="Bob")
    assert pairs == [("1", "2"), ("4", "5")]


def test_make_pairs_max_cap() -> None:
    turns = [("A", str(i)) if i % 2 == 0 else ("B", str(i)) for i in range(10)]
    pairs = mod.make_pairs(turns, target="B", max_pairs=2)
    assert len(pairs) == 2
