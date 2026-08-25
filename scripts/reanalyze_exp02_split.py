"""Re-analyze an exp02 result file with the in-corpus vs held-out split.

The smoke RQ2 results conflated situations that *do* appear in the
seed corpus with situations that *do not*. This script re-embeds the
saved responses (no new LLM calls) and reports W − A separately on:

    * full        — all 36 prompts (matches the original headline)
    * in_corpus   — base situations whose `in_corpus_for` includes the
                    dimension under test
    * held_out    — base situations whose `in_corpus_for` does NOT
                    include the dimension under test

Usage:
    python scripts/reanalyze_exp02_split.py runs/exp02_smoke_neuroticism_high.json
"""

from __future__ import annotations

import argparse
import json
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

SITUATIONS_PATH = ROOT / "data" / "situations" / "cross_situational.json"


def _embed(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return np.asarray(model.encode(texts, normalize_embeddings=False, show_progress_bar=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="path to exp02 result JSON")
    parser.add_argument("--out", default=None, help="optional path to write split-metric JSON")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    cfg = data["config"]
    dim = cfg["dimension"]
    polarity = cfg["polarity"]

    cs = CrossSituationalSet.load(SITUATIONS_PATH)
    in_corpus = cs.in_corpus_for(dim)
    held_out = cs.held_out_for(dim)
    print(f"[reanalyze] dimension={dim} polarity={polarity}")
    print(f"  in_corpus base ids ({len(in_corpus)}): {sorted(in_corpus)}")
    print(f"  held_out  base ids ({len(held_out)}): {sorted(held_out)}\n")

    print("| method | subset | n_paraphrases | W | A | W-A |")
    print("| --- | --- | --- | --- | --- | --- |")

    per_method_split: dict[str, dict] = {}
    for method, payload in data["per_method"].items():
        responses = payload["responses"]
        paraphrases = [
            Paraphrase(
                base_id=r["base_id"],
                domain=r.get("domain", ""),
                text=r.get("prompt", ""),
                in_corpus_for=tuple(["?"]),
            )
            for r in responses
        ]
        emb = _embed([r["response"] for r in responses])
        per_method_split[method] = {}
        for label, base_filter in [
            ("full", None),
            ("in_corpus", in_corpus),
            ("held_out", held_out),
        ]:
            n_para = sum(1 for p in paraphrases if base_filter is None or p.base_id in base_filter)
            if n_para < 4:
                print(f"| {method} | {label} | {n_para} | — | — | (insufficient) |")
                per_method_split[method][label] = None
                continue
            metrics = compute_signature(paraphrases, emb, base_filter=base_filter)
            per_method_split[method][label] = metrics.as_dict()
            print(
                f"| {method} | {label} | {n_para} "
                f"| {metrics.within_paraphrase_mean:.3f} "
                f"| {metrics.across_base_mean:.3f} "
                f"| {metrics.signature_strength:+.3f} |"
            )
        print("|" + " --- |" * 6)

    if args.out:
        out_payload = {
            "source": str(args.input),
            "config": cfg,
            "in_corpus_base_ids": sorted(in_corpus),
            "held_out_base_ids": sorted(held_out),
            "per_method": per_method_split,
        }
        Path(args.out).write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
        print(f"\n[reanalyze] wrote -> {args.out}")


if __name__ == "__main__":
    main()
