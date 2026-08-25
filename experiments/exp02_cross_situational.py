"""Experiment 02 — RQ2: Cross-situational behavioral signature.

The CAPS theory predicts: a real personality produces stable responses to
*the same situation paraphrased differently* (within-paraphrase consistency)
and *distinct* responses across genuinely different situations
(across-base discrimination). Trait theory predicts cross-situational
*uniformity* — the same level of, say, extraversion regardless of
situation. A flat persona prompt that doesn't condition on situation
specifics tends to produce more uniform output than a CAPS-RAG agent
that retrieves *situation-relevant* contingencies.

Operationally, we run each agent over all 36 prompts (12 base situations
× 3 paraphrases), embed each response, and compute:

    W = mean cosine sim within each paraphrase set
    A = mean cosine sim across base situations
    signature_strength = W - A

Higher W = behavioral consistency under paraphrase.
Higher (W - A) = stronger situation-behavior signature.

Predictions tested:
    1. CAPS agent has W - A significantly larger than the trait-label baseline.
    2. CAPS agent has W comparable to or higher than the persona-description baseline.

Usage:
    python experiments/exp02_cross_situational.py --backend minimax \\
        --dimension neuroticism --polarity high \\
        --out runs/exp02_neuroticism_high_minimax.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from persona_caps.agent import (  # noqa: E402
    AgentResponse,
    CapsAgent,
    PersonaDescriptionAgent,
    TraitPromptAgent,
)
from persona_caps.corpus import IfThenCorpus  # noqa: E402
from persona_caps.llm import get_client  # noqa: E402
from persona_caps.situations import CrossSituationalSet, compute_signature  # noqa: E402

DEFAULT_CORPUS_DIR = ROOT / "data" / "if_then"
BASELINES = json.loads((ROOT / "data" / "baselines" / "trait_prompts.json").read_text())
SITUATIONS_PATH = ROOT / "data" / "situations" / "cross_situational.json"
NOISE_CORPUS_PATH = ROOT / "data" / "if_then" / "noise_demos.jsonl"

METHODS = ("caps", "trait_label", "persona_description", "random_demos")


def _build_agent(method: str, dimension: str, polarity: str, llm, corpus_dir: Path) -> object:
    key = f"{polarity}_{dimension}"
    if method == "caps":
        path = corpus_dir / f"{key}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"corpus missing: {path}; run scripts/seed_corpus.py")
        return CapsAgent(corpus=IfThenCorpus.from_jsonl(path), llm=llm, top_k=4)
    if method == "trait_label":
        return TraitPromptAgent(BASELINES[key]["trait_label"], llm=llm)
    if method == "persona_description":
        return PersonaDescriptionAgent(BASELINES[key]["persona_description"], llm=llm)
    if method == "random_demos":
        if not NOISE_CORPUS_PATH.exists():
            raise FileNotFoundError(f"noise corpus missing: {NOISE_CORPUS_PATH}")
        return CapsAgent(corpus=IfThenCorpus.from_jsonl(NOISE_CORPUS_PATH), llm=llm, top_k=4)
    raise ValueError(method)


def _embed_responses(texts: list[str]) -> np.ndarray:
    """Embed using the same default model as the retriever, for consistency."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return np.asarray(model.encode(texts, normalize_embeddings=False, show_progress_bar=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="minimax")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--dimension", required=True)
    parser.add_argument("--polarity", required=True, choices=["high", "low"])
    parser.add_argument(
        "--methods", nargs="+", default=list(METHODS), choices=list(METHODS),
    )
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel LLM calls per method")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR,
                        help="Directory containing {polarity}_{dim}.jsonl. "
                             "Default = legacy hand-authored seeds. For paper-track "
                             "use data/if_then/paper/beck_2022_v2/")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cs = CrossSituationalSet.load(SITUATIONS_PATH)
    paraphrases = cs.all_paraphrases()
    print(f"[exp02] {len(paraphrases)} prompts ({len(cs.base_situations)} base x ~3 paraphrases)")

    llm = get_client(args.backend, temperature=args.temperature)

    per_method: dict[str, dict] = {}

    for method in args.methods:
        print(f"\n[exp02] running {method} ({args.polarity} {args.dimension})")
        try:
            agent = _build_agent(method, args.dimension, args.polarity, llm, args.corpus_dir)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        texts_buf: list[str | None] = [None] * len(paraphrases)
        responses_buf: list[dict | None] = [None] * len(paraphrases)

        def _ask_one(idx: int, prompt_text: str, base_id: str, domain: str) -> tuple[int, str, dict]:
            r: AgentResponse = agent.respond(prompt_text)  # type: ignore[union-attr]
            return idx, r.text, {
                "base_id": base_id,
                "domain": domain,
                "prompt": prompt_text,
                "response": r.text,
            }

        if args.workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [
                    pool.submit(_ask_one, i, p.text, p.base_id, p.domain)
                    for i, p in enumerate(paraphrases)
                ]
                done = 0
                for f in as_completed(futures):
                    idx, txt, rec = f.result()
                    texts_buf[idx] = txt
                    responses_buf[idx] = rec
                    done += 1
                    if done % 6 == 0:
                        print(f"  [{done}/{len(paraphrases)}] done")
        else:
            for i, p in enumerate(paraphrases):
                idx, txt, rec = _ask_one(i, p.text, p.base_id, p.domain)
                texts_buf[idx] = txt
                responses_buf[idx] = rec
                if (i + 1) % 6 == 0:
                    print(f"  [{i + 1}/{len(paraphrases)}] done")
        texts: list[str] = [t for t in texts_buf if t is not None]
        responses: list[dict] = [r for r in responses_buf if r is not None]

        emb = _embed_responses(texts)
        metrics = compute_signature(paraphrases, emb)
        per_method[method] = {"metrics": metrics.as_dict(), "responses": responses}

        m = metrics
        print(
            f"  W={m.within_paraphrase_mean:.3f}±{m.within_paraphrase_std:.3f}  "
            f"A={m.across_base_mean:.3f}±{m.across_base_std:.3f}  "
            f"W-A={m.signature_strength:+.3f}"
        )

    print("\n=== RQ2 summary ===")
    print(f"target: {args.polarity} {args.dimension}  backend: {args.backend}\n")
    print("| method | W (within-paraphrase) | A (across-base) | W-A (signature) |")
    print("| --- | --- | --- | --- |")
    for method in args.methods:
        if method not in per_method:
            print(f"| {method} | — | — | — |")
            continue
        m = per_method[method]["metrics"]
        print(
            f"| {method} | {m['within_paraphrase_mean']:.3f}±{m['within_paraphrase_std']:.3f} "
            f"| {m['across_base_mean']:.3f}±{m['across_base_std']:.3f} "
            f"| {m['signature_strength']:+.3f} |"
        )

    print(
        "\nCAPS prediction: W-A is larger for the 'caps' row than for 'trait_label'.\n"
        "Higher W-A means the agent treats different situations as different — \n"
        "the empirical signature of CAPS-style personality (Mischel & Shoda 1995)."
    )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "backend": args.backend,
                "temperature": args.temperature,
                "dimension": args.dimension,
                "polarity": args.polarity,
                "methods": args.methods,
                "corpus_dir": str(args.corpus_dir),
            },
            "per_method": per_method,
        }
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[exp02] wrote -> {out}")


if __name__ == "__main__":
    main()
