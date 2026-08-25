"""PersonalityBench head-to-head: CAPS (Beck v2) vs SP/P² baselines.

Reproduces the IRiS / NPTI evaluation protocol on Llama-replaced MiniMax to
get apples-to-apples numbers we can place next to IRiS Table 1.

Stage 1 (this script): generate responses for each (trait × polarity × method × question).
Stage 2 (judge_personalitybench.py): GPT-4o LLM-as-judge on a 1-5 scale.

Output JSONL format matches NPTI/code/gpt4_score.py expectations:
    {"question": str, "answer": str}
written to runs/personalitybench/{method}/{Trait}_{pol}.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# allow importing src/persona_caps without install
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from persona_caps.agent import CapsAgent, TraitPromptAgent
from persona_caps.corpus import load_corpus_paper_only
from persona_caps.llm import LLMClient
from persona_caps.retriever import Retriever


# IRiS / NPTI exact baseline templates (baselines/NPTI/NPTI/code/baseline_prompt.py)
TEMPLATE_SP = (
    "Imagine you are {personality} person rather than a language model, and "
    "you're asked by the following question. Write your response based on your "
    "authentic thoughts and emotions. \n\n"
    "Do not overthink your answer—let your thoughts flow naturally as you "
    "write. Focus on expressing your genuine feelings and reactions. Aim to "
    "write no more than 300 words.\n\n"
    "### Question:\n{question}\n\n### Response:\n"
)

# P² template (verbatim from baselines/NPTI/NPTI/code/baseline_prompt.py:TEMPLATE_p2)
TEMPLATE_P2 = (
    "Imagine you are a real person rather than a language model. "
    "{personality}.Now you're asked by the following question. Write your "
    "response based on your authentic thoughts and emotions. \n\n"
    "Do not overthink your answer—let your thoughts flow naturally as you "
    "write. Focus on expressing your genuine feelings and reactions. Aim to "
    "write no more than 300 words.\n\n"
    "### Question:\n{question}\n\n### Response:\n"
)

PERSONALITY_ADJ = {
    ("Extraversion", "high"): "an extraverted",
    ("Extraversion", "low"): "an introverted",
    ("Openness", "high"): "an open",
    ("Openness", "low"): "a closed",
    ("Conscientiousness", "high"): "a conscientious",
    ("Conscientiousness", "low"): "an unconscientious",
    ("Agreeableness", "high"): "an agreeable",
    ("Agreeableness", "low"): "a disagreeable",
    ("Neuroticism", "high"): "a neurotic",
    ("Neuroticism", "low"): "a calm",
}

# P² requires a paragraph-level personality description per (trait, polarity).
# We use IRiS / NPTI's released description bank (baselines/NPTI/NPTI/dataset/
# description.json) and follow their per-call random-choice protocol so that
# each question samples one of multiple equivalently-targeted descriptions.
P2_KEY = {
    ("Extraversion", "high"): "Extraversion",
    ("Extraversion", "low"): "Extraversion_reversed",
    ("Openness", "high"): "Openness",
    ("Openness", "low"): "Openness_reversed",
    ("Conscientiousness", "high"): "Conscientiousness",
    ("Conscientiousness", "low"): "Conscientiousness_reversed",
    ("Agreeableness", "high"): "Agreeableness",
    ("Agreeableness", "low"): "Agreeableness_reversed",
    ("Neuroticism", "high"): "Neuroticism",
    ("Neuroticism", "low"): "Neuroticism_reversed",
}

TRAIT_DIR = {
    "Extraversion": "extraversion",
    "Openness": "openness",
    "Conscientiousness": "conscientiousness",
    "Agreeableness": "agreeableness",
    "Neuroticism": "neuroticism",
}


def load_p2_descriptions(path: str | Path) -> dict[str, list[str]]:
    """Load NPTI's description.json — one JSON object per line, each object
    maps every (trait/trait_reversed) key to a paragraph-level description.
    Concatenates all rows so each (trait, polarity) gets a list of equivalent
    descriptions to sample from per question (matching NPTI baseline_prompt.py)."""
    bank: dict[str, list[str]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key, val in row.items():
                bank.setdefault(key, []).append(val)
    return bank


def load_questions(trait: str, qdir: Path) -> list[str]:
    path = qdir / f"{trait}.json"
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line)["question"])
    return out


def run_caps(
    questions: list[str],
    trait: str,
    pol: str,
    llm: LLMClient,
    corpus_dir: Path,
    top_k: int,
    workers: int,
    strip_affect_metadata: bool = False,
) -> list[dict]:
    pole_file = corpus_dir / f"{pol}_{TRAIT_DIR[trait]}.jsonl"
    corpus = load_corpus_paper_only(pole_file)
    retriever = Retriever(corpus)
    agent = CapsAgent(
        corpus=corpus,
        llm=llm,
        retriever=retriever,
        top_k=top_k,
        strip_affect_metadata=strip_affect_metadata,
    )
    out: list[dict | None] = [None] * len(questions)

    def one(i: int, q: str) -> tuple[int, dict]:
        resp = agent.respond(q)
        return i, {"question": q, "answer": resp.text, "fired_atom_ids": [a.id for a, _ in resp.fired_atoms]}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, i, q) for i, q in enumerate(questions)]
        for f in as_completed(futs):
            i, row = f.result()
            out[i] = row
    return out  # type: ignore


def run_sp(
    questions: list[str],
    trait: str,
    pol: str,
    llm: LLMClient,
    workers: int,
) -> list[dict]:
    persona = PERSONALITY_ADJ[(trait, pol)]
    agent = TraitPromptAgent(trait_prompt=f"You are {persona} person.", llm=llm)
    out: list[dict | None] = [None] * len(questions)

    def one(i: int, q: str) -> tuple[int, dict]:
        prompt = TEMPLATE_SP.format(personality=persona, question=q)
        text = llm.chat(messages=[{"role": "user", "content": prompt}], system=None) if False else None
        from persona_caps.llm import ChatMessage
        text = llm.chat(messages=[ChatMessage(role="user", content=prompt)])
        return i, {"question": q, "answer": text.strip()}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, i, q) for i, q in enumerate(questions)]
        for f in as_completed(futs):
            i, row = f.result()
            out[i] = row
    return out  # type: ignore


def run_p2(
    questions: list[str],
    trait: str,
    pol: str,
    llm: LLMClient,
    p2_bank: dict[str, list[str]],
    workers: int,
    rng_seed: int = 20260511,
) -> list[dict]:
    """P² baseline: paragraph-level personality description sampled per call
    (matching NPTI/code/baseline_prompt.py random.choice over description bank).
    """
    import random as _random
    from persona_caps.llm import ChatMessage

    key = P2_KEY[(trait, pol)]
    descs = p2_bank.get(key, [])
    if not descs:
        raise ValueError(f"P² description bank empty for {key}")
    rng = _random.Random(rng_seed)
    # pre-pick descriptions deterministically given seed for reproducibility
    picks = [rng.choice(descs) for _ in questions]

    out: list[dict | None] = [None] * len(questions)

    def one(i: int, q: str) -> tuple[int, dict]:
        prompt = TEMPLATE_P2.format(personality=picks[i], question=q)
        text = llm.chat(messages=[ChatMessage(role="user", content=prompt)])
        return i, {"question": q, "answer": text.strip(), "p2_description_used": picks[i][:80] + "..."}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, i, q) for i, q in enumerate(questions)]
        for f in as_completed(futs):
            i, row = f.result()
            out[i] = row
    return out  # type: ignore


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", required=True, choices=["caps", "sp", "p2"], help="Which method to generate")
    p.add_argument("--p2-descriptions", default="baselines/NPTI/NPTI/dataset/description.json",
                   help="Path to NPTI description bank for P² baseline")
    p.add_argument("--backend", default="minimax")
    p.add_argument("--model", default="MiniMax-M2.7-highspeed")
    p.add_argument("--corpus-dir", default="data/if_then/paper/beck_2022_v2")
    p.add_argument("--questions-dir", default="baselines/NPTI/NPTI/dataset/test")
    p.add_argument("--out-dir", default="runs/personalitybench")
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--traits", nargs="+", default=list(TRAIT_DIR.keys()))
    p.add_argument("--polarities", nargs="+", default=["high", "low"])
    p.add_argument("--limit", type=int, default=None, help="Cap questions per condition (debug)")
    p.add_argument(
        "--strip-affect-metadata",
        action="store_true",
        help=(
            "CAPS-only ablation: drop the (felt: ...) and (goal: ...) parentheticals "
            "from rendered atoms before injecting into the system prompt. Targets the "
            "L8B regurgitation failure mode reported in §A.4.1."
        ),
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir) / args.method
    out_dir.mkdir(parents=True, exist_ok=True)
    qdir = Path(args.questions_dir)
    cdir = Path(args.corpus_dir)

    llm = LLMClient(backend=args.backend, model=args.model)

    total_t = time.time()
    for trait in args.traits:
        questions = load_questions(trait, qdir)
        if args.limit:
            questions = questions[: args.limit]
        for pol in args.polarities:
            t = time.time()
            outfile = out_dir / f"{trait}_{pol}.jsonl"
            if outfile.exists() and outfile.stat().st_size > 0:
                # skip already-done
                with open(outfile) as f:
                    n_done = sum(1 for _ in f)
                if n_done >= len(questions):
                    print(f"[skip] {trait}/{pol}: already {n_done} responses", flush=True)
                    continue
            print(f"[gen]  {trait}/{pol} method={args.method} N={len(questions)} ...", flush=True)
            if args.method == "caps":
                rows = run_caps(
                    questions, trait, pol, llm, cdir, args.top_k, args.workers,
                    strip_affect_metadata=args.strip_affect_metadata,
                )
            elif args.method == "sp":
                rows = run_sp(questions, trait, pol, llm, args.workers)
            elif args.method == "p2":
                if "p2_bank" not in locals():
                    p2_bank = load_p2_descriptions(args.p2_descriptions)
                    print(f"[init] P² description bank: {sum(len(v) for v in p2_bank.values())} descriptions across {len(p2_bank)} (trait, polarity) keys", flush=True)
                rows = run_p2(questions, trait, pol, llm, p2_bank, args.workers)
            else:
                raise ValueError(f"unknown method: {args.method}")
            with open(outfile, "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[done] {trait}/{pol} {len(rows)} rows in {time.time()-t:.1f}s -> {outfile}", flush=True)
    print(f"=== ALL DONE in {time.time()-total_t:.1f}s ===")


if __name__ == "__main__":
    main()
