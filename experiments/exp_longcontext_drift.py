"""Long-context drift — does persona-as-string degrade with conversation length?

Hypothesis. A SOUL.md / persona-string injected once into the system prompt has
to compete with everything else in the context window for the model's
attention. As the conversation history grows, the persona instructions get
relatively smaller, more "buried", and the model drifts back toward its base
distribution. CAPS, in contrast, re-anchors *every* turn: the retrieved
contingencies are inserted into the system prompt fresh on every call,
based on the current situation, regardless of how much chitchat preceded.

We test this directly. For each (probe, history_length, method):
1. Prepend `history_length` *unrelated* chitchat turns (`data/conditional/history_filler.json`).
2. Append the probe as the final user turn.
3. Generate the agent's response.
4. Score the response 1-5 on conscientiousness expression (GPT-4o judge).

Methods (all configured for HIGH conscientiousness):
* `caps`           — `CapsAgent` over the high-conscientiousness paper-track corpus.
* `persona_string` — `PersonaDescriptionAgent` whose system prompt is a plain
   English description of a highly conscientious person (the SOUL.md /
   character-card analogue).
* `trait_label`    — `TraitPromptAgent` minimal "You are someone who is highly
   conscientious." prompt — an extra reference point.

We then plot mean conscientiousness rating as a function of history length
for each method. The CAPS prediction: roughly flat. The string prediction:
monotonically decreasing as the persona gets diluted.

Usage:
    python experiments/exp_longcontext_drift.py \
        --gen-backend gemini-flash --judge-backend openai \
        --history-lengths 0 5 20 50 \
        --runs 2 \
        --out runs/longcontext/run.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from persona_caps.agent import (  # noqa: E402
    CapsAgent,
    PersonaDescriptionAgent,
    TraitPromptAgent,
    CAPS_SYSTEM_PROMPT,
)
from persona_caps.corpus import IfThenCorpus  # noqa: E402
from persona_caps.llm import ChatMessage, get_client  # noqa: E402

PAPER_CORPUS_DIR = ROOT / "data" / "if_then" / "paper" / "beck_2022_v2"
HIGH_C_PATH = PAPER_CORPUS_DIR / "high_conscientiousness.jsonl"
SCENARIOS_PATH = ROOT / "data" / "conditional" / "conscientiousness_work_family.json"
FILLER_PATH = ROOT / "data" / "conditional" / "history_filler.json"

PERSONA_STRING = (
    "You are a person who is highly conscientious — organized, disciplined, "
    "deliberate, planful, and methodical. You pay close attention to detail, "
    "follow through on commitments, work from structured plans, double-check "
    "your work, and prefer order to spontaneity. This is a stable feature of "
    "your character that shows up across all situations and conversations."
)

TRAIT_LABEL_STRING = "You are someone who is highly conscientious."

JUDGE_PROMPT = """You will read a question and a person's response.

Rate, on a 1-5 scale, how much the response expresses CONSCIENTIOUSNESS — \
that is: organization, planning, follow-through, attention to detail, \
discipline, deliberateness, and orderly thinking.

Use the full scale:
  1 = strongly low conscientiousness (impulsive, disorganized, casual, "wing it")
  2 = somewhat low
  3 = neutral / mixed
  4 = somewhat high
  5 = strongly high conscientiousness (planful, systematic, methodical, detail-oriented)

Do NOT rate the response's quality or politeness — rate ONLY the conscientiousness it expresses.
Reply with the rating in the EXACT format: "Rating: <integer 1-5>" on a single line, then optionally one short justification sentence.

Question: {question}

Response: {answer}"""


def extract_rating(raw: str) -> int | None:
    if not raw:
        return None
    m = re.search(r"[Rr]ating\s*[:=]\s*([1-5])", raw)
    if m:
        return int(m.group(1))
    m = re.search(r"\b([1-5])\b", raw[:80])
    return int(m.group(1)) if m else None


def judge_response(judge_client, judge_model, is_llm_client, q, a, max_retries=6):
    prompt = JUDGE_PROMPT.format(question=q, answer=a)
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            if is_llm_client:
                raw = judge_client.chat(messages=[ChatMessage(role="user", content=prompt)])
            else:
                resp = judge_client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                raw = resp.choices[0].message.content
            return {"raw": raw, "rating": extract_rating(raw or ""), "error": None}
        except Exception as e:
            last_err = str(e).lower()
            if attempt < max_retries and any(s in last_err for s in ("429", "rate limit", "timeout", "503", "502", "overloaded")):
                time.sleep(min(2 ** attempt, 30) + random.uniform(0, 1.0))
                continue
            return {"raw": None, "rating": None, "error": str(e)}
    return {"raw": None, "rating": None, "error": last_err}


def build_messages(history_pairs, probe):
    msgs = []
    for pair in history_pairs:
        msgs.append(ChatMessage(role="user", content=pair["user"]))
        msgs.append(ChatMessage(role="assistant", content=pair["assistant"]))
    msgs.append(ChatMessage(role="user", content=probe))
    return msgs


def gen_caps(corpus, llm, history_pairs, probe, top_k=4):
    """Manually replicate CapsAgent but with conversation history.

    Retrieve based ONLY on the current probe (the situation), then inject
    those atoms into a fresh system prompt and call the LLM with the full
    history + probe as the message stream. This is the natural extension
    of the per-turn retrieval pattern to a multi-turn conversation.
    """
    from persona_caps.retriever import Retriever
    retr = Retriever(corpus)
    fired = retr.retrieve(probe, k=top_k)
    rendered = "\n".join(f"- {atom.render()}" for atom, _ in fired)
    system = CAPS_SYSTEM_PROMPT.format(contingencies=rendered)
    msgs = build_messages(history_pairs, probe)
    text = llm.chat(messages=msgs, system=system)
    return text.strip(), [
        {"atom_id": a.id, "render": a.render(), "similarity": round(float(s), 4)}
        for a, s in fired
    ]


def gen_string(system_prompt, llm, history_pairs, probe):
    msgs = build_messages(history_pairs, probe)
    text = llm.chat(messages=msgs, system=system_prompt)
    return text.strip(), []


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gen-backend", default="gemini-flash")
    p.add_argument("--judge-backend", default="openai", choices=["openai", "minimax", "gemini-flash"])
    p.add_argument("--judge-model", default="gpt-4o-2024-08-06")
    p.add_argument("--history-lengths", type=int, nargs="+", default=[0, 5, 20, 50])
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--gen-temperature", type=float, default=0.7)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260424)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    scenarios = json.loads(SCENARIOS_PATH.read_text())["scenarios"]
    probes = [s["work"] for s in scenarios]
    print(f"[lc] {len(probes)} work-context probes (high-C target)")

    filler = json.loads(FILLER_PATH.read_text())["turns"]
    if max(args.history_lengths) > len(filler):
        sys.exit(f"Need {max(args.history_lengths)} filler turns but only {len(filler)} available")

    rng = random.Random(args.seed)
    history_per_run: list[list[dict]] = []
    for r in range(args.runs):
        ordering = list(range(len(filler)))
        rng.shuffle(ordering)
        history_per_run.append([filler[i] for i in ordering])

    gen_llm = get_client(args.gen_backend, temperature=args.gen_temperature)
    corpus_high_c = IfThenCorpus.from_jsonl(HIGH_C_PATH)
    print(f"[lc] high-C corpus: {len(corpus_high_c.atoms)} atoms (paper-track Beck v2)")

    tasks = []
    for run_id in range(args.runs):
        full_history = history_per_run[run_id]
        for hlen in args.history_lengths:
            history_pairs = full_history[:hlen]
            for probe_idx, probe in enumerate(probes):
                for method in ("caps", "persona_string", "trait_label"):
                    tasks.append({
                        "run_id": run_id,
                        "history_len": hlen,
                        "probe_idx": probe_idx,
                        "probe": probe,
                        "method": method,
                        "history_pairs": history_pairs,
                    })
    print(f"[lc] total generations queued: {len(tasks)}")

    def worker(t):
        s = time.time()
        try:
            if t["method"] == "caps":
                text, fired = gen_caps(corpus_high_c, gen_llm, t["history_pairs"], t["probe"])
            elif t["method"] == "persona_string":
                text, fired = gen_string(PERSONA_STRING, gen_llm, t["history_pairs"], t["probe"])
            else:
                text, fired = gen_string(TRAIT_LABEL_STRING, gen_llm, t["history_pairs"], t["probe"])
            err = None
        except Exception as e:
            text, fired, err = "", [], str(e)
        return {
            "run_id": t["run_id"],
            "history_len": t["history_len"],
            "probe_idx": t["probe_idx"],
            "probe": t["probe"],
            "method": t["method"],
            "response": text,
            "fired_atoms": fired,
            "gen_error": err,
            "gen_seconds": round(time.time() - s, 2),
        }

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(worker, t) for t in tasks]
        for i, f in enumerate(as_completed(futs)):
            rows.append(f.result())
            if (i + 1) % 16 == 0 or i + 1 == len(futs):
                print(f"[lc] gen {i+1}/{len(futs)}")

    rows.sort(key=lambda r: (r["run_id"], r["history_len"], r["method"], r["probe_idx"]))

    if args.judge_backend == "openai":
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            sys.exit("OPENAI_API_KEY required")
        judge_client = OpenAI(api_key=api_key, max_retries=0, timeout=60.0)
        is_llm_client = False
        judge_model = args.judge_model
    else:
        judge_client = get_client(args.judge_backend, temperature=0.0)
        is_llm_client = True
        judge_model = "(LLMClient)"

    print(f"[lc] judging {len(rows)} responses with {args.judge_backend} ...")
    for i, r in enumerate(rows):
        if r.get("gen_error") or not r["response"]:
            r["rating"] = None
            r["judge_raw"] = None
            r["judge_error"] = "no response"
            continue
        j = judge_response(judge_client, judge_model, is_llm_client, r["probe"], r["response"])
        r["rating"] = j["rating"]
        r["judge_raw"] = j["raw"]
        r["judge_error"] = j["error"]
        if (i + 1) % 16 == 0 or i + 1 == len(rows):
            print(f"[lc] judged {i+1}/{len(rows)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": vars(args) | {"out": str(args.out)},
        "rows": rows,
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[lc] wrote -> {args.out}")

    print("\n=== Mean conscientiousness rating per (history_len, method) ===")
    print(f"{'history_len':>11} | " + " | ".join(f"{m:>14}" for m in ("caps", "persona_string", "trait_label")))
    print("-" * 70)
    for hlen in args.history_lengths:
        line = [f"{hlen:>11}"]
        for method in ("caps", "persona_string", "trait_label"):
            vals = [r["rating"] for r in rows if r["history_len"] == hlen and r["method"] == method and r["rating"] is not None]
            line.append(f"{mean(vals):>14.2f}" if vals else f"{'-':>14}")
        print(" | ".join(line))

    print("\n=== Drift (history_len 50 - history_len 0) per method ===")
    h_max = max(args.history_lengths)
    for method in ("caps", "persona_string", "trait_label"):
        v0 = [r["rating"] for r in rows if r["history_len"] == 0 and r["method"] == method and r["rating"] is not None]
        vM = [r["rating"] for r in rows if r["history_len"] == h_max and r["method"] == method and r["rating"] is not None]
        if v0 and vM:
            d = mean(vM) - mean(v0)
            print(f"  {method:<16}  history=0 -> {mean(v0):.2f}, history={h_max} -> {mean(vM):.2f}, drift = {d:+.2f}")


if __name__ == "__main__":
    main()
