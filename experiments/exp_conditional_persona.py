"""Killer demo — context-conditional personality.

Hypothesis. CAPS uniquely supports *intra-individual cross-situational
personality coherence*: the same agent, with the same retrieval+conditioning
code, can express HIGH conscientiousness in a "work" context and LOW
conscientiousness in a "family" context, simply by routing the situation to
a different on-disk corpus before retrieval. Persona-as-string and persona-
as-vector approaches cannot reliably encode "be high-C in context A, low-C
in context B" because they have only one global persona slot.

Test design.
* 8 paired scenarios (``data/conditional/conscientiousness_work_family.json``).
  Each scenario probes the same underlying behavior in two parallel framings:
  one work-flavored, one family-flavored.
* Four methods generate one response per (scenario, context):
    1. ``cond_caps``    : ``ConditionalCapsAgent`` with ``{"work": high_C, "family": low_C}``.
    2. ``string_cond``  : ``PersonaDescriptionAgent`` whose system prompt explicitly
       states the conditional rule ("highly conscientious at work, relaxed at home").
    3. ``single_high_caps``: ``CapsAgent`` with the high-C corpus only (cannot route).
    4. ``single_low_caps`` : ``CapsAgent`` with the low-C corpus only (cannot route).
* Each response is rated 1-5 on conscientiousness expression by an LLM judge
  (default GPT-4o, same protocol as §4.4). Scores per (method, context) are
  averaged across the 8 scenarios.
* Headline metric: WORK − FAMILY mean score per method. Conditional CAPS
  should produce a large positive gap; single-corpus CAPS should produce ~0;
  string-conditional should be in between (and noisier).

Usage:
    python experiments/exp_conditional_persona.py \
        --gen-backend gemini-flash --judge-backend openai \
        --out runs/conditional/run.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from persona_caps.agent import (  # noqa: E402
    CapsAgent,
    ConditionalCapsAgent,
    PersonaDescriptionAgent,
)
from persona_caps.corpus import IfThenCorpus  # noqa: E402
from persona_caps.llm import ChatMessage, get_client  # noqa: E402

DEFAULT_SCENARIOS = ROOT / "data" / "conditional" / "conscientiousness_work_family.json"
PAPER_CORPUS_DIR = ROOT / "data" / "if_then" / "paper" / "beck_2022_v2"

DEFAULT_JUDGE_PROMPT_CONSCIENTIOUSNESS = """You will read a question and a person's response.

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

DEFAULT_CONFIG_CONSCIENTIOUSNESS = {
    "trait": "conscientiousness",
    "contexts": ["work", "family"],
    "expected_polarity": {"work": "high", "family": "low"},
    "context_string_for_baseline": (
        "You are a person who is highly conscientious, organized, and disciplined "
        "in professional settings (at work, with colleagues, in your career), but "
        "relaxed, spontaneous, and easygoing at home with family (with siblings, "
        "parents, partners, children, in informal family settings). The same person, "
        "two clearly different modes depending on whether the situation is professional "
        "or familial."
    ),
    "judge_prompt": DEFAULT_JUDGE_PROMPT_CONSCIENTIOUSNESS,
}


def extract_rating(raw: str) -> int | None:
    if not raw:
        return None
    m = re.search(r"[Rr]ating\s*[:=]\s*([1-5])", raw)
    if m:
        return int(m.group(1))
    m = re.search(r"\b([1-5])\b", raw[:80])
    return int(m.group(1)) if m else None


def judge_response(judge_client, judge_model: str, is_llm_client: bool,
                   judge_prompt_template: str,
                   question: str, answer: str, max_retries: int = 6) -> dict:
    prompt = judge_prompt_template.format(question=question, answer=answer)
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
            last_err = str(e)
            msg = last_err.lower()
            transient = any(s in msg for s in ("429", "rate limit", "timeout", "503", "502", "overloaded"))
            if attempt < max_retries and transient:
                import random
                time.sleep(min(2 ** attempt, 30) + random.uniform(0, 1.0))
                continue
            break
    return {"raw": None, "rating": None, "error": last_err}


def build_agents(gen_llm, trait: str, contexts: list,
                 expected_polarity: dict, string_prompt: str) -> dict:
    """Build the four method agents for a given trait + paired-context spec.

    ``contexts`` is the ordered pair of context labels (e.g. ["work", "family"]
    or ["support", "conflict"]). ``expected_polarity`` maps each context label
    to either "high" or "low", indicating which pole of the trait corpus that
    context should route to.
    """
    if set(expected_polarity.values()) != {"high", "low"}:
        raise ValueError("expected_polarity must map contexts to exactly one 'high' and one 'low'")
    high_corpus = IfThenCorpus.from_jsonl(PAPER_CORPUS_DIR / f"high_{trait}.jsonl")
    low_corpus = IfThenCorpus.from_jsonl(PAPER_CORPUS_DIR / f"low_{trait}.jsonl")
    pole_corpus = {"high": high_corpus, "low": low_corpus}

    routing = {ctx: pole_corpus[expected_polarity[ctx]] for ctx in contexts}

    return {
        "cond_caps":         ConditionalCapsAgent(
            contexts=routing, llm=gen_llm, top_k=4,
        ),
        "string_cond":       PersonaDescriptionAgent(string_prompt, llm=gen_llm),
        "single_high_caps":  CapsAgent(corpus=high_corpus, llm=gen_llm, top_k=4),
        "single_low_caps":   CapsAgent(corpus=low_corpus, llm=gen_llm, top_k=4),
    }


def call_agent(method: str, agent, situation: str, context: str):
    if method == "cond_caps":
        resp = agent.respond(situation, context=context)
    else:
        resp = agent.respond(situation)
    fired = [
        {"atom_id": atom.id, "render": atom.render(), "similarity": round(float(sim), 4)}
        for atom, sim in resp.fired_atoms
    ]
    return resp.text, fired


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    p.add_argument("--gen-backend", default="gemini-flash",
                   help="LLM backend for response generation (preset name).")
    p.add_argument("--judge-backend", default="openai",
                   choices=["openai", "minimax", "gemini-flash", "gemini-pro"],
                   help="LLM-as-judge backend.")
    p.add_argument("--judge-model", default="gpt-4o-2024-08-06")
    p.add_argument("--gen-temperature", type=float, default=0.7)
    p.add_argument("--workers", type=int, default=2,
                   help="Parallel generation workers (judge runs serially with retry).")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--skip-judge", action="store_true",
                   help="Generate responses but skip judging (for inspection).")
    args = p.parse_args()

    raw = json.loads(args.scenarios.read_text())
    scenarios = raw["scenarios"]
    # Pull trait config from the scenarios file when present; fall back to
    # the hard-coded conscientiousness defaults so the original work/family
    # invocation still runs unchanged.
    trait = raw.get("trait", DEFAULT_CONFIG_CONSCIENTIOUSNESS["trait"])
    contexts = raw.get("contexts", DEFAULT_CONFIG_CONSCIENTIOUSNESS["contexts"])
    expected_polarity = raw.get("expected_polarity",
                                DEFAULT_CONFIG_CONSCIENTIOUSNESS["expected_polarity"])
    string_prompt = raw.get("context_string_for_baseline",
                            DEFAULT_CONFIG_CONSCIENTIOUSNESS["context_string_for_baseline"])
    judge_prompt_template = raw.get("judge_prompt",
                                    DEFAULT_CONFIG_CONSCIENTIOUSNESS["judge_prompt"])
    if len(contexts) != 2:
        sys.exit(f"expected exactly 2 context labels, got {contexts}")

    print(f"[cond] trait={trait}  contexts={contexts}  "
          f"expected_polarity={expected_polarity}")
    print(f"[cond] {len(scenarios)} scenarios × 2 contexts × 4 methods = {len(scenarios)*2*4} generations")

    gen_llm = get_client(args.gen_backend, temperature=args.gen_temperature)
    agents = build_agents(gen_llm, trait, contexts, expected_polarity, string_prompt)

    rows: list[dict] = []
    tasks = []
    for sc in scenarios:
        for ctx in contexts:
            situation = sc[ctx]
            for method, agent in agents.items():
                tasks.append((sc["id"], ctx, method, situation, agent))

    def worker(task):
        sc_id, ctx, method, situation, agent = task
        t = time.time()
        try:
            text, fired = call_agent(method, agent, situation, ctx)
            err = None
        except Exception as e:
            text, fired, err = "", [], str(e)
        return {
            "scenario_id": sc_id,
            "context": ctx,
            "method": method,
            "situation": situation,
            "response": text,
            "fired_atoms": fired,
            "gen_error": err,
            "gen_seconds": round(time.time() - t, 2),
        }

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(worker, t) for t in tasks]
        for i, f in enumerate(as_completed(futs)):
            rows.append(f.result())
            if (i + 1) % 8 == 0 or i + 1 == len(futs):
                print(f"[cond] gen {i+1}/{len(futs)}")

    rows.sort(key=lambda r: (r["scenario_id"], r["context"], r["method"]))

    if not args.skip_judge:
        if args.judge_backend == "openai":
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                sys.exit("OPENAI_API_KEY required for judge")
            judge_client = OpenAI(api_key=api_key, max_retries=0, timeout=60.0)
            is_llm_client = False
            judge_model = args.judge_model
        else:
            judge_client = get_client(args.judge_backend, temperature=0.0)
            is_llm_client = True
            judge_model = "(via LLMClient)"

        print(f"[cond] judging {len(rows)} responses with {args.judge_backend} / {judge_model} ...")
        for i, r in enumerate(rows):
            if r.get("gen_error") or not r["response"]:
                r["rating"] = None
                r["judge_raw"] = None
                r["judge_error"] = "no response generated"
                continue
            j = judge_response(judge_client, judge_model, is_llm_client,
                               judge_prompt_template, r["situation"], r["response"])
            r["rating"] = j["rating"]
            r["judge_raw"] = j["raw"]
            r["judge_error"] = j["error"]
            if (i + 1) % 8 == 0 or i + 1 == len(rows):
                print(f"[cond] judged {i+1}/{len(rows)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "trait": trait,
            "contexts": contexts,
            "expected_polarity": expected_polarity,
            "gen_backend": args.gen_backend,
            "judge_backend": args.judge_backend,
            "judge_model": args.judge_model if args.judge_backend == "openai" else "(LLMClient)",
            "gen_temperature": args.gen_temperature,
            "scenarios_file": str(args.scenarios),
            "n_scenarios": len(scenarios),
        },
        "rows": rows,
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[cond] wrote -> {args.out}")

    if not args.skip_judge:
        from statistics import mean
        # Order so that the "high"-pole context comes first in the printed table,
        # so the gap is always (high_ctx - low_ctx).
        ordered = sorted(contexts, key=lambda c: 0 if expected_polarity[c] == "high" else 1)
        ctx_high, ctx_low = ordered
        print(f"\n=== Per-(method, context) mean {trait} rating (1-5) ===")
        gap_label = f"{ctx_high}-{ctx_low}"
        print(f"{'method':<22} {ctx_high:>10} {ctx_low:>10} {gap_label:>14}  {'n_valid':>7}")
        for method in ("cond_caps", "string_cond", "single_high_caps", "single_low_caps"):
            line = [f"{method:<22}"]
            for ctx in ordered:
                vals = [r["rating"] for r in rows
                        if r["method"] == method and r["context"] == ctx and r["rating"] is not None]
                line.append(f"{mean(vals):>10.2f}" if vals else f"{'-':>10}")
            ws = [r["rating"] for r in rows
                  if r["method"] == method and r["context"] == ctx_high and r["rating"] is not None]
            fs = [r["rating"] for r in rows
                  if r["method"] == method and r["context"] == ctx_low and r["rating"] is not None]
            gap = mean(ws) - mean(fs) if (ws and fs) else float("nan")
            line.append(f"{gap:>+13.2f} ")
            n = sum(1 for r in rows if r["method"] == method and r["rating"] is not None)
            line.append(f"{n:>7d}")
            print(" ".join(line))


if __name__ == "__main__":
    main()
