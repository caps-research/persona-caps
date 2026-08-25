"""GPT-4o LLM-as-judge for PersonalityBench, replicating IRiS / NPTI protocol.

Reads runs/personalitybench/{method}/{Trait}_{pol}.jsonl (each row has
``question`` + ``answer``) and writes runs/personalitybench_scored/{method}/
{Trait}_{pol}.jsonl with an additional ``rating`` field (int 1-5).

We then aggregate per (method, trait) using the IRiS rule:
    pos_score = mean(rating | polarity=high)
    neg_score = mean(6 - rating | polarity=low)   # invert the negative pole
    domain_score = pos_score + neg_score          # max 10
    domain_var = var across questions in that domain (combined high+low)

Reproduces baselines/NPTI/NPTI/code/gpt4_score.py prompt verbatim.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# IRiS / NPTI exact factor list (NPTI/code/gpt4_score.py)
FACTORS = {
    "Openness": '"imagination","artistic","interests","emotionality","adventurousness","intellect" and liberalism"',
    "Conscientiousness": '"self-efficacy","orderliness","dutifulness","achievement-striving","self-discipline","cautiousness"',
    "Extraversion": '"friendliness","gregariousness","assertiveness","activity level","excitement-seeking" and "cheerfulness"',
    "Agreeableness": '"trust","morality","altruism","cooperation","modesty","sympathy"',
    "Neuroticism": '"anxiety","anger","depression","self-consciousness","immoderation","vulnerability"',
}

JUDGE_PROMPT = """[System] 
        Please act as an impartial judge and evaluate the personality trait of {BFI} exhibited in the response provided by an AI assistant to the user question displayed below.        Your evaluation should consider factors such as {factors} demonstrated in the response.        Begin your evaluation by providing a short explanation of no more than 30 words. Be as objective as possible. After providing your explanation, please rate the response on a scale of 1 to 5,where a higher score indicates a stronger reflection of the specified trait.
        Please follow this format for your rating: "[[rating]]", for example: "Rating: [[5]]".
        [Question]
        {question}
        [The Start of Assistant's Answer]
        {answer}
        [The End of Assistant's Answer]
        """

RATING_RE = re.compile(r"\[\[\s*(\d)(?:\s*/\s*5)?\s*\]\]")


def extract_rating(text: str) -> int | None:
    m = RATING_RE.search(text)
    if not m:
        return None
    val = int(m.group(1))
    if 1 <= val <= 5:
        return val
    return None


def judge_one(client, model: str, trait: str, question: str, answer: str, is_llm_client: bool, max_retries: int = 8) -> dict:
    """client may be openai.OpenAI or persona_caps.llm.LLMClient.

    On 429 / transient errors, retry with exponential backoff up to ``max_retries``.
    """
    prompt = JUDGE_PROMPT.format(
        BFI=trait, factors=FACTORS[trait], question=question, answer=answer
    )
    last_err: str | None = None
    for attempt in range(max_retries + 1):
        try:
            if is_llm_client:
                from persona_caps.llm import ChatMessage
                raw = client.chat(messages=[ChatMessage(role="user", content=prompt)])
            else:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                raw = resp.choices[0].message.content
            return {"raw": raw, "rating": extract_rating(raw or ""), "error": None}
        except Exception as e:
            last_err = str(e)
            msg = last_err.lower()
            transient = (
                "429" in msg
                or "rate limit" in msg
                or "timeout" in msg
                or "503" in msg
                or "502" in msg
                or "overloaded" in msg
            )
            if attempt < max_retries and transient:
                # exponential backoff with jitter: 2^attempt seconds, cap 60
                import random
                delay = min(2 ** attempt, 60) + random.uniform(0, 1.5)
                time.sleep(delay)
                continue
            return {"raw": None, "rating": None, "error": last_err}
    return {"raw": None, "rating": None, "error": last_err}


def score_file(
    src_path: Path, dst_path: Path, trait: str, client, model: str, workers: int, is_llm_client: bool
) -> tuple[int, int]:
    """Row-level resume:
    - Load src questions+answers (n rows).
    - If dst exists, load its rows and pre-fill out_rows where rating is not None.
    - Re-judge ONLY the rows whose dst rating is missing (None).
    - Re-write the full dst file at the end with merged results.
    """
    rows = []
    with open(src_path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    n = len(rows)
    out_rows: list[dict | None] = [None] * n

    # Pre-fill from existing dst file (row-level resume)
    if dst_path.exists():
        with open(dst_path) as f:
            existing = [json.loads(l) for l in f if l.strip()]
        # Match by index when lengths agree (questions are deterministic in order)
        if len(existing) == n:
            for i, r in enumerate(existing):
                if r.get("rating") is not None:
                    # Carry forward already-good rating
                    out_rows[i] = {
                        "question": rows[i]["question"],
                        "answer": rows[i]["answer"],
                        "rating": r["rating"],
                        "judge_raw": r.get("judge_raw"),
                        "judge_error": None,
                    }

    pending = [i for i in range(n) if out_rows[i] is None]
    if not pending:
        print(f"  [skip] {dst_path.name}: already fully scored", flush=True)
        return n, n

    def one(i: int, row: dict) -> tuple[int, dict]:
        result = judge_one(client, model, trait, row["question"], row["answer"], is_llm_client)
        return i, {
            "question": row["question"],
            "answer": row["answer"],
            "rating": result["rating"],
            "judge_raw": result["raw"],
            "judge_error": result["error"],
        }

    t = time.time()
    n_pending = len(pending)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, i, rows[i]) for i in pending]
        for f in as_completed(futs):
            i, r = f.result()
            out_rows[i] = r

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_ok = sum(1 for r in out_rows if r and r.get("rating") is not None)
    print(
        f"  [done] {dst_path.name}: {n_ok}/{n} valid (re-judged {n_pending}) in {time.time()-t:.1f}s",
        flush=True,
    )
    return n_ok, n


def aggregate(scored_dir: Path) -> list[dict]:
    """Summary in IRiS Table 1 format: per-method × per-trait mean + variance."""
    summary = []
    for method_dir in sorted(scored_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        per_trait_means: dict[str, float] = {}
        per_trait_vars: dict[str, float] = {}
        all_combined: list[float] = []
        for trait in ["Agreeableness", "Conscientiousness", "Extraversion", "Neuroticism", "Openness"]:
            scores = []
            for pol in ["high", "low"]:
                fp = method_dir / f"{trait}_{pol}.jsonl"
                if not fp.exists():
                    continue
                with open(fp) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        r = row.get("rating")
                        if r is None:
                            continue
                        if pol == "high":
                            scores.append(float(r))
                        else:
                            scores.append(6.0 - float(r))
            if scores:
                # IRiS reports SUM(positive_mean + negative_mean) per trait, but
                # we report the mean of all rating-aligned scores * 2 to match scale
                # — simplest: report mean(rating_aligned) which is on 1-5, then *2 for max-10 comparability.
                mean_aligned = statistics.mean(scores)
                per_trait_means[trait] = mean_aligned * 2.0  # scale to /10 like IRiS
                per_trait_vars[trait] = (
                    statistics.variance(scores) if len(scores) > 1 else 0.0
                )
                all_combined.extend(scores)
        avg_mean = statistics.mean(per_trait_means.values()) if per_trait_means else float("nan")
        avg_var = statistics.mean(per_trait_vars.values()) if per_trait_vars else float("nan")
        summary.append(
            {
                "method": method,
                "per_trait_mean_x2": per_trait_means,
                "per_trait_variance": per_trait_vars,
                "average_mean_x2": avg_mean,
                "average_variance": avg_var,
                "n_total": len(all_combined),
            }
        )
    return summary


def print_table(summary: list[dict]) -> None:
    print()
    print("=" * 88)
    print(f"{'Method':<10} {'A':>10} {'C':>10} {'E':>10} {'N':>10} {'O':>10} {'Avg':>10}")
    print("-" * 88)
    for s in summary:
        m = s["per_trait_mean_x2"]
        row = f"{s['method']:<10}"
        for t in ["Agreeableness", "Conscientiousness", "Extraversion", "Neuroticism", "Openness"]:
            row += f" {m.get(t, 0):>10.2f}"
        row += f" {s['average_mean_x2']:>10.2f}"
        print(row)
    print()
    print("(All numbers scaled to /10 for IRiS Table 1 comparability)")
    print("IRiS Table 1 (Llama-3-8B): SP=8.70  P²=9.10  ActAdd=7.39  NPTI=9.09  IRiS=9.26  SFT=9.27")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", default="runs/personalitybench", help="Per-method generations")
    p.add_argument("--out-dir", default="runs/personalitybench_scored")
    p.add_argument("--judge-backend", default="openai", choices=["openai", "minimax", "anthropic"])
    p.add_argument("--model", default="gpt-4o-2024-08-06")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--summary-only", action="store_true", help="Skip judging; just aggregate")
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.summary_only:
        if args.judge_backend == "openai":
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                sys.exit("OPENAI_API_KEY env var required")
            # max_retries=0: we handle retry+backoff manually in judge_one to log errors;
            # SDK's silent retry would mask 429 timing data we want for paper.
            client = OpenAI(api_key=api_key, max_retries=0, timeout=60.0)
        else:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
            from persona_caps.llm import LLMClient
            model = args.model if args.model != "gpt-4o-2024-08-06" else "MiniMax-M2.7-highspeed"
            client = LLMClient(backend=args.judge_backend, model=model)
            args.model = model

        method_dirs = []
        for d in sorted(in_dir.iterdir()):
            if not d.is_dir():
                continue
            if args.methods and d.name not in args.methods:
                continue
            method_dirs.append(d)

        for md in method_dirs:
            print(f"=== Judging method: {md.name} ===")
            for fp in sorted(md.glob("*.jsonl")):
                stem = fp.stem  # e.g. Extraversion_high
                trait = stem.split("_")[0]
                if trait not in FACTORS:
                    print(f"  [skip] unknown trait: {fp.name}")
                    continue
                dst = out_dir / md.name / fp.name
                score_file(fp, dst, trait, client, args.model, args.workers, args.judge_backend != "openai")

    summary = aggregate(out_dir)
    print_table(summary)
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
