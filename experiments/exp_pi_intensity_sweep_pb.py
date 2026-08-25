#!/usr/bin/env python3
"""CAPS supplemental: π-intensity sweep on PersonalityBench Extraversion questions.

Not a standard continuous-personality benchmark — PB questions are reused as
situations; π controls high/low atom quotas (intensity), then MiniMax judges
Extraversion. See docs/pi-intensity-sweep-personalitybench-q-20260719.md.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from persona_caps.agent import CAPS_SYSTEM_PROMPT  # noqa: E402
from persona_caps.corpus import load_corpus_paper_only  # noqa: E402
from persona_caps.llm import ChatMessage, LLMClient  # noqa: E402
from persona_caps.retriever import Retriever  # noqa: E402

from judge_personalitybench import JUDGE_PROMPT, FACTORS, extract_rating  # noqa: E402

PI_GRID = [0.05, 0.3, 0.5, 0.7, 0.95]
HARD_LEAK = [
    re.compile(r"\b(low|high|moderate)\s+extraversion\b", re.I),
    re.compile(r"\bpi(?:_vector)?\s*[:=]\s*|\bπ\s*[:=]", re.I),
    re.compile(
        r"Imagine you are (an?|a)\s+(extraverted|introverted)\b",
        re.I,
    ),
    re.compile(r"Social engagement intensity", re.I),
]


def proportional_count(pi: float, slots: int) -> int:
    return int(np.floor(float(pi) * slots + 0.5))


def load_questions(path: Path) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line)["question"])
    return out


def intensity_retrieve(
    question: str,
    *,
    pi: float,
    top_k: int,
    high_ret: Retriever,
    low_ret: Retriever,
) -> tuple[list, int, int]:
    k_high = proportional_count(pi, top_k)
    k_low = top_k - k_high
    high = high_ret.retrieve(question, k=k_high) if k_high else []
    low = low_ret.retrieve(question, k=k_low) if k_low else []
    fired = list(high) + list(low)
    # stable: high first then low (matches CAPSWorld intensity concat)
    return fired, k_high, k_low


def spearman(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3 or len(set(ys)) < 2:
        return None

    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = math.sqrt(
        sum((rx[i] - mx) ** 2 for i in range(n))
        * sum((ry[i] - my) ** 2 for i in range(n))
    )
    return num / den if den else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--questions",
        default="baselines/NPTI/NPTI/dataset/test/Extraversion.json",
    )
    p.add_argument(
        "--corpus-dir",
        default="data/if_then/paper/beck_2022_v2",
    )
    p.add_argument("--out-dir", default="runs/pi_intensity_sweep_pb_e_20260719")
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--limit", type=int, default=30, help="Questions to use (supplemental)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--backend", default="minimax")
    p.add_argument("--model", default="MiniMax-M2.7-highspeed")
    p.add_argument("--pi-grid", nargs="+", type=float, default=PI_GRID)
    p.add_argument("--gen-only", action="store_true")
    p.add_argument("--judge-only", action="store_true")
    p.add_argument("--analyze-only", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_path = out_dir / "generations.jsonl"
    judged_path = out_dir / "judged.jsonl"
    summary_path = out_dir / "summary.json"

    questions = load_questions(Path(args.questions))
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]

    corpus_dir = Path(args.corpus_dir)
    high_c = load_corpus_paper_only(corpus_dir / "high_extraversion.jsonl")
    low_c = load_corpus_paper_only(corpus_dir / "low_extraversion.jsonl")
    high_ret = Retriever(high_c)
    low_ret = Retriever(low_c, embed_fn=high_ret.embed_fn)

    llm = LLMClient(backend=args.backend, model=args.model)

    if not args.judge_only and not args.analyze_only:
        jobs: list[tuple[int, str, float]] = []
        for qi, q in enumerate(questions):
            for pi in args.pi_grid:
                jobs.append((qi, q, float(pi)))

        print(
            f"generate n_questions={len(questions)} pi_grid={args.pi_grid} "
            f"n_jobs={len(jobs)} top_k={args.top_k}",
            flush=True,
        )

        def one(job: tuple[int, str, float]) -> dict:
            qi, q, pi = job
            fired, k_high, k_low = intensity_retrieve(
                q, pi=pi, top_k=args.top_k, high_ret=high_ret, low_ret=low_ret
            )
            rendered = "\n".join(f"- {atom.render()}" for atom, _ in fired)
            system = CAPS_SYSTEM_PROMPT.format(contingencies=rendered)
            leak_hits = [name for name, pat in [
                ("low_high_extraversion", HARD_LEAK[0]),
                ("pi_assignment", HARD_LEAK[1]),
                ("imagine_trait", HARD_LEAK[2]),
                ("engagement_dial", HARD_LEAK[3]),
            ] if pat.search(system)]
            text = llm.chat(
                messages=[ChatMessage(role="user", content=q)],
                system=system,
            )
            return {
                "question_index": qi,
                "question": q,
                "pi": pi,
                "k_high": k_high,
                "k_low": k_low,
                "expected_k_high": proportional_count(pi, args.top_k),
                "fired_atom_ids": [a.id for a, _ in fired],
                "fired_polarities": [
                    "high" if "high_" in a.id else "low" for a, _ in fired
                ],
                "answer": text.strip(),
                "system_prompt": system,
                "leak_hits": leak_hits,
            }

        rows: list[dict] = []
        with gen_path.open("w", encoding="utf-8") as handle:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futs = [pool.submit(one, j) for j in jobs]
                for i, fut in enumerate(as_completed(futs), start=1):
                    row = fut.result()
                    rows.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    if i % 10 == 0 or i == len(jobs):
                        print(
                            f"  gen [{i}/{len(jobs)}] q={row['question_index']} "
                            f"π={row['pi']} k=({row['k_high']},{row['k_low']})",
                            flush=True,
                        )

        mix_ok = sum(
            1
            for r in rows
            if r["k_high"] == r["expected_k_high"] and r["k_low"] == args.top_k - r["k_high"]
        )
        leak_n = sum(1 for r in rows if r["leak_hits"])
        meta = {
            "n_generations": len(rows),
            "n_questions": len(questions),
            "pi_grid": args.pi_grid,
            "top_k": args.top_k,
            "mix_structure_ok": mix_ok,
            "mix_structure_rate": mix_ok / len(rows) if rows else 0,
            "leak_n": leak_n,
            "leak_status": "PASS" if leak_n == 0 else "FAIL",
            "note": "CAPS supplemental π-intensity sweep; not a continuous-personality benchmark",
        }
        (out_dir / "gen_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps(meta, indent=2), flush=True)
        if leak_n:
            raise SystemExit("leakage detected in CAPS system prompts")

    if args.gen_only:
        return

    # Judge
    if not args.analyze_only:
        gens = [
            json.loads(l)
            for l in gen_path.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        print(f"judge n={len(gens)} model={args.model}", flush=True)
        judge_llm = LLMClient(backend=args.backend, model=args.model)

        def judge_row(row: dict) -> dict:
            prompt = JUDGE_PROMPT.format(
                BFI="Extraversion",
                factors=FACTORS["Extraversion"],
                question=row["question"],
                answer=row["answer"],
            )
            raw = judge_llm.chat(messages=[ChatMessage(role="user", content=prompt)])
            rating = extract_rating(raw)
            return {
                **{k: row[k] for k in (
                    "question_index", "question", "pi", "k_high", "k_low",
                    "fired_atom_ids", "answer",
                )},
                "rating": rating,
                "judge_raw": raw,
                "judge_error": None if rating is not None else "parse_fail",
            }

        judged: list[dict] = []
        with judged_path.open("w", encoding="utf-8") as handle:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futs = [pool.submit(judge_row, r) for r in gens]
                for i, fut in enumerate(as_completed(futs), start=1):
                    row = fut.result()
                    judged.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    if i % 10 == 0 or i == len(gens):
                        print(
                            f"  judge [{i}/{len(gens)}] π={row['pi']} "
                            f"rating={row.get('rating')}",
                            flush=True,
                        )

    # Analyze
    judged = [
        json.loads(l)
        for l in judged_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    ok = [r for r in judged if r.get("rating") is not None]
    by_pi: dict[float, list[float]] = {float(pi): [] for pi in args.pi_grid}
    for r in ok:
        by_pi[float(r["pi"])].append(float(r["rating"]))
    mean_by_pi = {
        str(pi): (sum(vs) / len(vs) if vs else None)
        for pi, vs in sorted(by_pi.items())
    }
    # within-question Spearman
    from collections import defaultdict

    by_q: dict[int, list[dict]] = defaultdict(list)
    for r in ok:
        by_q[int(r["question_index"])].append(r)
    rhos = []
    deltas = []
    n_pos = 0
    for items in by_q.values():
        items = sorted(items, key=lambda x: float(x["pi"]))
        xs = [float(x["pi"]) for x in items]
        ys = [float(x["rating"]) for x in items]
        rho = spearman(xs, ys)
        if rho is not None:
            rhos.append(rho)
        lo = [y for x, y in zip(xs, ys) if x <= 0.1]
        hi = [y for x, y in zip(xs, ys) if x >= 0.9]
        if lo and hi:
            d = sum(hi) / len(hi) - sum(lo) / len(lo)
            deltas.append(d)
            if d > 0:
                n_pos += 1

    gen_meta = {}
    meta_path = out_dir / "gen_meta.json"
    if meta_path.exists():
        gen_meta = json.loads(meta_path.read_text())

    summary = {
        "protocol": "CAPS supplemental π-intensity sweep (not continuous-personality benchmark)",
        "n_judged": len(judged),
        "n_rated": len(ok),
        "mean_rating_by_pi": mean_by_pi,
        "mean_endpoint_delta": sum(deltas) / len(deltas) if deltas else None,
        "mean_within_question_spearman": sum(rhos) / len(rhos) if rhos else None,
        "n_questions_with_rho": len(rhos),
        "n_positive_endpoint": n_pos,
        "n_with_endpoint": len(deltas),
        "gen_meta": gen_meta,
        "leak_status": gen_meta.get("leak_status"),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)

    report = out_dir / "REPORT.md"
    report.write_text(
        "\n".join(
            [
                "# π-intensity sweep on PB Extraversion questions (CAPS supplemental)",
                "",
                f"- questions: {gen_meta.get('n_questions', '?')} × π grid {args.pi_grid}",
                f"- mix structure ok: {gen_meta.get('mix_structure_ok')}/{gen_meta.get('n_generations')}",
                f"- leakage: **{gen_meta.get('leak_status')}**",
                f"- mean rating by π: {mean_by_pi}",
                f"- endpoint Δ: {summary['mean_endpoint_delta']}",
                f"- mean within-q ρ: {summary['mean_within_question_spearman']}",
                f"- positive endpoint: {n_pos}/{len(deltas)}",
                "",
                "Not a continuous-personality benchmark; PB items reused as situations.",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
