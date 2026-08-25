"""Independent-parser robustness check for the Mini-IPIP scoring pipeline.

The headline §4.6 numbers parse the subject LLM's free-text Likert response
with a deterministic regex (`RATING:\\s*(-?\\d+)` in src/persona_caps/eval.py).
A reviewer concern: does that single-pass regex hide systematic mis-parses
that bias the headline target spread?

To bound this, we:
  (1) re-administer the Mini-IPIP on the headline backend (MiniMax-M2.7-highspeed)
      under the load-bearing CAPS method (Beck v2 corpus), N=1 per cell, 5 dims x 2 pol;
  (2) save the raw response text per item;
  (3) for each raw response, ask GPT-4o-2024-08-06 ($T=0$) to extract the
      Likert rating, with a deliberately different prompt template;
  (4) report per-item exact agreement, mean absolute disagreement, and
      cell-mean target-spread under each parser.

If the two parsers agree at the per-item level >= 95% and the cell-mean
target spreads agree to within +/- 0.1 inventory points, the headline
spread is parser-robust.

Output: runs/parser_check/parser_check.json (raw + summary).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from persona_caps.agent import CapsAgent  # noqa: E402
from persona_caps.corpus import load_corpus_paper_only  # noqa: E402
from persona_caps.eval import Inventory, administer  # noqa: E402
from persona_caps.llm import ChatMessage, LLMClient  # noqa: E402
from persona_caps.retriever import Retriever  # noqa: E402

DIMS = ["neuroticism", "extraversion", "conscientiousness", "agreeableness", "openness"]
POLS = ["high", "low"]

CORPUS_DIR = ROOT / "data/if_then/paper/beck_2022_v2"
INV_PATH = ROOT / "data/inventories/ipip_short.json"
OUT_DIR = ROOT / "runs/parser_check"

GPT_PARSER_PROMPT = """You are reading a respondent's answer to a personality-questionnaire item.
The item asked them to rate, on a 1 (strongly disagree) to 5 (strongly agree) scale,
how well a statement describes them.

The respondent was instructed to reply in the format:
  RATING: <integer 1-5>
  WHY: <one sentence>

But they may not have followed the format perfectly.

Your task: extract the integer Likert rating (1-5) the respondent intended.
- If a clear RATING: line is present, use that integer.
- If the format was violated, infer from the prose.
- If you cannot infer a single integer 1-5 with confidence, reply UNPARSEABLE.

Reply with ONLY the integer 1-5, or the literal token UNPARSEABLE. No prose.

Respondent's answer:
\"\"\"
{raw}
\"\"\"
"""

REGEX = re.compile(r"RATING\s*:\s*(-?\d+)", re.IGNORECASE)


def regex_parse(raw: str) -> int | None:
    m = REGEX.search(raw)
    if not m:
        return None
    v = int(m.group(1))
    return v if 1 <= v <= 5 else None


def gpt4o_parse(raw: str, parser_llm: LLMClient) -> int | None:
    out = parser_llm.chat(
        [ChatMessage(role="user", content=GPT_PARSER_PROMPT.format(raw=raw))]
    ).strip()
    if out.upper().startswith("UNPARSEABLE"):
        return None
    m = re.match(r"\s*([1-5])\b", out)
    return int(m.group(1)) if m else None


def cell_mean_target(report, dim: str) -> float | None:
    return report.dimension_scores.get(dim)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    inventory = Inventory.load(INV_PATH)
    subj = LLMClient("minimax", os.getenv("MINIMAX_MODEL", "MiniMax-M2.7-highspeed"), temperature=0.7)
    parser_llm = LLMClient("openai", "gpt-4o-2024-08-06", temperature=0.0)

    rows: list[dict] = []
    summary_cells: list[dict] = []

    for dim in DIMS:
        for pol in POLS:
            cell_t0 = time.time()
            corpus = load_corpus_paper_only(CORPUS_DIR / f"{pol}_{dim}.jsonl")
            retriever = Retriever(corpus)
            agent = CapsAgent(corpus=corpus, llm=subj, retriever=retriever, top_k=4)
            report = administer(agent, inventory, workers=8)
            cell_dt = time.time() - cell_t0

            # for each per-item raw_text, ask GPT-4o to re-parse
            def reparse(item_resp):
                gpt_rating = gpt4o_parse(item_resp.raw_text, parser_llm)
                return {
                    "item_id": item_resp.item.id,
                    "dim": item_resp.item.dimension,
                    "keyed": item_resp.item.keyed,
                    "raw_text": item_resp.raw_text,
                    "regex_rating": item_resp.rating,
                    "gpt4o_rating": gpt_rating,
                }

            with ThreadPoolExecutor(max_workers=8) as ex:
                cell_rows = list(ex.map(reparse, report.responses))

            for r in cell_rows:
                r["cell_dim"] = dim
                r["cell_pol"] = pol
            rows.extend(cell_rows)

            # also recompute the dim score under each parser
            def to_scored(rating, keyed):
                if rating is None:
                    return None
                return float(6 - rating) if keyed == "-" else float(rating)

            regex_dim = [
                to_scored(r["regex_rating"], r["keyed"])
                for r in cell_rows
                if r["dim"] == dim
            ]
            gpt4o_dim = [
                to_scored(r["gpt4o_rating"], r["keyed"])
                for r in cell_rows
                if r["dim"] == dim
            ]
            regex_mean = (
                sum(v for v in regex_dim if v is not None)
                / max(1, sum(1 for v in regex_dim if v is not None))
            )
            gpt4o_mean = (
                sum(v for v in gpt4o_dim if v is not None)
                / max(1, sum(1 for v in gpt4o_dim if v is not None))
            )
            n_agree = sum(
                1
                for r in cell_rows
                if r["regex_rating"] is not None
                and r["gpt4o_rating"] is not None
                and r["regex_rating"] == r["gpt4o_rating"]
            )
            n_both = sum(
                1
                for r in cell_rows
                if r["regex_rating"] is not None and r["gpt4o_rating"] is not None
            )

            summary_cells.append(
                {
                    "dim": dim,
                    "pol": pol,
                    "n_items": len(cell_rows),
                    "n_target_dim_items": len(regex_dim),
                    "regex_target_mean": round(regex_mean, 3),
                    "gpt4o_target_mean": round(gpt4o_mean, 3),
                    "diff": round(gpt4o_mean - regex_mean, 3),
                    "per_item_exact_agreement": round(n_agree / max(1, n_both), 3),
                    "n_both_parsed": n_both,
                    "regex_failures": sum(1 for r in cell_rows if r["regex_rating"] is None),
                    "gpt4o_failures": sum(1 for r in cell_rows if r["gpt4o_rating"] is None),
                    "wall_s": round(cell_dt, 1),
                }
            )
            print(
                f"[{dim} {pol}] regex={regex_mean:.2f}  gpt4o={gpt4o_mean:.2f}  "
                f"agree={n_agree}/{n_both}  ({cell_dt:.1f}s)"
            )

    # per-dim target spread under each parser
    spread_table = []
    for dim in DIMS:
        h = next((c for c in summary_cells if c["dim"] == dim and c["pol"] == "high"), None)
        l = next((c for c in summary_cells if c["dim"] == dim and c["pol"] == "low"), None)
        if h and l:
            spread_table.append(
                {
                    "dim": dim,
                    "regex_spread": round(h["regex_target_mean"] - l["regex_target_mean"], 3),
                    "gpt4o_spread": round(h["gpt4o_target_mean"] - l["gpt4o_target_mean"], 3),
                    "spread_diff": round(
                        (h["gpt4o_target_mean"] - l["gpt4o_target_mean"])
                        - (h["regex_target_mean"] - l["regex_target_mean"]),
                        3,
                    ),
                }
            )

    overall_agreement = sum(c["per_item_exact_agreement"] * c["n_both_parsed"] for c in summary_cells) / max(
        1, sum(c["n_both_parsed"] for c in summary_cells)
    )

    out = {
        "config": {
            "method": "caps",
            "subject_backend": "minimax",
            "subject_model": os.getenv("MINIMAX_MODEL", "MiniMax-M2.7-highspeed"),
            "subject_temperature": 0.7,
            "parser_backend": "openai",
            "parser_model": "gpt-4o-2024-08-06",
            "parser_temperature": 0.0,
            "corpus_dir": str(CORPUS_DIR.relative_to(ROOT)),
            "inventory": str(INV_PATH.relative_to(ROOT)),
            "n_runs_per_cell": 1,
        },
        "summary_cells": summary_cells,
        "spread_table": spread_table,
        "overall_per_item_agreement": round(overall_agreement, 4),
        "n_total_items": len(rows),
    }

    with open(OUT_DIR / "parser_check_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    with open(OUT_DIR / "parser_check_per_item.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print()
    print(f"Overall per-item exact agreement: {overall_agreement:.3f}")
    print()
    print("Per-dim target spread:")
    for s in spread_table:
        print(
            f"  {s['dim']:20s}  regex={s['regex_spread']:+.2f}  gpt4o={s['gpt4o_spread']:+.2f}  "
            f"diff={s['spread_diff']:+.3f}"
        )
    print()
    print(f"Wrote {OUT_DIR}/parser_check_summary.json and parser_check_per_item.jsonl")


if __name__ == "__main__":
    main()
