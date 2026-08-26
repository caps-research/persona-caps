"""App. C.1-C.3 — behavior as a conditional probability given the situation.

Three sweeps are bundled together so they can be invoked from one CLI:

  --sweep A   Single-cell tau sweep:
              for each (dim, polarity) cell, run CapsAgent with
              mode in {"argmax", "softmax"} x tau in {0.05, 0.2, 1.0}.
              Tests whether the headline §4.6 numbers are robust when the
              MAP / argmax retrieval is replaced by genuine softmax sampling
              from P(atom | query); recovers an interpolation curve between
              argmax (tau -> 0) and the random_demos ablation (tau -> inf).

  --sweep B   Joint Big Five mixture (uniform):
              build a JointCapsAgent over a target Big Five profile
              (one polarity per dim, all 5 dims active, uniform weights),
              run the full Mini-IPIP, and check whether per-dim spreads
              come out in the predicted direction. Compares argmax and
              softmax sampling per cell.

  --sweep C   Single / dual-trait personas:
              same JointCapsAgent machinery but with only 1 or 2 cells
              active. Tests whether per-dim conditional retrieval is fair
              to single-trait personas (i.e. degrades gracefully back to
              the §4.6 single-cell baseline) and that dual-trait personas
              compose without one dim eating the other.

Default backend: MiniMax-M2.7-highspeed (matches the §4.1 / §4.4 headline).
Default N=3 runs per cell; pass --n-runs to override.

Outputs:
    runs/conditional_retrieval/<sweep>/<sub-id>.json  (per-condition raw report)
    runs/conditional_retrieval/<sweep>/summary.json   (aggregated table)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from persona_caps.agent import CapsAgent, JointCapsAgent  # noqa: E402
from persona_caps.corpus import load_corpus_paper_only  # noqa: E402
from persona_caps.eval import Inventory, administer  # noqa: E402
from persona_caps.llm import LLMClient  # noqa: E402
from persona_caps.retriever import Retriever, RetrievalMode  # noqa: E402

DIMS = ["neuroticism", "extraversion", "conscientiousness", "agreeableness", "openness"]
POLS = ["high", "low"]

CORPUS_DIR = ROOT / os.getenv(
    "PERSONA_CAPS_CORPUS_DIR",
    "data/if_then/paper/beck_2022_v2",
)
INV_PATH = ROOT / "data/inventories/ipip_short.json"
OUT_DIR_BASE = ROOT / os.getenv(
    "PERSONA_CAPS_OUT_DIR",
    "runs/conditional_retrieval",
)


def cell_path(dim: str, pol: str) -> Path:
    return CORPUS_DIR / f"{pol}_{dim}.jsonl"


def make_subject(args: argparse.Namespace) -> LLMClient:
    return LLMClient(
        args.subject_backend,
        args.subject_model,
        temperature=args.subject_temp,
    )


def target_score_for_cell(report, dim: str, pol: str) -> float | None:
    """Cell-level target-dimension score: agent's mean score on inventory items
    keyed to ``dim``. Independent of polarity (we look at the score under
    *this* persona; spread is computed across persona polarities later)."""
    return report.dimension_scores.get(dim)


def serialise_report(report) -> dict:
    return {
        "inventory_name": report.inventory_name,
        "parse_failures": report.parse_failures,
        "dimension_scores": dict(report.dimension_scores),
        "dimension_n": dict(report.dimension_n),
        "n_items": len(report.responses),
    }


# -----------------------------------------------------------------------------
# Sweep A: single-cell tau sweep
# -----------------------------------------------------------------------------

def run_sweep_a(args: argparse.Namespace) -> None:
    out_dir = OUT_DIR_BASE / "A_single_cell_tau"
    out_dir.mkdir(parents=True, exist_ok=True)

    subj = make_subject(args)
    inventory = Inventory.load(INV_PATH)

    summary_rows: list[dict] = []
    cells = [(d, p) for d in DIMS for p in POLS]

    for dim, pol in cells:
        corpus = load_corpus_paper_only(cell_path(dim, pol))
        for mode_label, mode, tau in [
            ("argmax", "argmax", 0.05),  # tau ignored
            ("softmax_t005", "softmax", 0.05),
            ("softmax_t02", "softmax", 0.2),
            ("softmax_t10", "softmax", 1.0),
        ]:
            for run_id in range(args.n_runs):
                t0 = time.time()
                seed = (
                    args.seed
                    + 1000 * run_id
                    + 100 * cells.index((dim, pol))
                    + ["argmax", "softmax_t005", "softmax_t02", "softmax_t10"].index(mode_label)
                )
                rng = np.random.default_rng(seed)
                retriever = Retriever(corpus, mode=mode, tau=tau, rng=rng)
                agent = CapsAgent(
                    corpus=corpus, llm=subj, retriever=retriever, top_k=args.top_k
                )
                report = administer(agent, inventory, workers=args.workers)
                dt = time.time() - t0

                row = {
                    "dim": dim,
                    "pol": pol,
                    "mode_label": mode_label,
                    "mode": mode,
                    "tau": tau,
                    "run_id": run_id,
                    "seed": seed,
                    "target_score_on_target_dim": target_score_for_cell(report, dim, pol),
                    "all_dim_scores": dict(report.dimension_scores),
                    "parse_failures": report.parse_failures,
                    "wall_s": round(dt, 1),
                }
                summary_rows.append(row)
                print(
                    f"[A {dim:18s} {pol:4s} {mode_label:13s} run={run_id}] "
                    f"target={row['target_score_on_target_dim']}  "
                    f"({dt:.1f}s)"
                )

    # aggregate: spread per (dim, mode_label) across high/low and N runs
    spread_table: list[dict] = []
    for dim in DIMS:
        for mode_label in ["argmax", "softmax_t005", "softmax_t02", "softmax_t10"]:
            high_vals = [
                r["target_score_on_target_dim"]
                for r in summary_rows
                if r["dim"] == dim and r["pol"] == "high" and r["mode_label"] == mode_label
                and r["target_score_on_target_dim"] is not None
            ]
            low_vals = [
                r["target_score_on_target_dim"]
                for r in summary_rows
                if r["dim"] == dim and r["pol"] == "low" and r["mode_label"] == mode_label
                and r["target_score_on_target_dim"] is not None
            ]
            if high_vals and low_vals:
                spread = float(np.mean(high_vals) - np.mean(low_vals))
                spread_table.append(
                    {
                        "dim": dim,
                        "mode_label": mode_label,
                        "n_high_runs": len(high_vals),
                        "n_low_runs": len(low_vals),
                        "high_mean": round(float(np.mean(high_vals)), 3),
                        "low_mean": round(float(np.mean(low_vals)), 3),
                        "spread": round(spread, 3),
                    }
                )

    out = {
        "config": {
            "sweep": "A_single_cell_tau",
            "subject_backend": args.subject_backend,
            "subject_model": args.subject_model,
            "subject_temperature": args.subject_temp,
            "top_k": args.top_k,
            "n_runs_per_cell": args.n_runs,
            "modes": ["argmax", "softmax_t005", "softmax_t02", "softmax_t10"],
            "corpus_dir": str(CORPUS_DIR.relative_to(ROOT)),
            "inventory": str(INV_PATH.relative_to(ROOT)),
        },
        "rows": summary_rows,
        "spread_table": spread_table,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(out, f, indent=2)

    print()
    print("=== Sweep A summary: per-dim target-spread by retrieval mode ===")
    print(f"{'dim':22s} {'argmax':>10s} {'sm_005':>10s} {'sm_02':>10s} {'sm_10':>10s}")
    for dim in DIMS:
        cells_for_dim = {r["mode_label"]: r["spread"] for r in spread_table if r["dim"] == dim}
        print(
            f"{dim:22s} "
            f"{cells_for_dim.get('argmax', float('nan')):>10.2f} "
            f"{cells_for_dim.get('softmax_t005', float('nan')):>10.2f} "
            f"{cells_for_dim.get('softmax_t02', float('nan')):>10.2f} "
            f"{cells_for_dim.get('softmax_t10', float('nan')):>10.2f}"
        )
    print(f"\nWrote {out_dir/'summary.json'}")


# -----------------------------------------------------------------------------
# Sweep B / C: Joint Big Five via JointCapsAgent
# -----------------------------------------------------------------------------

def _label_for_profile(profile: dict[str, str]) -> str:
    parts = []
    for d in DIMS:
        if d in profile:
            parts.append(f"{profile[d][0].upper()}{d[0].upper()}")  # e.g. "HN" / "LE"
    return "-".join(parts)


def _expected_direction(report, dim: str, pol: str) -> int:
    """+1 if the per-dim score moves in the direction predicted by ``pol``,
    -1 if the opposite, 0 if undefined.  We use the raw mean (no spread vs
    opposite-pole comparison) here because joint profiles are run as one
    administration each."""
    score = report.dimension_scores.get(dim)
    if score is None:
        return 0
    # neutral is the inventory mid-point (1..5 -> 3.0); above midpoint means
    # the agent leaned toward the keyed-positive end of the dim
    mid = 3.0
    if pol == "high":
        return +1 if score > mid + 0.25 else (-1 if score < mid - 0.25 else 0)
    return +1 if score < mid - 0.25 else (-1 if score > mid + 0.25 else 0)


def _build_joint_agent(
    profile: dict[str, str],
    subj: LLMClient,
    *,
    top_k_per_dim: int,
    mode: RetrievalMode,
    tau: float,
    seed: int,
) -> JointCapsAgent:
    cells = {f"{pol}_{dim}": load_corpus_paper_only(cell_path(dim, pol))
             for dim, pol in profile.items()}
    return JointCapsAgent(
        cells=cells,
        llm=subj,
        top_k_per_dim=top_k_per_dim,
        mode=mode,
        tau=tau,
        rng=np.random.default_rng(seed),
    )


def _run_profile_battery(
    args: argparse.Namespace,
    subj: LLMClient,
    inventory: Inventory,
    out_dir: Path,
    profiles: list[dict[str, str]],
    sweep_label: str,
) -> dict:
    rows: list[dict] = []

    for prof_idx, profile in enumerate(profiles):
        prof_label = _label_for_profile(profile)
        for mode_label, mode, tau in [
            ("argmax", "argmax", 0.05),
            ("softmax_t02", "softmax", 0.2),
        ]:
            for run_id in range(args.n_runs):
                seed = args.seed + 10_000 * prof_idx + 1000 * run_id + (
                    0 if mode == "argmax" else 1
                )
                t0 = time.time()
                agent = _build_joint_agent(
                    profile, subj,
                    top_k_per_dim=args.top_k_per_dim,
                    mode=mode,
                    tau=tau,
                    seed=seed,
                )
                report = administer(agent, inventory, workers=args.workers)
                dt = time.time() - t0

                # for each cell active in profile, did the dim move correctly?
                per_dim_status: dict[str, dict] = {}
                num_correct = 0
                num_active = 0
                for dim, pol in profile.items():
                    sgn = _expected_direction(report, dim, pol)
                    per_dim_status[dim] = {
                        "expected_pol": pol,
                        "score": report.dimension_scores.get(dim),
                        "direction_match": sgn,
                    }
                    num_active += 1
                    if sgn > 0:
                        num_correct += 1
                # off-target leakage on inactive dims:
                inactive_dims = [d for d in DIMS if d not in profile]
                inactive_scores = {d: report.dimension_scores.get(d) for d in inactive_dims}

                row = {
                    "profile_label": prof_label,
                    "profile": dict(profile),
                    "mode_label": mode_label,
                    "mode": mode,
                    "tau": tau,
                    "run_id": run_id,
                    "seed": seed,
                    "all_dim_scores": dict(report.dimension_scores),
                    "per_dim_status": per_dim_status,
                    "num_active": num_active,
                    "num_correct_direction": num_correct,
                    "inactive_dim_scores": inactive_scores,
                    "parse_failures": report.parse_failures,
                    "wall_s": round(dt, 1),
                }
                rows.append(row)
                print(
                    f"[{sweep_label} prof={prof_label:25s} {mode_label:11s} run={run_id}] "
                    f"correct={num_correct}/{num_active}  "
                    f"({dt:.1f}s)"
                )

    # aggregate per (profile, mode)
    agg: list[dict] = []
    seen: set = set()
    for r in rows:
        key = (r["profile_label"], r["mode_label"])
        if key in seen:
            continue
        seen.add(key)
        sub = [x for x in rows if x["profile_label"] == r["profile_label"] and x["mode_label"] == r["mode_label"]]
        n_active = sub[0]["num_active"]
        # mean per-dim score across runs for active dims
        per_dim_mean = {}
        per_dim_match_rate = {}
        for dim in sub[0]["per_dim_status"].keys():
            scores = [x["per_dim_status"][dim]["score"] for x in sub
                      if x["per_dim_status"][dim]["score"] is not None]
            matches = [1 if x["per_dim_status"][dim]["direction_match"] > 0 else 0 for x in sub]
            per_dim_mean[dim] = round(float(np.mean(scores)), 3) if scores else None
            per_dim_match_rate[dim] = round(float(np.mean(matches)), 3)
        all_correct = float(np.mean([x["num_correct_direction"] / x["num_active"] for x in sub]))
        agg.append({
            "profile_label": r["profile_label"],
            "profile": r["profile"],
            "mode_label": r["mode_label"],
            "n_runs": len(sub),
            "n_active_dims": n_active,
            "per_dim_mean_score": per_dim_mean,
            "per_dim_direction_match_rate": per_dim_match_rate,
            "all_dim_correct_rate": round(all_correct, 3),
        })

    out = {
        "config": {
            "sweep": sweep_label,
            "subject_backend": args.subject_backend,
            "subject_model": args.subject_model,
            "subject_temperature": args.subject_temp,
            "top_k_per_dim": args.top_k_per_dim,
            "n_runs_per_profile": args.n_runs,
            "modes": ["argmax", "softmax_t02"],
            "corpus_dir": str(CORPUS_DIR.relative_to(ROOT)),
            "inventory": str(INV_PATH.relative_to(ROOT)),
        },
        "rows": rows,
        "agg": agg,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


def run_sweep_b(args: argparse.Namespace) -> None:
    """Joint 5-dim profiles (uniform mixture). Six representative profiles:
    5 single-pole sweeps (flip one dim, hold others high) + 1 all-low."""
    out_dir = OUT_DIR_BASE / "B_joint_uniform"
    out_dir.mkdir(parents=True, exist_ok=True)

    subj = make_subject(args)
    inventory = Inventory.load(INV_PATH)

    base_high = {d: "high" for d in DIMS}
    profiles = [base_high]  # all-high
    for flip in DIMS:
        p = dict(base_high)
        p[flip] = "low"
        profiles.append(p)
    profiles.append({d: "low" for d in DIMS})  # all-low

    out = _run_profile_battery(
        args, subj, inventory, out_dir, profiles, sweep_label="B_joint_uniform",
    )

    print()
    print("=== Sweep B summary: per-profile direction-match rate ===")
    for row in out["agg"]:
        if row["mode_label"] == "argmax":
            sm = next((r for r in out["agg"]
                       if r["profile_label"] == row["profile_label"]
                       and r["mode_label"] == "softmax_t02"), None)
            sm_rate = sm["all_dim_correct_rate"] if sm else float("nan")
            print(
                f"  {row['profile_label']:30s}  "
                f"argmax={row['all_dim_correct_rate']:.2f}  "
                f"softmax_t02={sm_rate:.2f}"
            )
    print(f"\nWrote {out_dir/'summary.json'}")


def run_sweep_c(args: argparse.Namespace) -> None:
    """Single / dual-trait personas. Tests degenerate-mixture behavior."""
    out_dir = OUT_DIR_BASE / "C_single_dual"
    out_dir.mkdir(parents=True, exist_ok=True)

    subj = make_subject(args)
    inventory = Inventory.load(INV_PATH)

    # 2 single-trait (sanity: should match §4.6 spread direction on that dim)
    # 4 dual-trait (compositionality)
    profiles = [
        {"neuroticism": "high"},
        {"extraversion": "low"},
        {"neuroticism": "high", "extraversion": "low"},
        {"conscientiousness": "high", "agreeableness": "high"},
        {"openness": "high", "neuroticism": "low"},
        {"agreeableness": "low", "conscientiousness": "low"},
    ]

    out = _run_profile_battery(
        args, subj, inventory, out_dir, profiles, sweep_label="C_single_dual",
    )

    print()
    print("=== Sweep C summary: per-profile direction-match rate ===")
    for row in out["agg"]:
        if row["mode_label"] == "argmax":
            sm = next((r for r in out["agg"]
                       if r["profile_label"] == row["profile_label"]
                       and r["mode_label"] == "softmax_t02"), None)
            sm_rate = sm["all_dim_correct_rate"] if sm else float("nan")
            print(
                f"  {row['profile_label']:30s}  "
                f"argmax={row['all_dim_correct_rate']:.2f}  "
                f"softmax_t02={sm_rate:.2f}"
            )
    print(f"\nWrote {out_dir/'summary.json'}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", choices=["A", "B", "C", "all"], required=True,
                   help="Which sub-experiment to run.")
    p.add_argument("--subject-backend", default="minimax")
    p.add_argument("--subject-model", default=os.getenv("MINIMAX_MODEL", "MiniMax-M2.7-highspeed"))
    p.add_argument("--subject-temp", type=float, default=0.7)
    p.add_argument("--n-runs", type=int, default=3, help="N runs per condition.")
    p.add_argument("--top-k", type=int, default=4, help="(Sweep A) atoms per query.")
    p.add_argument("--top-k-per-dim", type=int, default=1,
                   help="(Sweeps B/C) atoms per active cell per query.")
    p.add_argument("--workers", type=int, default=8, help="Concurrent inventory items.")
    p.add_argument("--seed", type=int, default=20260501, help="Base RNG seed.")
    args = p.parse_args()

    OUT_DIR_BASE.mkdir(parents=True, exist_ok=True)

    if args.sweep in ("A", "all"):
        run_sweep_a(args)
    if args.sweep in ("B", "all"):
        run_sweep_b(args)
    if args.sweep in ("C", "all"):
        run_sweep_c(args)


if __name__ == "__main__":
    main()
