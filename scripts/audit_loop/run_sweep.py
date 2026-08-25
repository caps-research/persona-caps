"""Orchestrate the full §4.7 sweep over (cell, K, seed).

For each (cell, K, seed) condition, this driver runs:

  1. ``inject_polarity_flips.py``   -> polluted corpus + ground-truth log
  2. ``sanity_check.py``            -> per-condition PASS/FAIL gate
  3. ``locate_via_responses.py``    -> Locator (a) result
  4. ``locate_via_metadata.py``     -> Locator (b) result (only if sanity PASSes)

It does *not* run the fix step or the P² baselines — those are separate
scripts so we can iterate them independently of the (slow, paid)
inject + sanity stages.

Each per-condition run is logged at the end into a single sweep
manifest JSON that downstream aggregation can iterate over.

Default cell set: one per Big Five dimension, the larger of the two
polarities in Beck-2022-v2 (E:high, A:low, C:low, N:high, O:low).
The smaller cells are excluded because K=15 within-cell would exceed
their atom counts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "audit_loop"
DATA = REPO / "data" / "if_then" / "paper" / "beck_2022_v2"
INVENTORY = REPO / "data" / "inventories" / "ipip_short.json"
OUT = REPO / "runs" / "audit_loop"

DEFAULT_CELLS = [
    ("high_extraversion", "extraversion"),
    ("low_agreeableness", "agreeableness"),
    ("low_conscientiousness", "conscientiousness"),
    ("high_neuroticism", "neuroticism"),
    ("low_openness", "openness"),
]


def _run(cmd: list[str], *, label: str) -> tuple[int, str]:
    print(f"[sweep] {label}: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode not in (0, 2):
        # 2 is allowed (sanity FAIL); anything else is a hard error.
        print(proc.stdout, flush=True)
        print(proc.stderr, file=sys.stderr, flush=True)
        raise RuntimeError(f"{label} failed with code {proc.returncode}")
    return proc.returncode, proc.stdout


def _condition_name(cell: str, k: int, seed: int) -> str:
    return f"{cell}__k{k}_s{seed}_targeted"


def _injection_paths(cell: str, k: int, seed: int) -> tuple[Path, Path]:
    name = _condition_name(cell, k, seed)
    return (
        OUT / "polluted" / f"{name}.jsonl",
        OUT / "polluted" / f"{name}.log.json",
    )


def _sanity_path(cell: str, k: int, seed: int) -> Path:
    return OUT / "sanity" / f"{_condition_name(cell, k, seed)}.json"


def _locate_a_path(cell: str, k: int, seed: int) -> Path:
    return OUT / "locate_a" / f"{_condition_name(cell, k, seed)}.json"


def _locate_b_path(cell: str, k: int, seed: int) -> Path:
    return OUT / "locate_b" / f"{_condition_name(cell, k, seed)}.json"


def _run_inject(cell: str, dim: str, k: int, seed: int, *, force: bool) -> Path:
    out_corpus, out_log = _injection_paths(cell, k, seed)
    if out_corpus.exists() and out_log.exists() and not force:
        print(f"[sweep] reuse inject: {out_corpus.name}")
        return out_log
    out_corpus.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            str(SCRIPTS / "inject_polarity_flips.py"),
            "--corpus", str(DATA / f"{cell}.jsonl"),
            "--k", str(k),
            "--seed", str(seed),
            "--strategy", "targeted",
            "--inventory", str(INVENTORY),
            "--targeted-dim", dim,
            "--out-corpus", str(out_corpus),
            "--out-log", str(out_log),
        ],
        label=f"inject {cell} K={k} seed={seed}",
    )
    return out_log


def _run_sanity(cell: str, dim: str, k: int, seed: int, *, backend: str, workers: int, force: bool) -> tuple[Path, bool]:
    out_corpus, _ = _injection_paths(cell, k, seed)
    sanity_out = _sanity_path(cell, k, seed)
    if sanity_out.exists() and not force:
        print(f"[sweep] reuse sanity: {sanity_out.name}")
    else:
        sanity_out.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                sys.executable,
                str(SCRIPTS / "sanity_check.py"),
                "--clean-corpus", str(DATA / f"{cell}.jsonl"),
                "--polluted-corpus", str(out_corpus),
                "--inventory", str(INVENTORY),
                "--backend", backend,
                "--targeted-dim", dim,
                "--workers", str(workers),
                "--out", str(sanity_out),
            ],
            label=f"sanity {cell} K={k} seed={seed}",
        )
    passed = json.loads(sanity_out.read_text(encoding="utf-8")).get("pass_overall", False)
    return sanity_out, passed


def _run_locate_a(cell: str, dim: str, k: int, seed: int, *, force: bool) -> Path:
    sanity_out = _sanity_path(cell, k, seed)
    _, inj_log = _injection_paths(cell, k, seed)
    out = _locate_a_path(cell, k, seed)
    if out.exists() and not force:
        print(f"[sweep] reuse locate-a: {out.name}")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            str(SCRIPTS / "locate_via_responses.py"),
            "--sanity", str(sanity_out),
            "--injection-log", str(inj_log),
            "--targeted-dim", dim,
            "--out", str(out),
        ],
        label=f"locate-a {cell} K={k} seed={seed}",
    )
    return out


def _run_locate_b(cell: str, k: int, seed: int, *, classifier: str, force: bool) -> Path:
    out_corpus, inj_log = _injection_paths(cell, k, seed)
    out = _locate_b_path(cell, k, seed)
    if out.exists() and not force:
        print(f"[sweep] reuse locate-b: {out.name}")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            str(SCRIPTS / "locate_via_metadata.py"),
            "--polluted-corpus", str(out_corpus),
            "--injection-log", str(inj_log),
            "--classifier", classifier,
            "--out", str(out),
        ],
        label=f"locate-b {cell} K={k} seed={seed}",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cells",
        nargs="+",
        default=None,
        help="space-separated cell names (e.g. 'high_extraversion high_neuroticism'); "
             "default = the 5 main-results cells",
    )
    p.add_argument(
        "--k-values", nargs="+", type=int, default=[5, 10, 15],
        help="K values to sweep (default: 5 10 15)",
    )
    p.add_argument(
        "--seeds", nargs="+", type=int, default=[1, 2, 3],
        help="seeds to sweep (default: 1 2 3)",
    )
    p.add_argument("--subject", default="minimax", help="subject LLM for sanity")
    p.add_argument("--classifier", default="gpt-4o", help="locator-b classifier")
    p.add_argument("--workers", type=int, default=8, help="inventory workers")
    p.add_argument("--force", action="store_true",
                   help="re-run all stages even if outputs exist")
    p.add_argument(
        "--skip-locate-b-on-fail",
        action="store_true",
        default=True,
        help="if sanity FAILs, skip locator-b (still cheap if you flip with --no-skip)",
    )
    p.add_argument("--no-skip-locate-b-on-fail", dest="skip_locate_b_on_fail", action="store_false")
    p.add_argument(
        "--out-manifest",
        default=str(OUT / "sweep_manifest.json"),
        help="aggregated per-condition manifest",
    )
    args = p.parse_args()

    cell_specs = []
    name_to_dim = dict(DEFAULT_CELLS)
    if args.cells is None:
        cell_specs = list(DEFAULT_CELLS)
    else:
        for name in args.cells:
            if name not in name_to_dim:
                raise ValueError(
                    f"unknown cell {name}; use one of {sorted(name_to_dim)} "
                    "or extend run_sweep.py DEFAULT_CELLS"
                )
            cell_specs.append((name, name_to_dim[name]))

    conditions: list[dict] = []
    t0 = time.time()
    for cell, dim in cell_specs:
        for k in args.k_values:
            atoms_total = sum(1 for _ in (DATA / f"{cell}.jsonl").open(encoding="utf-8"))
            if k > atoms_total:
                print(f"[sweep] SKIP {cell} K={k} (corpus has only {atoms_total} atoms)")
                continue
            for seed in args.seeds:
                cond = {
                    "cell": cell, "targeted_dim": dim, "k": k, "seed": seed,
                    "atoms_in_cell": atoms_total,
                }
                try:
                    _run_inject(cell, dim, k, seed, force=args.force)
                    _, sanity_passed = _run_sanity(
                        cell, dim, k, seed,
                        backend=args.subject, workers=args.workers, force=args.force,
                    )
                    cond["sanity_passed"] = sanity_passed
                    _run_locate_a(cell, dim, k, seed, force=args.force)
                    cond["locate_a_done"] = True
                    if sanity_passed or not args.skip_locate_b_on_fail:
                        _run_locate_b(cell, k, seed, classifier=args.classifier, force=args.force)
                        cond["locate_b_done"] = True
                    else:
                        cond["locate_b_done"] = False
                    cond["status"] = "ok"
                except Exception as exc:  # noqa: BLE001 -- log and continue
                    cond["status"] = "error"
                    cond["error"] = str(exc)
                conditions.append(cond)
                elapsed = time.time() - t0
                print(f"[sweep] elapsed={elapsed:.0f}s; condition={cond}", flush=True)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "subject": args.subject,
        "classifier": args.classifier,
        "cells": cell_specs,
        "k_values": args.k_values,
        "seeds": args.seeds,
        "conditions": conditions,
    }
    out = Path(args.out_manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[sweep] wrote manifest -> {out}  total elapsed={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
