"""End-to-end batch driver.

Three stages, each can be skipped independently:

1. ``seed`` — generate if-then corpora for all (Big Five dim x polarity)
   pairs that don't already exist on disk. Skips any corpus already there.
2. ``exp01`` — run RQ1 (inventory consistency) for each backend, with all
   three persona methods (caps / trait_label / persona_description).
3. ``exp02`` — run RQ2 (cross-situational signature) for each backend on
   a configurable set of (dim, polarity) targets.

Outputs land under ``runs/<timestamp>/`` so successive invocations don't
overwrite earlier results. Each stage is a thin wrapper over the
corresponding script in ``experiments/`` or ``scripts/``.

Usage examples:
    python scripts/run_all.py --backends minimax --runs 3
    python scripts/run_all.py --backends minimax claude gpt-4o-mini --skip seed
    python scripts/run_all.py --only exp02 --signature-targets neuroticism:high neuroticism:low
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "data" / "if_then"
DIMENSIONS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")
POLARITIES = ("high", "low")
DEFAULT_SIG_TARGETS = ["neuroticism:high", "neuroticism:low"]


def _run(cmd: list[str], *, dry_run: bool) -> int:
    pretty = " ".join(cmd)
    print(f"  $ {pretty}")
    if dry_run:
        return 0
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"  !! exit {proc.returncode}")
    return proc.returncode


def stage_seed(args: argparse.Namespace) -> None:
    print(f"\n[run_all] STAGE 1: seed corpora ({args.atoms_per_corpus} atoms per corpus)")
    seeded = 0
    for dim in DIMENSIONS:
        for pol in POLARITIES:
            out = CORPUS_DIR / f"{pol}_{dim}.jsonl"
            if out.exists() and not args.force_seed:
                print(f"  skip {out.name} (exists)")
                continue
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "seed_corpus.py"),
                "--trait", dim,
                "--level", pol,
                "--n", str(args.atoms_per_corpus),
                "--backend", args.seed_backend,
                "--out", str(out),
            ]
            if _run(cmd, dry_run=args.dry_run) == 0:
                seeded += 1
    print(f"[run_all] seeded {seeded} new corpora")


def stage_exp01(args: argparse.Namespace, run_dir: Path) -> None:
    print("\n[run_all] STAGE 2: exp01 (RQ1 inventory consistency)")
    for backend in args.backends:
        out = run_dir / f"exp01_{backend.replace(':', '_')}.json"
        cmd = [
            sys.executable,
            str(ROOT / "experiments" / "exp01_consistency.py"),
            "--backend", backend,
            "--runs", str(args.runs),
            "--temperature", str(args.temperature),
            "--dimensions", *args.exp01_dimensions,
            "--out", str(out),
        ]
        _run(cmd, dry_run=args.dry_run)


def stage_exp02(args: argparse.Namespace, run_dir: Path) -> None:
    print("\n[run_all] STAGE 3: exp02 (RQ2 cross-situational signature)")
    for backend in args.backends:
        for target in args.signature_targets:
            if ":" not in target:
                print(f"  bad target spec: {target} (expected dim:pol)")
                continue
            dim, pol = target.split(":", 1)
            out = run_dir / f"exp02_{backend.replace(':', '_')}_{pol}_{dim}.json"
            cmd = [
                sys.executable,
                str(ROOT / "experiments" / "exp02_cross_situational.py"),
                "--backend", backend,
                "--temperature", str(args.temperature),
                "--dimension", dim,
                "--polarity", pol,
                "--out", str(out),
            ]
            _run(cmd, dry_run=args.dry_run)


def stage_exp03(args: argparse.Namespace, run_dir: Path) -> None:
    print("\n[run_all] STAGE 4: exp03 (RQ3 compositional editing)")
    for backend in args.backends:
        for target in args.editing_targets:
            if ":" not in target:
                print(f"  bad target spec: {target} (expected dim:pol)")
                continue
            dim, pol = target.split(":", 1)
            out = run_dir / f"exp03_{backend.replace(':', '_')}_{dim}_to_{pol}.json"
            cmd = [
                sys.executable,
                str(ROOT / "experiments" / "exp03_editing.py"),
                "--backend", backend,
                "--temperature", str(args.temperature),
                "--dimension", dim,
                "--target-polarity", pol,
                "--steps", *[str(s) for s in args.editing_steps],
                "--runs", str(args.editing_runs),
                "--out", str(out),
            ]
            _run(cmd, dry_run=args.dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends", nargs="+", default=["minimax"],
        help="LLM backends to run experiments on (preset names or backend:model)",
    )
    parser.add_argument(
        "--seed-backend", default=None,
        help="backend used for corpus seeding (defaults to backends[0])",
    )
    parser.add_argument("--atoms-per-corpus", type=int, default=30)
    parser.add_argument("--runs", type=int, default=3, help="exp01 runs per agent")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--exp01-dimensions", nargs="+", default=list(DIMENSIONS),
        help="which Big Five dimensions to test in exp01",
    )
    parser.add_argument(
        "--signature-targets", nargs="+", default=DEFAULT_SIG_TARGETS,
        help="which (dim:pol) targets to test in exp02",
    )
    parser.add_argument(
        "--editing-targets", nargs="+", default=["neuroticism:high"],
        help="which (dim:pol) targets to edit toward in exp03",
    )
    parser.add_argument(
        "--editing-steps", nargs="+", type=int, default=[0, 5, 10, 15, 20],
        help="atom counts at which exp03 re-administers the inventory",
    )
    parser.add_argument("--editing-runs", type=int, default=2,
                        help="independent inventory runs per edit step in exp03")
    parser.add_argument("--force-seed", action="store_true",
                        help="re-seed corpora even if files exist")
    parser.add_argument(
        "--skip", nargs="*", default=[], choices=["seed", "exp01", "exp02", "exp03"],
        help="stages to skip",
    )
    parser.add_argument(
        "--only", nargs="*", default=None, choices=["seed", "exp01", "exp02", "exp03"],
        help="if set, only run these stages",
    )
    parser.add_argument("--dry-run", action="store_true", help="print commands but don't execute")
    args = parser.parse_args()

    if args.seed_backend is None:
        args.seed_backend = args.backends[0]

    stages = {"seed", "exp01", "exp02", "exp03"}
    if args.only is not None:
        stages = set(args.only)
    stages -= set(args.skip)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "runs" / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run_all] output dir: {run_dir}")
    print(f"[run_all] stages: {sorted(stages)}  backends: {args.backends}")
    if args.dry_run:
        print("[run_all] DRY RUN — no commands will execute")

    if "seed" in stages:
        stage_seed(args)
    if "exp01" in stages:
        stage_exp01(args, run_dir)
    if "exp02" in stages:
        stage_exp02(args, run_dir)
    if "exp03" in stages:
        stage_exp03(args, run_dir)

    print(f"\n[run_all] done. results -> {run_dir}")


if __name__ == "__main__":
    main()
