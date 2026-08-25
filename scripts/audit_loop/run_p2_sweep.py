"""P² baseline sweep: for every (cell, K, seed) that CAPS sanity'd, run
the P² persona pipeline end-to-end (paragraph build, P² sanity, then
all three audit strategies plus fix and measure).

This orchestrator deliberately re-uses the polluted corpora and
ground-truth logs already produced by ``run_sweep.py`` -- there is no
re-injection, so the persona *content* under both methods is exactly
the same; only the *delivery format* differs (retrieved atoms vs full
paragraph), as required by the §4.7 head-to-head.

For each condition we skip the (slow, $) audit-fix stages when the
P² sanity fails (then we have nothing to repair). The aggregated
manifest is written to ``runs/audit_loop/p2_sweep_manifest.json``.
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


CELL_INFO = {
    "high_extraversion": ("extraversion", "high"),
    "low_agreeableness": ("agreeableness", "low"),
    "low_conscientiousness": ("conscientiousness", "low"),
    "high_neuroticism": ("neuroticism", "high"),
    "low_openness": ("openness", "low"),
}


def _run(cmd, *, label):
    print(f"[p2-sweep] {label}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout, flush=True)
        print(proc.stderr, file=sys.stderr, flush=True)
        raise RuntimeError(f"{label} failed code={proc.returncode}")
    return proc.stdout


def _cond_name(cell, k, seed) -> str:
    return f"{cell}__k{k}_s{seed}_targeted"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default=str(OUT / "sweep_manifest.json"),
                   help="CAPS sweep manifest to read conditions from")
    p.add_argument("--cells", nargs="+", default=None)
    p.add_argument("--k-values", nargs="+", type=int, default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--subject", default="minimax")
    p.add_argument("--auditor", default="gpt-4o")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--strategies", nargs="+", default=["oracle", "blind", "detect"])
    p.add_argument("--only-passing", action="store_true", default=True,
                   help="only audit conditions whose CAPS sanity passed (saves API)")
    p.add_argument("--also-failed", dest="only_passing", action="store_false")
    p.add_argument("--force", action="store_true")
    p.add_argument("--out-manifest", default=str(OUT / "p2_sweep_manifest.json"))
    args = p.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    conditions = manifest["conditions"]
    if args.cells:
        conditions = [c for c in conditions if c["cell"] in args.cells]
    if args.k_values:
        conditions = [c for c in conditions if c["k"] in args.k_values]
    if args.seeds:
        conditions = [c for c in conditions if c["seed"] in args.seeds]
    if args.only_passing:
        conditions = [c for c in conditions if c.get("sanity_passed")]

    out_conditions = []
    t0 = time.time()
    for cond in conditions:
        cell = cond["cell"]
        k = cond["k"]
        seed = cond["seed"]
        dim, polarity = CELL_INFO[cell]
        name = _cond_name(cell, k, seed)
        clean_corpus = DATA / f"{cell}.jsonl"
        polluted_corpus = OUT / "polluted" / f"{name}.jsonl"
        inj_log = OUT / "polluted" / f"{name}.log.json"
        p2_para = OUT / "p2_paragraphs" / f"{name}.json"
        p2_sanity = OUT / "p2_sanity" / f"{name}.json"

        record = {"cell": cell, "k": k, "seed": seed, "targeted_dim": dim, "polarity": polarity}

        try:
            if not p2_sanity.exists() or args.force:
                p2_sanity.parent.mkdir(parents=True, exist_ok=True)
                _run(
                    [
                        sys.executable, str(SCRIPTS / "p2_inject_and_sanity.py"),
                        "--clean-corpus", str(clean_corpus),
                        "--polluted-corpus", str(polluted_corpus),
                        "--inventory", str(INVENTORY),
                        "--backend", args.subject,
                        "--workers", str(args.workers),
                        "--targeted-dim", dim,
                        "--out-paragraphs", str(p2_para),
                        "--out-sanity", str(p2_sanity),
                    ],
                    label=f"p2-sanity {name}",
                )
            p2_passed = json.loads(p2_sanity.read_text(encoding="utf-8")).get("pass_overall", False)
            record["p2_sanity_passed"] = p2_passed

            if p2_passed:
                record["strategies"] = {}
                for strat in args.strategies:
                    audit_out = OUT / "p2_audit" / f"{name}__{strat}.json"
                    if audit_out.exists() and not args.force:
                        record["strategies"][strat] = "reused"
                        continue
                    audit_out.parent.mkdir(parents=True, exist_ok=True)
                    _run(
                        [
                            sys.executable, str(SCRIPTS / "p2_audit_and_fix.py"),
                            "--strategy", strat,
                            "--clean-corpus", str(clean_corpus),
                            "--polluted-corpus", str(polluted_corpus),
                            "--injection-log", str(inj_log),
                            "--p2-sanity", str(p2_sanity),
                            "--inventory", str(INVENTORY),
                            "--targeted-dim", dim,
                            "--polarity", polarity,
                            "--backend", args.subject,
                            "--auditor", args.auditor,
                            "--workers", str(args.workers),
                            "--out", str(audit_out),
                        ],
                        label=f"p2-{strat} {name}",
                    )
                    record["strategies"][strat] = "ok"
            else:
                record["strategies"] = {}
                print(f"[p2-sweep] skip audits, p2-sanity failed for {name}")
            record["status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            record["status"] = "error"
            record["error"] = str(exc)
            print(f"[p2-sweep] ERROR on {name}: {exc}", file=sys.stderr)

        out_conditions.append(record)
        print(f"[p2-sweep] elapsed={time.time() - t0:.0f}s done {name}: {record.get('strategies')}", flush=True)

    out = Path(args.out_manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "subject": args.subject,
        "auditor": args.auditor,
        "strategies": args.strategies,
        "source_manifest": args.manifest,
        "conditions": out_conditions,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[p2-sweep] wrote {out}  total_elapsed={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
