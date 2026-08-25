"""Cross-vendor (GPT-4o subject) pilot for §4.7 audit-loop.

Re-runs the full §4.7 pipeline on the high_extraversion / K=15 / seeds
{1,2,3} conditions using GPT-4o-2024-08-06 as the *subject* model, to
demonstrate that the CAPS-vs-P² findings hold when the subject is
swapped from MiniMax-M2.7 to a different model family.

Reuses (subject-independent):
  - polluted corpora     -> runs/audit_loop/polluted/
  - inject ground-truth logs
  - locator-b metadata classifications (those depend on the
    auditor LLM, not the subject)

Re-runs (subject-dependent):
  - sanity_check.py                     -> runs/audit_loop_xvendor/sanity/
  - locate_via_responses.py             -> runs/audit_loop_xvendor/locate_a/
  - apply_fix_and_measure.py (both A/B) -> runs/audit_loop_xvendor/fix/
  - p2_inject_and_sanity.py             -> runs/audit_loop_xvendor/p2_sanity/
  - p2_audit_and_fix.py (all 3 strat.)  -> runs/audit_loop_xvendor/p2_audit/

Locator B is reused from the main run because the locator-B classifier
(GPT-4o by default in both runs) is orthogonal to the subject; only the
*fix* re-administration cares about the subject's responses.

Total cost estimate at default config: ~$3-5 in GPT-4o credit.
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
MAIN_OUT = REPO / "runs" / "audit_loop"
XV_OUT = REPO / "runs" / "audit_loop_xvendor"


def _run(cmd: list[str], *, label: str) -> int:
    print(f"[xv] {label}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout[-1500:], flush=True)
    if proc.returncode not in (0, 2):
        print(proc.stderr[-1500:], file=sys.stderr, flush=True)
        raise RuntimeError(f"{label} failed code={proc.returncode}")
    if proc.returncode == 2:
        print(f"[xv] {label}: sanity returned code=2 (FAIL gate) -- continuing", flush=True)
    return proc.returncode


def _cond_name(cell: str, k: int, seed: int) -> str:
    return f"{cell}__k{k}_s{seed}_targeted"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cell", default="high_extraversion")
    p.add_argument("--targeted-dim", default="extraversion")
    p.add_argument("--polarity", default="high")
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--subject", default="openai:gpt-4o-2024-08-06")
    p.add_argument("--auditor", default="gpt-4o")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument(
        "--skip-p2", action="store_true",
        help="run CAPS-side only (debug)",
    )
    p.add_argument(
        "--out-manifest",
        default=str(XV_OUT / "xvendor_manifest.json"),
    )
    args = p.parse_args()

    for sub in ("sanity", "locate_a", "fix", "p2_sanity", "p2_paragraphs", "p2_audit"):
        (XV_OUT / sub).mkdir(parents=True, exist_ok=True)

    conditions: list[dict] = []
    t0 = time.time()

    for seed in args.seeds:
        name = _cond_name(args.cell, args.k, seed)
        polluted = MAIN_OUT / "polluted" / f"{name}.jsonl"
        inj_log = MAIN_OUT / "polluted" / f"{name}.log.json"
        if not polluted.exists():
            print(f"[xv] SKIP {name}: polluted corpus not found")
            continue

        sanity_out = XV_OUT / "sanity" / f"{name}.json"
        locate_a_out = XV_OUT / "locate_a" / f"{name}.json"
        locate_b_main = MAIN_OUT / "locate_b" / f"{name}.json"

        rec: dict = {
            "cell": args.cell, "k": args.k, "seed": seed,
            "targeted_dim": args.targeted_dim, "subject": args.subject,
        }

        try:
            if not sanity_out.exists():
                _run(
                    [
                        sys.executable, str(SCRIPTS / "sanity_check.py"),
                        "--clean-corpus", str(DATA / f"{args.cell}.jsonl"),
                        "--polluted-corpus", str(polluted),
                        "--inventory", str(INVENTORY),
                        "--backend", args.subject,
                        "--targeted-dim", args.targeted_dim,
                        "--workers", str(args.workers),
                        "--out", str(sanity_out),
                    ],
                    label=f"sanity {name}",
                )
            sanity_json = json.loads(sanity_out.read_text(encoding="utf-8"))
            rec["sanity_passed"] = sanity_json.get("pass_overall", False)

            if not locate_a_out.exists():
                _run(
                    [
                        sys.executable, str(SCRIPTS / "locate_via_responses.py"),
                        "--sanity", str(sanity_out),
                        "--injection-log", str(inj_log),
                        "--targeted-dim", args.targeted_dim,
                        "--out", str(locate_a_out),
                    ],
                    label=f"locate-a {name}",
                )

            fix_a_out = XV_OUT / "fix" / f"{name}__locatea.json"
            if rec["sanity_passed"] and not fix_a_out.exists():
                _run(
                    [
                        sys.executable, str(SCRIPTS / "apply_fix_and_measure.py"),
                        "--clean-corpus", str(DATA / f"{args.cell}.jsonl"),
                        "--polluted-corpus", str(polluted),
                        "--locator", str(locate_a_out),
                        "--sanity", str(sanity_out),
                        "--inventory", str(INVENTORY),
                        "--targeted-dim", args.targeted_dim,
                        "--backend", args.subject,
                        "--workers", str(args.workers),
                        "--out", str(fix_a_out),
                    ],
                    label=f"fix-a {name}",
                )

            fix_b_out = XV_OUT / "fix" / f"{name}__locateb.json"
            if rec["sanity_passed"] and locate_b_main.exists() and not fix_b_out.exists():
                _run(
                    [
                        sys.executable, str(SCRIPTS / "apply_fix_and_measure.py"),
                        "--clean-corpus", str(DATA / f"{args.cell}.jsonl"),
                        "--polluted-corpus", str(polluted),
                        "--locator", str(locate_b_main),
                        "--sanity", str(sanity_out),
                        "--inventory", str(INVENTORY),
                        "--targeted-dim", args.targeted_dim,
                        "--backend", args.subject,
                        "--workers", str(args.workers),
                        "--out", str(fix_b_out),
                    ],
                    label=f"fix-b {name}",
                )

            if not args.skip_p2:
                p2_para = XV_OUT / "p2_paragraphs" / f"{name}.json"
                p2_sanity = XV_OUT / "p2_sanity" / f"{name}.json"
                if not p2_sanity.exists():
                    _run(
                        [
                            sys.executable, str(SCRIPTS / "p2_inject_and_sanity.py"),
                            "--clean-corpus", str(DATA / f"{args.cell}.jsonl"),
                            "--polluted-corpus", str(polluted),
                            "--inventory", str(INVENTORY),
                            "--backend", args.subject,
                            "--workers", str(args.workers),
                            "--targeted-dim", args.targeted_dim,
                            "--out-paragraphs", str(p2_para),
                            "--out-sanity", str(p2_sanity),
                        ],
                        label=f"p2-sanity {name}",
                    )
                p2_passed = json.loads(p2_sanity.read_text(encoding="utf-8")).get(
                    "pass_overall", False
                )
                rec["p2_sanity_passed"] = p2_passed

                if p2_passed:
                    for strat in ("oracle", "blind", "detect"):
                        audit_out = XV_OUT / "p2_audit" / f"{name}__{strat}.json"
                        if audit_out.exists():
                            continue
                        _run(
                            [
                                sys.executable, str(SCRIPTS / "p2_audit_and_fix.py"),
                                "--strategy", strat,
                                "--clean-corpus", str(DATA / f"{args.cell}.jsonl"),
                                "--polluted-corpus", str(polluted),
                                "--injection-log", str(inj_log),
                                "--p2-sanity", str(p2_sanity),
                                "--inventory", str(INVENTORY),
                                "--targeted-dim", args.targeted_dim,
                                "--polarity", args.polarity,
                                "--backend", args.subject,
                                "--auditor", args.auditor,
                                "--workers", str(args.workers),
                                "--out", str(audit_out),
                            ],
                            label=f"p2-{strat} {name}",
                        )

            rec["status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            rec["status"] = "error"
            rec["error"] = str(exc)
            print(f"[xv] ERROR on {name}: {exc}", file=sys.stderr, flush=True)

        conditions.append(rec)
        print(f"[xv] elapsed={time.time()-t0:.0f}s done {name}: {rec}", flush=True)

    out = Path(args.out_manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "subject": args.subject,
        "auditor": args.auditor,
        "cell": args.cell,
        "k": args.k,
        "seeds": args.seeds,
        "conditions": conditions,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[xv] wrote {out}  total_elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
