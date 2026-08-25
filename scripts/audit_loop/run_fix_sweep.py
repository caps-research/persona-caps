"""Run apply_fix_and_measure.py over every (cell, K, seed) condition
that CAPS sanity passed, for both Locator-A and Locator-B flag sets.

This produces the CAPS-side fix-and-off-target rows that the
aggregate table consumes. P² fix rows come from p2_audit_and_fix.py
via run_p2_sweep.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "audit_loop"
DATA = REPO / "data" / "if_then" / "paper" / "beck_2022_v2"
INVENTORY = REPO / "data" / "inventories" / "ipip_short.json"
OUT = REPO / "runs" / "audit_loop"

CELL_INFO = {
    "high_extraversion": "extraversion",
    "low_agreeableness": "agreeableness",
    "low_conscientiousness": "conscientiousness",
    "high_neuroticism": "neuroticism",
    "low_openness": "openness",
}


def _run(cmd, *, label):
    print(f"[fix-sweep] {label}", flush=True)
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
    p.add_argument("--manifest", default=str(OUT / "sweep_manifest.json"))
    p.add_argument("--locators", nargs="+", default=["a", "b"])
    p.add_argument("--subject", default="minimax")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--force", action="store_true")
    p.add_argument("--out-manifest", default=str(OUT / "fix_sweep_manifest.json"))
    args = p.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    conditions = [c for c in manifest["conditions"] if c.get("sanity_passed")]
    out = []
    t0 = time.time()
    for cond in conditions:
        cell = cond["cell"]
        k = cond["k"]
        seed = cond["seed"]
        dim = CELL_INFO[cell]
        name = _cond_name(cell, k, seed)
        clean_corpus = DATA / f"{cell}.jsonl"
        polluted_corpus = OUT / "polluted" / f"{name}.jsonl"
        sanity_json = OUT / "sanity" / f"{name}.json"
        record = {"cell": cell, "k": k, "seed": seed, "targeted_dim": dim, "fixes": {}}
        try:
            for loc in args.locators:
                locator_json = OUT / f"locate_{loc}" / f"{name}.json"
                if not locator_json.exists():
                    record["fixes"][loc] = "no_locator"
                    continue
                fix_out = OUT / "fix" / f"{name}__locate{loc}.json"
                if fix_out.exists() and not args.force:
                    record["fixes"][loc] = "reused"
                    continue
                fix_out.parent.mkdir(parents=True, exist_ok=True)
                _run(
                    [
                        sys.executable, str(SCRIPTS / "apply_fix_and_measure.py"),
                        "--clean-corpus", str(clean_corpus),
                        "--polluted-corpus", str(polluted_corpus),
                        "--locator", str(locator_json),
                        "--sanity", str(sanity_json),
                        "--inventory", str(INVENTORY),
                        "--targeted-dim", dim,
                        "--backend", args.subject,
                        "--workers", str(args.workers),
                        "--out", str(fix_out),
                    ],
                    label=f"fix-{loc} {name}",
                )
                record["fixes"][loc] = "ok"
            record["status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            record["status"] = "error"
            record["error"] = str(exc)
        out.append(record)
        print(f"[fix-sweep] elapsed={time.time() - t0:.0f}s done {name}", flush=True)

    out_path = Path(args.out_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "subject": args.subject,
        "source_manifest": args.manifest,
        "conditions": out,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[fix-sweep] wrote {out_path}  total={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
