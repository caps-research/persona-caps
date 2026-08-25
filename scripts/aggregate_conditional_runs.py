"""Aggregate N runs of exp_conditional_persona.py into mean±SD report.

Reads the trait + context labels from the first run's `config` block (the
exp_conditional_persona.py script writes `trait`, `contexts`, and
`expected_polarity` into each run JSON), so this aggregator works for both
the conscientiousness work/family runs and the agreeableness support/conflict
runs without a code change.
"""
from __future__ import annotations
import argparse, json, statistics
from collections import defaultdict
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", type=Path, help="run JSON files")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    first_payload = json.loads(args.runs[0].read_text())
    cfg = first_payload.get("config", {})
    contexts = cfg.get("contexts", ["work", "family"])
    expected_polarity = cfg.get("expected_polarity",
                                {"work": "high", "family": "low"})
    trait = cfg.get("trait", "conscientiousness")
    if len(contexts) != 2:
        raise SystemExit(f"expected exactly 2 contexts, got {contexts}")
    ordered = sorted(contexts, key=lambda c: 0 if expected_polarity.get(c) == "high" else 1)
    ctx_high, ctx_low = ordered
    gap_label = f"{ctx_high}-{ctx_low}"

    for rf in args.runs[1:]:
        rcfg = json.loads(rf.read_text()).get("config", {})
        if rcfg.get("contexts", contexts) != contexts:
            raise SystemExit(f"context mismatch across runs: {rf} has {rcfg.get('contexts')}, expected {contexts}")

    per_run_summary = []
    per_method_ctx = defaultdict(list)
    methods = ("cond_caps", "string_cond", "single_high_caps", "single_low_caps")

    for run_file in args.runs:
        payload = json.loads(run_file.read_text())
        rows = payload["rows"]
        means_this_run = {}
        for method in methods:
            for ctx in contexts:
                vals = [r["rating"] for r in rows
                        if r["method"] == method and r["context"] == ctx and r["rating"] is not None]
                m = statistics.mean(vals) if vals else None
                means_this_run[(method, ctx)] = m
                if m is not None:
                    per_method_ctx[(method, ctx)].append(m)
            ws = means_this_run[(method, ctx_high)]
            fs = means_this_run[(method, ctx_low)]
            gap = (ws - fs) if (ws is not None and fs is not None) else None
            means_this_run[(method, "gap")] = gap
            if gap is not None:
                per_method_ctx[(method, "gap")].append(gap)
        per_run_summary.append({"file": str(run_file),
                                "means": {f"{m}|{c}": v for (m, c), v in means_this_run.items()}})

    print(f"\n=== Aggregate over N={len(args.runs)} runs (mean ± SD across runs) ===")
    print(f"trait={trait}  contexts={contexts}  expected_polarity={expected_polarity}")
    print(f"{'method':<22} {ctx_high:>14} {ctx_low:>14} {gap_label:>14}")
    out_table = {}
    for method in methods:
        line = [f"{method:<22}"]
        row = {}
        for key in (ctx_high, ctx_low, "gap"):
            vals = per_method_ctx.get((method, key), [])
            if len(vals) >= 2:
                m, sd = statistics.mean(vals), statistics.stdev(vals)
                cell = f"{m:>+6.2f} ± {sd:.2f}" if key == "gap" else f"{m:>5.2f} ± {sd:.2f}"
                row[key] = {"mean": m, "sd": sd, "n": len(vals), "vals": vals}
            elif len(vals) == 1:
                cell = f"{vals[0]:>+6.2f} (N=1)" if key == "gap" else f"{vals[0]:>5.2f} (N=1)"
                row[key] = {"mean": vals[0], "sd": None, "n": 1, "vals": vals}
            else:
                cell = "-"
                row[key] = None
            line.append(f"{cell:>14}")
        out_table[method] = row
        print(" ".join(line))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "trait": trait,
        "contexts": contexts,
        "expected_polarity": expected_polarity,
        "n_runs": len(args.runs),
        "runs": [str(p) for p in args.runs],
        "per_run_summary": per_run_summary,
        "aggregate": out_table,
    }, indent=2))
    print(f"\n[ok] wrote -> {args.out}")

if __name__ == "__main__":
    main()
