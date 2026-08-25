"""Direct lineage diagnostic: does P^2 look like IPIP-NEO paraphrases? Does CAPS?

This is the *correct* test for the section 4.4 claim that NPTI's P^2 descriptions
were derived from IPIP-NEO while CAPS atoms come from independent ESM source.

We measure pairwise same-trait similarity:
  - P^2 atoms vs IPIP items  (expected HIGH if P^2 is paraphrased from IPIP)
  - CAPS atoms vs IPIP items (expected LOW if CAPS is from independent source)
  - CAPS atoms vs P^2 atoms  (cross-check)

For each source-item, we take its same-trait same-polarity nearest neighbor in the
target source and record max + top-5 mean cosine, plus trigram-Jaccard max.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
NPTI = REPO / "baselines" / "NPTI" / "NPTI" / "dataset"
CAPS_DIR = REPO / "data" / "if_then" / "paper" / "beck_2022_v2"
OUT_DIR = REPO / "runs" / "npti_overlap_diagnostic"

DIMS = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]
DIM_CAP = {d: d.capitalize() for d in DIMS}


def load_p2():
    rows = []
    raw = []
    with (NPTI / "description.json").open() as f:
        for line in f:
            line = line.strip()
            if line:
                raw.append(json.loads(line))
    for ri, row in enumerate(raw):
        for d in DIMS:
            hi = row.get(DIM_CAP[d])
            lo = row.get(f"{DIM_CAP[d]}_reversed")
            if hi:
                rows.append({"id": f"p2_{d}_high_{ri}", "trait": d, "polarity": "high", "text": hi.strip()})
            if lo:
                rows.append({"id": f"p2_{d}_low_{ri}", "trait": d, "polarity": "low", "text": lo.strip()})
    return rows


def load_ipip():
    spec = importlib.util.spec_from_file_location(
        "ipip_neo", str(REPO / "src/persona_caps/data_sources/ipip_neo.py")
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["ipip_neo"] = m
    spec.loader.exec_module(m)
    out = []
    for it in m.load_items():
        out.append({
            "id": f"ipip_{it.item_id}",
            "trait": it.domain,
            "polarity": "high" if it.polarity == "+" else "low",
            "text": it.text.strip(),
        })
    return out


def load_caps():
    rows = []
    for path in sorted(CAPS_DIR.glob("*.jsonl")):
        if "patterns" in path.name:
            continue
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                a = json.loads(line)
                dim = (a.get("target_dimension") or "").strip().lower()
                pol = (a.get("polarity") or "").strip().lower()
                sit = (a.get("situation") or "").strip()
                beh = (a.get("behavior") or "").strip()
                if not (dim and pol and sit and beh):
                    continue
                rows.append({
                    "id": a["id"], "trait": dim, "polarity": pol,
                    "text": f"{sit} {beh}",
                })
    return rows


def trigrams(text):
    toks = text.lower().split()
    if len(toks) < 3:
        return set()
    return {tuple(toks[i:i+3]) for i in range(len(toks) - 2)}


def jacc(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def topk_mean(arr, k):
    if arr.size == 0:
        return float("nan")
    k = min(k, arr.size)
    return float(np.partition(arr, -k)[-k:].mean())


def pairwise_summary(src, src_v, src_tg, tgt, tgt_v, tgt_tg, match_polarity=True):
    """For each src item, find top-K same-trait (and same-polarity if asked)
    matches in tgt. Returns lists of top1, top5_mean, trigram_max."""
    tgt_idx = defaultdict(list)
    for i, r in enumerate(tgt):
        key = (r["trait"], r["polarity"]) if match_polarity else r["trait"]
        tgt_idx[key].append(i)

    t1, t5, tg = [], [], []
    examples = []
    for i, r in enumerate(src):
        key = (r["trait"], r["polarity"]) if match_polarity else r["trait"]
        cand = tgt_idx.get(key, [])
        if not cand:
            continue
        sub_v = tgt_v[cand]
        sims = (src_v[i:i+1] @ sub_v.T)[0]
        top1 = float(sims.max())
        top5 = topk_mean(sims, 5)
        tri_vals = [jacc(src_tg[i], tgt_tg[j]) for j in cand]
        tri_max = max(tri_vals) if tri_vals else 0.0
        t1.append(top1); t5.append(top5); tg.append(tri_max)
        ami = int(np.argmax(sims))
        examples.append({
            "cos": top1, "tri": tri_max,
            "src_id": r["id"], "src_text": r["text"],
            "tgt_id": tgt[cand[ami]]["id"], "tgt_text": tgt[cand[ami]]["text"],
            "trait": r["trait"], "polarity": r["polarity"],
        })
    return t1, t5, tg, examples


def stat(xs):
    if not xs:
        return {"mean": None, "median": None, "max": None, "min": None, "n": 0}
    return {
        "mean": statistics.mean(xs),
        "median": statistics.median(xs),
        "max": max(xs),
        "min": min(xs),
        "n": len(xs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[load] P2"); p2 = load_p2(); print(f"  P2 total: {len(p2)}")
    print("[load] IPIP"); ipip = load_ipip(); print(f"  IPIP total: {len(ipip)}")
    print("[load] CAPS"); caps = load_caps(); print(f"  CAPS total: {len(caps)}")

    print(f"[encoder] loading {args.encoder}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.encoder)

    def enc(rows):
        v = model.encode([r["text"] for r in rows], normalize_embeddings=True,
                         show_progress_bar=False, batch_size=64)
        return np.asarray(v, dtype=np.float32)

    p2_v = enc(p2); ip_v = enc(ipip); cp_v = enc(caps)
    p2_tg = [trigrams(r["text"]) for r in p2]
    ip_tg = [trigrams(r["text"]) for r in ipip]
    cp_tg = [trigrams(r["text"]) for r in caps]

    pairs = [
        ("P2_to_IPIP",   p2,   p2_v,   p2_tg,  ipip, ip_v, ip_tg, True),
        ("CAPS_to_IPIP", caps, cp_v,   cp_tg,  ipip, ip_v, ip_tg, True),
        ("CAPS_to_P2",   caps, cp_v,   cp_tg,  p2,   p2_v, p2_tg, True),
        ("IPIP_to_P2",   ipip, ip_v,   ip_tg,  p2,   p2_v, p2_tg, True),
    ]

    agg = {}
    examples_all = {}
    for name, src, sv, stg, tgt, tv, ttg, mp in pairs:
        t1, t5, tg, ex = pairwise_summary(src, sv, stg, tgt, tv, ttg, mp)
        agg[name] = {
            "top1_cosine": stat(t1),
            "top5_mean_cosine": stat(t5),
            "trigram_max": stat(tg),
        }
        examples_all[name] = sorted(ex, key=lambda x: -x["cos"])[:5]
        print(f"\n[{name}] n={len(t1)}  top1 mean={statistics.mean(t1):.3f}  "
              f"top5 mean={statistics.mean(t5):.3f}  trigram-max mean={statistics.mean(tg):.3f}")

    (out / "lineage_aggregate.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    (out / "lineage_examples.json").write_text(json.dumps(examples_all, indent=2), encoding="utf-8")

    def fmt(x, n=3):
        return "-" if x is None else f"{x:.{n}f}"

    lines = [
        "# Direct lineage diagnostic (same-trait, same-polarity)",
        "",
        "| Pair | n | cos top-1 mean | cos top-5 mean | trigram-max mean |",
        "|---|---|---|---|---|",
    ]
    for name in ["P2_to_IPIP", "CAPS_to_IPIP", "CAPS_to_P2", "IPIP_to_P2"]:
        b = agg[name]
        lines.append(
            f"| {name} | {b['top1_cosine']['n']} | "
            f"{fmt(b['top1_cosine']['mean'])} | "
            f"{fmt(b['top5_mean_cosine']['mean'])} | "
            f"{fmt(b['trigram_max']['mean'])} |"
        )
    lines.append("")
    lines.append("## Top-5 P^2 -> IPIP matches (closest paraphrases)")
    lines.append("")
    for i, e in enumerate(examples_all["P2_to_IPIP"], 1):
        lines.append(f"**{i}. trait={e['trait']} pol={e['polarity']} cos={e['cos']:.3f}**")
        lines.append(f"- P^2 ({e['src_id']}): {e['src_text'][:240]}")
        lines.append(f"- IPIP ({e['tgt_id']}): {e['tgt_text']}")
        lines.append("")
    lines.append("## Top-5 CAPS -> IPIP matches (closest)")
    lines.append("")
    for i, e in enumerate(examples_all["CAPS_to_IPIP"], 1):
        lines.append(f"**{i}. trait={e['trait']} pol={e['polarity']} cos={e['cos']:.3f}**")
        lines.append(f"- CAPS ({e['src_id']}): {e['src_text'][:240]}")
        lines.append(f"- IPIP ({e['tgt_id']}): {e['tgt_text']}")
        lines.append("")
    (out / "lineage_table.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[out] {out}/lineage_table.md")

    # Final terminal summary
    print()
    print("=== Lineage table ===")
    print(f"{'pair':14s}   n     cos@1   cos@5   tri-max")
    for name in ["P2_to_IPIP", "CAPS_to_IPIP", "CAPS_to_P2", "IPIP_to_P2"]:
        b = agg[name]
        print(f"{name:14s}  {b['top1_cosine']['n']:4d}   "
              f"{fmt(b['top1_cosine']['mean'])}   "
              f"{fmt(b['top5_mean_cosine']['mean'])}   "
              f"{fmt(b['trigram_max']['mean'])}")
    p2_ip = agg["P2_to_IPIP"]["top1_cosine"]["mean"]
    cp_ip = agg["CAPS_to_IPIP"]["top1_cosine"]["mean"]
    print()
    print(f"P^2->IPIP / CAPS->IPIP ratio: {p2_ip/cp_ip:.2f}x")
    if p2_ip > cp_ip + 0.05:
        print("=> P^2 is meaningfully closer to IPIP than CAPS is. Lineage claim SUPPORTED.")
    elif p2_ip > cp_ip:
        print("=> P^2 slightly closer to IPIP, but gap small.")
    else:
        print("=> CAPS at least as close to IPIP as P^2. Lineage claim NOT supported by this metric.")


if __name__ == "__main__":
    main()
