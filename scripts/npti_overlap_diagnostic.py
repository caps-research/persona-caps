"""PB-vs-{P2,IPIP,CAPS} overlap diagnostic for paper section 4.4."""

from __future__ import annotations

import argparse
import csv
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
PB_FILES = {d: NPTI / "test" / f"{DIM_CAP[d]}.json" for d in DIMS}


def load_pb():
    rows = []
    for trait, path in PB_FILES.items():
        n = 0
        with path.open() as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                q = (obj.get("question") or "").strip()
                if not q:
                    continue
                rows.append({"id": f"{trait}__{i}", "trait": trait, "polarity": None, "text": q})
                n += 1
        print(f"  PB[{trait}]: {n}")
    return rows


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


def stat_summary(xs):
    xs = [x for x in xs if x == x]  # drop NaN
    if not xs:
        return {"mean": None, "median": None, "max": None, "min": None, "std": None, "n": 0}
    return {
        "mean": statistics.mean(xs),
        "median": statistics.median(xs),
        "max": max(xs),
        "min": min(xs),
        "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
        "n": len(xs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[load] PB"); pb = load_pb()
    print(f"  PB total: {len(pb)}")
    print("[load] P2"); p2 = load_p2()
    print(f"  P2 total: {len(p2)}  per trait: " + ", ".join(f"{d}={sum(1 for r in p2 if r['trait']==d)}" for d in DIMS))
    print("[load] IPIP"); ipip = load_ipip()
    print(f"  IPIP total: {len(ipip)}  per trait: " + ", ".join(f"{d}={sum(1 for r in ipip if r['trait']==d)}" for d in DIMS))
    print("[load] CAPS"); caps = load_caps()
    print(f"  CAPS total: {len(caps)}  per trait: " + ", ".join(f"{d}={sum(1 for r in caps if r['trait']==d)}" for d in DIMS))

    print(f"[encoder] loading {args.encoder}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.encoder)

    def enc(texts):
        v = model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
        return np.asarray(v, dtype=np.float32)

    pb_v = enc([r["text"] for r in pb])
    p2_v = enc([r["text"] for r in p2])
    ip_v = enc([r["text"] for r in ipip])
    cp_v = enc([r["text"] for r in caps])
    pb_tg = [trigrams(r["text"]) for r in pb]
    p2_tg = [trigrams(r["text"]) for r in p2]
    ip_tg = [trigrams(r["text"]) for r in ipip]
    cp_tg = [trigrams(r["text"]) for r in caps]

    # Indexes by trait for each source
    def by_trait(rows):
        idx = defaultdict(list)
        for i, r in enumerate(rows):
            idx[r["trait"]].append(i)
        return idx

    p2_byT = by_trait(p2); ip_byT = by_trait(ipip); cp_byT = by_trait(caps)

    rows_out = []
    top_examples = {"p2_cos": [], "p2_tri": [], "ipip_cos": [], "caps_cos": []}

    for qi, q in enumerate(pb):
        trait = q["trait"]
        qv = pb_v[qi:qi+1]
        qtg = pb_tg[qi]
        rec = {"question_id": q["id"], "target_trait": trait, "target_polarity": "",
               "question_text": q["text"]}

        for tag, vecs, tgs, idxs in [
            ("p2", p2_v, p2_tg, p2_byT[trait]),
            ("ipip", ip_v, ip_tg, ip_byT[trait]),
            ("caps", cp_v, cp_tg, cp_byT[trait]),
        ]:
            if not idxs:
                rec.update({f"{tag}_top1_cosine": None, f"{tag}_top5_mean_cosine": None,
                            f"{tag}_max_trigram": None, f"{tag}_mean_trigram": None})
                continue
            sub_v = vecs[idxs]
            sims = (qv @ sub_v.T)[0]
            top1 = float(sims.max())
            top5m = topk_mean(sims, 5)
            tri = [jacc(qtg, tgs[i]) for i in idxs]
            tri_max = max(tri) if tri else 0.0
            tri_mean = statistics.mean(tri) if tri else 0.0
            rec[f"{tag}_top1_cosine"] = top1
            rec[f"{tag}_top5_mean_cosine"] = top5m
            rec[f"{tag}_max_trigram"] = tri_max
            rec[f"{tag}_mean_trigram"] = tri_mean

            # Track top examples
            ami = int(np.argmax(sims))
            atom_idx = idxs[ami]
            partner = {"p2": p2, "ipip": ipip, "caps": caps}[tag][atom_idx]
            entry = {"cos": top1, "pb_id": q["id"], "pb_text": q["text"],
                     "partner_id": partner["id"], "partner_text": partner["text"],
                     "trait": trait, "partner_polarity": partner.get("polarity")}
            key_cos = f"{tag}_cos"
            top_examples[key_cos].append(entry)
            if tag == "p2":
                tri_idx = int(np.argmax(tri))
                tri_partner = p2[idxs[tri_idx]]
                top_examples["p2_tri"].append({
                    "trigram": tri_max, "pb_id": q["id"], "pb_text": q["text"],
                    "partner_id": tri_partner["id"], "partner_text": tri_partner["text"],
                    "trait": trait, "partner_polarity": tri_partner.get("polarity"),
                })
        rows_out.append(rec)

    # Write CSV
    csv_path = out / "per_question.csv"
    fields = ["question_id", "target_trait", "target_polarity", "question_text",
              "p2_top1_cosine", "p2_top5_mean_cosine", "p2_max_trigram", "p2_mean_trigram",
              "ipip_top1_cosine", "ipip_top5_mean_cosine", "ipip_max_trigram", "ipip_mean_trigram",
              "caps_top1_cosine", "caps_top5_mean_cosine", "caps_max_trigram", "caps_mean_trigram"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows_out:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"[out] CSV -> {csv_path}")

    # Aggregate JSON
    agg = {}
    for tag in ("p2", "ipip", "caps"):
        block = {}
        for metric in ("top1_cosine", "top5_mean_cosine", "max_trigram", "mean_trigram"):
            key = f"{tag}_{metric}"
            xs = [r[key] for r in rows_out if r.get(key) is not None]
            block[metric] = stat_summary(xs)
        per_trait = {}
        for d in DIMS:
            d_xs_cos = [r[f"{tag}_top1_cosine"] for r in rows_out
                        if r["target_trait"] == d and r.get(f"{tag}_top1_cosine") is not None]
            d_xs_tri = [r[f"{tag}_max_trigram"] for r in rows_out
                        if r["target_trait"] == d and r.get(f"{tag}_max_trigram") is not None]
            per_trait[d] = {
                "n": len(d_xs_cos),
                "top1_cosine_mean": statistics.mean(d_xs_cos) if d_xs_cos else None,
                "max_trigram_mean": statistics.mean(d_xs_tri) if d_xs_tri else None,
            }
        block["per_trait"] = per_trait
        agg[f"{tag}_vs_pb"] = block

    agg_path = out / "aggregate.json"
    agg_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(f"[out] aggregate -> {agg_path}")

    # Comparison table markdown
    def fmt(x, n=3):
        return "-" if x is None else f"{x:.{n}f}"

    tbl_path = out / "comparison_table.md"
    lines = [
        "# PB-overlap comparison",
        "",
        "Pairing: same-trait, polarities pooled (PB has no per-Q polarity).",
        f"PB n = {len(pb)} questions over 5 traits.",
        "",
        "| Source | cosine top-1 mean | cosine top-5 mean | trigram max mean | trigram mean mean |",
        "|---|---|---|---|---|",
    ]
    for tag, name in [("ipip", "IPIP-NEO-300"), ("p2", "P^2 paragraphs"), ("caps", "CAPS atoms")]:
        b = agg[f"{tag}_vs_pb"]
        lines.append(
            f"| {name} | {fmt(b['top1_cosine']['mean'])} | "
            f"{fmt(b['top5_mean_cosine']['mean'])} | "
            f"{fmt(b['max_trigram']['mean'])} | {fmt(b['mean_trigram']['mean'])} |"
        )
    # ratios
    def ratio(a, b):
        if a is None or b in (None, 0):
            return None
        return a / b
    caps_b = agg["caps_vs_pb"]
    p2_b = agg["p2_vs_pb"]
    lines.append(
        f"| **P^2 / CAPS ratio** | "
        f"{fmt(ratio(p2_b['top1_cosine']['mean'], caps_b['top1_cosine']['mean']))}x | "
        f"{fmt(ratio(p2_b['top5_mean_cosine']['mean'], caps_b['top5_mean_cosine']['mean']))}x | "
        f"{fmt(ratio(p2_b['max_trigram']['mean'], caps_b['max_trigram']['mean']))}x | "
        f"{fmt(ratio(p2_b['mean_trigram']['mean'], caps_b['mean_trigram']['mean']))}x |"
    )
    lines.append("")
    lines.append("## Per-trait top-1 cosine mean (rows = source, cols = trait)")
    lines.append("")
    lines.append("| Source | E | A | C | N | O |")
    lines.append("|---|---|---|---|---|---|")
    for tag, name in [("ipip", "IPIP-NEO-300"), ("p2", "P^2 paragraphs"), ("caps", "CAPS atoms")]:
        pt = agg[f"{tag}_vs_pb"]["per_trait"]
        vals = [pt[d]["top1_cosine_mean"] for d in DIMS]
        lines.append(f"| {name} | " + " | ".join(fmt(v) for v in vals) + " |")
    tbl_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[out] comparison_table -> {tbl_path}")

    # Top examples
    ex_path = out / "top_overlap_examples.md"
    ex_lines = ["# Top-overlap examples (PB vs source)", ""]
    def shorten(s, n=300):
        return (s[:n] + "...") if len(s) > n else s
    for label, key, mkey in [
        ("Top-5 PB-P^2 by cosine", "p2_cos", "cos"),
        ("Top-5 PB-P^2 by trigram max", "p2_tri", "trigram"),
        ("Top-5 PB-IPIP by cosine", "ipip_cos", "cos"),
        ("Top-5 PB-CAPS by cosine (control)", "caps_cos", "cos"),
    ]:
        ex_lines.append(f"## {label}")
        ex_lines.append("")
        items = sorted(top_examples[key], key=lambda x: -x[mkey])[:5]
        for i, e in enumerate(items, 1):
            ex_lines.append(f"**{i}. trait={e['trait']} polarity={e.get('partner_polarity')} score={e[mkey]:.3f}**")
            ex_lines.append("")
            ex_lines.append(f"- PB ({e['pb_id']}): {shorten(e['pb_text'])}")
            ex_lines.append(f"- Partner ({e['partner_id']}): {shorten(e['partner_text'])}")
            ex_lines.append("")
    ex_path.write_text("\n".join(ex_lines), encoding="utf-8")
    print(f"[out] top_overlap_examples -> {ex_path}")

    # Final stdout summary
    print()
    print("=== Comparison table (means across", len(pb), "PB questions) ===")
    print(f"{'source':14s}  cos@1    cos@5    tri-max  tri-mean")
    for tag, name in [("ipip", "IPIP-NEO-300"), ("p2", "P^2 paragraphs"), ("caps", "CAPS atoms")]:
        b = agg[f"{tag}_vs_pb"]
        print(f"{name:14s}  {fmt(b['top1_cosine']['mean'])}    {fmt(b['top5_mean_cosine']['mean'])}    "
              f"{fmt(b['max_trigram']['mean'])}    {fmt(b['mean_trigram']['mean'])}")
    print()
    cap_c1 = caps_b['top1_cosine']['mean']
    p2_c1 = p2_b['top1_cosine']['mean']
    ip_c1 = agg['ipip_vs_pb']['top1_cosine']['mean']
    cap_tg = caps_b['max_trigram']['mean']
    p2_tg = p2_b['max_trigram']['mean']
    print(f"P^2/CAPS cos ratio: {p2_c1/cap_c1:.2f}x")
    print(f"P^2/CAPS trigram ratio: {(p2_tg/cap_tg if cap_tg else float('inf')):.2f}x")
    print(f"IPIP cos@1 mean: {ip_c1:.3f}")


if __name__ == "__main__":
    main()
