#!/usr/bin/env python
"""Atom corpus diagnostic.

For each (dim, polarity) cell in a corpus, computes:

  1. Internal diversity         -- mean / median pairwise cosine distance
                                   among atom situation embeddings.
  2. Test-query coverage        -- for each Mini-IPIP item targeting this
                                   dim, max cosine similarity into each
                                   polarity's atom set.
  3. Cross-cell discrimination  -- for each atom, ratio (max sim within own
                                   cell - max sim outside cell). Negative
                                   means the atom is closer to a wrong
                                   cell than its own.

Output: a single Markdown table to stdout.

Usage:
    python scripts/diagnose_atoms.py \
        --corpus-dir data/if_then/paper/beck_2022_v3 \
        --inventory data/inventories/ipip_short.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DIMS = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]
POLS = ["high", "low"]


def cell_path(root: Path, dim: str, pol: str) -> Path:
    return root / f"{pol}_{dim}.jsonl"


def load_atoms(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", default="data/if_then/paper/beck_2022_v3")
    ap.add_argument("--inventory", default="data/inventories/ipip_short.json")
    ap.add_argument("--coverage-threshold", type=float, default=0.5)
    args = ap.parse_args()

    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    root = Path(args.corpus_dir)

    # Load all cells
    cells: dict[tuple[str, str], list[dict]] = {}
    embs: dict[tuple[str, str], np.ndarray] = {}
    for dim in DIMS:
        for pol in POLS:
            atoms = load_atoms(cell_path(root, dim, pol))
            cells[(dim, pol)] = atoms
            sits = [a["situation"] for a in atoms]
            embs[(dim, pol)] = encoder.encode(sits, normalize_embeddings=True)

    # Load inventory
    inv = json.load(open(args.inventory))
    items = inv["items"]
    item_embs = encoder.encode([it["text"] for it in items], normalize_embeddings=True)

    # =====================================================================
    # 1. Internal diversity (pairwise cos distance within cell)
    # =====================================================================
    print("## 1. Internal diversity (pairwise cosine distance within cell)")
    print()
    print("Higher = atoms cover a more diverse situation space within the cell.")
    print("Identity = 0 distance. Random unrelated sentences ≈ 0.7-0.9.")
    print()
    print("| Cell | n | mean dist | median dist | min dist | max dist |")
    print("|---|---|---|---|---|---|")
    for dim in DIMS:
        for pol in POLS:
            E = embs[(dim, pol)]
            n = len(E)
            if n < 2:
                print(f"| {pol}_{dim} | {n} | n/a | n/a | n/a | n/a |")
                continue
            sim = E @ E.T
            iu = np.triu_indices(n, k=1)
            dists = 1.0 - sim[iu]
            print(
                f"| {pol}_{dim} | {n} | {dists.mean():.3f} | "
                f"{np.median(dists):.3f} | {dists.min():.3f} | "
                f"{dists.max():.3f} |"
            )

    # =====================================================================
    # 2. Test-query coverage
    # =====================================================================
    print()
    print(f"## 2. Test-query coverage (Mini-IPIP items into target cell, threshold={args.coverage_threshold})")
    print()
    print("For each Mini-IPIP item targeting dim D, max cosine sim into the")
    print("(D, high) and (D, low) atom pools. The persona used at test time")
    print("matches the polarity, so coverage is the polarity-matched column.")
    print()
    print(f"`covered` = # items with max sim ≥ {args.coverage_threshold} in matched cell.")
    print()
    print("| Cell | n_items | mean max-sim | median max-sim | min max-sim | covered@thresh |")
    print("|---|---|---|---|---|---|")
    item_max_per_cell: dict[tuple[str, str], list[float]] = {}
    for dim in DIMS:
        dim_items_idx = [i for i, it in enumerate(items) if it["dimension"] == dim]
        for pol in POLS:
            E = embs[(dim, pol)]
            if len(E) == 0:
                print(f"| {pol}_{dim} | 0 | n/a | n/a | n/a | n/a |")
                continue
            max_sims = []
            for i in dim_items_idx:
                q = item_embs[i]
                sims = E @ q
                max_sims.append(float(sims.max()))
            item_max_per_cell[(dim, pol)] = max_sims
            covered = sum(1 for s in max_sims if s >= args.coverage_threshold)
            print(
                f"| {pol}_{dim} | {len(max_sims)} | {statistics.mean(max_sims):.3f} | "
                f"{statistics.median(max_sims):.3f} | {min(max_sims):.3f} | "
                f"{covered}/{len(max_sims)} |"
            )

    # =====================================================================
    # 3. Per-item which cell it lands in
    # =====================================================================
    print()
    print("## 3. Per-item nearest-cell vs target cell")
    print()
    print("For each test query, what cell does its nearest atom actually live in?")
    print("If the test polarity is `high D`, ideally the nearest atom is in `high_D`.")
    print()
    print("| Item | Target dim | Top cell (high+low pool) | Argmax sim |")
    print("|---|---|---|---|")
    cell_keys = list(cells.keys())
    all_E = np.vstack([embs[k] for k in cell_keys])
    cell_offsets = np.cumsum([0] + [len(embs[k]) for k in cell_keys])

    def which_cell(global_idx: int) -> tuple[str, str]:
        for ki, k in enumerate(cell_keys):
            if global_idx < cell_offsets[ki + 1]:
                return k
        return cell_keys[-1]

    for i, it in enumerate(items):
        sims = all_E @ item_embs[i]
        gi = int(sims.argmax())
        ck = which_cell(gi)
        print(f"| {it['id']}: \"{it['text'][:50]}\" | {it['dimension']} | {ck[1]}_{ck[0]} | {sims[gi]:.3f} |")

    # =====================================================================
    # 4. Cross-cell discrimination
    # =====================================================================
    print()
    print("## 4. Cross-cell discrimination per atom")
    print()
    print("For each atom, mean of (max sim within own cell to OTHERS) − (max sim outside own cell).")
    print("Positive = atom is uniquely closer to its own cell. Negative = atom is also similar")
    print("to atoms outside its label, suggesting noisy or ambiguous polarity.")
    print()
    print("| Cell | n | own-cell mean inner sim | best other-cell sim | margin |")
    print("|---|---|---|---|---|")
    for dim in DIMS:
        for pol in POLS:
            own = embs[(dim, pol)]
            n_own = len(own)
            if n_own < 2:
                print(f"| {pol}_{dim} | {n_own} | n/a | n/a | n/a |")
                continue
            other_keys = [k for k in cell_keys if k != (dim, pol)]
            other_E = np.vstack([embs[k] for k in other_keys])

            inner = own @ own.T
            np.fill_diagonal(inner, -1)
            inner_max = inner.max(axis=1)

            outer = own @ other_E.T
            outer_max = outer.max(axis=1)

            margin = inner_max - outer_max
            print(
                f"| {pol}_{dim} | {n_own} | {inner_max.mean():.3f} | "
                f"{outer_max.mean():.3f} | {margin.mean():+.3f} |"
            )


if __name__ == "__main__":
    main()
