#!/usr/bin/env python
"""Plot π_d sweep curves for all five Big Five dims, side-by-side
five-vendor cross-replication: Gemini-2.5-Flash, GPT-4o-2024-08-06,
MiniMax-M2.7-highspeed, Qwen3.5-9B (Q4_K_M), and Llama-3-8B-Instruct (Q4_0)
— both open-weight backbones served via Ollama on a single 3090."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

DIMS = [
    ("neuroticism",        "N"),
    ("extraversion",       "E"),
    ("conscientiousness",  "C"),
    ("agreeableness",      "A"),
    ("openness",           "O"),
]
COLORS = {"N": "#d62728", "E": "#ff7f0e", "C": "#2ca02c",
          "A": "#1f77b4", "O": "#9467bd"}
SHORT2DIR = {short: short.lower() for _, short in DIMS}
SHORT2DIR["N"] = "n"; SHORT2DIR["E"] = "e"; SHORT2DIR["C"] = "c"
SHORT2DIR["A"] = "a"; SHORT2DIR["O"] = "o"


def load(vendor_tag: str, short: str):
    p = Path(f"runs/pi_sweep_{SHORT2DIR[short]}_{vendor_tag}/summary.json")
    s = json.load(open(p))
    ws = [a["w_high"] for a in s["agg"]]
    means = [a["mean"] for a in s["agg"]]
    stds = [a["stdev"] for a in s["agg"]]
    return ws, means, stds


VENDORS = [
    ("Gemini-2.5-Flash",       "gem"),
    ("GPT-4o-2024-08-06",      "g4o"),
    ("MiniMax-M2.7-highspeed", "mm"),
    ("Qwen3.5-9B (Q4, local)", "q9b"),
    ("Llama-3-8B (Q4, local)", "l8b"),
]
fig, axes = plt.subplots(1, len(VENDORS), figsize=(4.0 * len(VENDORS), 4.4), dpi=150,
                         sharey=True, sharex=True)

for ax, (vendor_name, vendor_tag) in zip(axes, VENDORS):
    xs_ref = np.linspace(0, 1, 11)
    ax.plot(xs_ref, 1 + 4 * xs_ref, color="grey", ls="--", lw=1.0,
            alpha=0.6, label="Linear ideal", zorder=0)

    spreads = []
    for _, short in DIMS:
        ws, means, stds = load(vendor_tag, short)
        # Compute spread on rounded means so it matches the paper table
        # (avoids the 4.917-1.833=3.083 vs 4.92-1.83=3.09 cell-subtraction
        # ambiguity that would otherwise differ from the paper's §4.5 table).
        spread = round(means[-1], 2) - round(means[0], 2)
        spreads.append(spread)
        ax.errorbar(ws, means, yerr=stds, marker="o", lw=2.0, ms=6,
                    capsize=3, color=COLORS[short],
                    label=f"{short} (Δ={spread:+.2f})")

    mean_spread = sum(spreads) / len(spreads)
    ax.set_xlabel(r"$\pi_{\mathrm{high}}$  (mixture weight on high-pole cell)",
                  fontsize=11)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylim(0.7, 5.3)
    ax.grid(True, ls=":", alpha=0.4)
    ax.set_title(
        f"{vendor_name}  (mean spread {mean_spread:+.2f})",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=8, ncol=2,
              frameon=True, framealpha=0.95)

axes[0].set_ylabel("Inventory score on target dim (1–5)", fontsize=11)

# Footer with shared methodological note
fig.text(
    0.99, -0.005,
    "T=0.7 | n=3 runs/weight | Mini-IPIP (4 items × dim) | "
    "per query: sample one of {high_d, low_d} with prob (π, 1−π); top-k=4 atoms | "
    "pairwise 5-vendor 25-cell mean-grid Pearson r ∈ [+0.853, +0.998] (≥+0.97 within MM/GPT-4o/Gem/Qwen; App. A.7)",
    ha="right", va="bottom", fontsize=7, color="dimgrey", style="italic",
)

fig.suptitle(
    "Continuous personality strength control via mixture weights $\\pi_d$, "
    "replicated across three commercial LLMs and two open-weight ~9B / 8B models",
    fontsize=12, y=1.00,
)

plt.tight_layout(rect=[0, 0.02, 1, 0.97])
out = Path("paper/figures/pi_sweep_curves_xvendor.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, bbox_inches="tight", dpi=200)
plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print(f"Wrote {out} (and .pdf)")
