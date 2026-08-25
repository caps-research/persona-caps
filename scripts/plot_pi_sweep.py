#!/usr/bin/env python
"""Plot π_d sweep curves for all five Big Five dims (publication-ready)."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

DIMS = [
    ("neuroticism",        "N", "runs/pi_sweep_n_gem/summary.json"),
    ("extraversion",       "E", "runs/pi_sweep_e_gem/summary.json"),
    ("conscientiousness",  "C", "runs/pi_sweep_c_gem/summary.json"),
    ("agreeableness",      "A", "runs/pi_sweep_a_gem/summary.json"),
    ("openness",           "O", "runs/pi_sweep_o_gem/summary.json"),
]
COLORS = {"N": "#d62728", "E": "#ff7f0e", "C": "#2ca02c", "A": "#1f77b4", "O": "#9467bd"}

fig, ax = plt.subplots(figsize=(6.5, 4.6), dpi=150)

# Reference: linear ideal (1 + 4w)
xs_ref = np.linspace(0, 1, 11)
ax.plot(xs_ref, 1 + 4 * xs_ref, color="grey", ls="--", lw=1.0, alpha=0.6,
        label="Linear ideal", zorder=0)

for full, short, summary_path in DIMS:
    s = json.load(open(summary_path))
    ws = [a["w_high"] for a in s["agg"]]
    means = [a["mean"] for a in s["agg"]]
    stds = [a["stdev"] for a in s["agg"]]
    spread = means[-1] - means[0]
    ax.errorbar(ws, means, yerr=stds, marker="o", lw=2.0, ms=6, capsize=3,
                color=COLORS[short], label=f"{short} (Δ={spread:+.2f})")

ax.set_xlabel(r"$\pi_{\mathrm{high}}$  (mixture weight on high-pole cell)",
              fontsize=11)
ax.set_ylabel("Inventory score on target dim (1–5)", fontsize=11)
ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_ylim(0.7, 5.3)
ax.grid(True, ls=":", alpha=0.4)
ax.legend(loc="upper left", fontsize=9, ncol=2, frameon=True, framealpha=0.95)
ax.set_title("Continuous personality strength control via mixture weights",
             fontsize=12)

# Annotate test setup
ax.text(0.99, 0.02,
        "Subject: Gemini-2.5-flash, T=0.7 | n=3 runs/weight | Mini-IPIP (4 items × dim)\n"
        "Per query: sample one of {high_d, low_d} with prob (π, 1−π); top-k=4 atoms",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7, color="dimgrey", style="italic")

plt.tight_layout()
out = Path("paper/figures/pi_sweep_curves.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, bbox_inches="tight", dpi=200)
plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print(f"Wrote {out} (and .pdf)")
