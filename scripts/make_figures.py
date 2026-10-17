"""Regenerate Figure 1 (paper/figures/fig_bias_magnitudes.png) from raw pilot data.

Reads results/pilot_results.jsonl, recomputes bias magnitudes per lever
with code/pilot_analysis.bias_magnitude, and plots them against the
published human effect sizes. Run from the project root:
    python3 code/make_figures.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_analysis import load, analyze, bias_magnitude, HUMAN_TARGETS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figures", "fig_bias_magnitudes.png")

BIAS_ORDER = ["framing", "conjunction", "anchoring", "endowment", "crt"]
LABELS = ["Framing", "Conjunction", "Anchoring", "Endowment", "CRT"]
LEVER_LABELS = {
    "L0": "L0 (neutral)",
    "L1": "L1 (act-humanlike)",
    "L2": "L2 (gut)",
    "L3": "L3 (norm)",
}


def main():
    recs = load()
    cell, _, _ = analyze(recs)
    targets = [HUMAN_TARGETS[b][0] for b in BIAS_ORDER]
    mags = {}
    for lever in ("L0", "L1", "L2", "L3"):
        mags[lever] = [bias_magnitude(b, lever, cell) for b in BIAS_ORDER]

    x = np.arange(len(BIAS_ORDER))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, lever in enumerate(("L0", "L1", "L2", "L3")):
        ax.bar(x + (i - 1.5) * width, mags[lever], width,
               label=LEVER_LABELS[lever])
    ax.scatter(x, targets, color="black", marker="D", s=45, zorder=5,
               label="Human target")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("Bias magnitude ([0,1] scale)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Measured bias magnitudes by lever vs. published human "
                 "effect sizes (Qwen3-1.7B)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
