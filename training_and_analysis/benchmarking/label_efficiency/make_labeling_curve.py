#!/usr/bin/env python
"""Labeling-efficiency curve for per-dataset mitochondria adaptation, plotted from the CV JSONs
written by mito_cv.py and mito_empanada_ft.py (cv_*.json in --results-dir).
Schema per file: {"base_crossimg": <k=0 zero-shot cross-image Dice>, "curve": {k: [mean, std, R, ...]}}.
Held-out test Dice, image-disjoint CV (hold out 2 whole images, R=8 random train/test swaps), +/-1 std bands.
Writes labeling_curve.png and labeling_curve.svg into --out-dir.
Usage: python make_labeling_curve.py --results-dir DIR --out-dir DIR"""
import argparse, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(description="Plot the labeling-efficiency curve from cv_*.json files.")
ap.add_argument("--results-dir", required=True, help="directory holding the cv_*.json files")
ap.add_argument("--out-dir", required=True, help="output directory for labeling_curve.png/.svg")
args = ap.parse_args()
SRC = args.results_dir
OUT = args.out_dir
# (json stem, legend label, colour, marker)
ARMS = [
    ("cv_qem_cem_head_only",  "QuantEM ViT-B · head-only", "#1f77b4", "o"),
    ("cv_omni_cem_head_only", "OmniEM ViT-L · head-only",  "#7E57C2", "s"),
    ("cv_omni_cem_lora",      "OmniEM ViT-L · LoRA",       "#2CA25F", "D"),
    ("cv_mitonet_none",       "MitoNet · empanada FT (none)", "#C0392B", "^"),
    ("cv_mitonet_all",        "MitoNet · empanada FT (all)",  "#E6824A", "v"),
]

fig, ax = plt.subplots(figsize=(7.6, 5.2), facecolor="white")
for stem, label, colour, marker in ARMS:
    d = json.load(open(f"{SRC}/{stem}.json"))
    ks = sorted(int(k) for k in d["curve"])
    xs = [0] + ks
    ys = [float(d["base_crossimg"])] + [float(d["curve"][str(k)][0]) for k in ks]
    sd = [0.0] + [float(d["curve"][str(k)][1]) for k in ks]
    ys, sd = np.array(ys), np.array(sd)
    ax.plot(xs, ys, marker=marker, ms=5, lw=2.0, color=colour, label=label)
    ax.fill_between(xs, ys - sd, ys + sd, color=colour, alpha=0.15, lw=0)

ax.axhline(0.90, ls="--", lw=1.0, color="0.35")     # 0.90 reference line
ax.set_xlabel("# labeled training regions")
ax.set_ylabel("Dice")
ax.set_ylim(0, 1.0)
ax.set_xticks([0, 1, 2, 3, 4, 6, 8, 10])
ax.grid(alpha=0.25); ax.set_axisbelow(True)
ax.legend(fontsize=9, loc="lower right", frameon=False)
fig.tight_layout()
os.makedirs(OUT, exist_ok=True)
fig.savefig(f"{OUT}/labeling_curve.png", dpi=200, facecolor="white")
fig.savefig(f"{OUT}/labeling_curve.svg", facecolor="white")
print(f"saved labeling_curve.png/.svg -> {OUT}")
