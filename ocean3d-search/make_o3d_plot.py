# -*- coding: utf-8 -*-
# --- repository root (auto-located) ---
import os
from pathlib import Path
REPO_ROOT = Path(__file__).resolve()
for _ in range(8):
    if (REPO_ROOT / 'configs').is_dir():
        break
    REPO_ROOT = REPO_ROOT.parent
# --------------------------------------
"""make_o3d_plot.py — 3D experiment comparison figure (queries bar + full-target-rate line + coverage)
data: results/ocean3d/o3d_summary.csv
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
matplotlib.rcParams["axes.unicode_minus"] = False

SUM = Path(REPO_ROOT / "results/ocean3d/o3d_summary.csv")
OUT = Path(REPO_ROOT / "results/ocean3d/o3d_comparison.png")
OUT2 = Path(REPO_ROOT / "ocean3d-search/results")
OUT2.mkdir(parents=True, exist_ok=True)

rows = list(csv.DictReader(open(SUM, encoding="utf-8")))
names = [r["config"] for r in rows]
qrs = [float(r["mean_queries"]) for r in rows]
pts = [float(r["mean_points"]) for r in rows]
cov = [float(r["mean_covered"]) for r in rows]
rate = [float(r["full_target_rate%"]) for r in rows]

colors = ["#8ecae6"] * 10
colors[5] = "#fb8500"     # params7 best
colors[8] = "#e76f51"     # random
colors[9] = "#d62828"     # grid failed

fig, ax1 = plt.subplots(figsize=(13, 6.6), dpi=150)
x = np.arange(len(names))
b = ax1.bar(x, qrs, color=colors, alpha=0.9, edgecolor="black", linewidth=0.6)
ax1.set_xticks(x)
ax1.set_xticklabels(names, rotation=32, ha="right", fontsize=10)
ax1.set_ylabel("mean number of queries (lower is better)", fontsize=11)
ax1.set_title("3D cross-domain experiment: real Mediterranean dissolved-oxygen hypoxia-core search (69,861 cells / 349 targets / target=40 / budget 10,000 / 20 seeds)",
              fontsize=12)
ax1.set_ylim(0, 11500)
for xi, q in zip(x, qrs):
    ax1.text(xi, q + 120, f"{q:.0f}", ha="center", va="bottom", fontsize=8.5)
# annotate the best
ax1.annotate("O3D5_params7 best\n3894 queries", xy=(5, 3894), xytext=(3.1, 7800),
             arrowprops=dict(arrowstyle="->", color="#fb8500"), fontsize=10, color="#fb8500", fontweight="bold")

ax2 = ax1.twinx()
ax2.plot(x, cov, "o--", color="green", lw=1.5, ms=5, label="covered cells")
ax2.set_ylabel("covered cells", fontsize=11)
ax2.set_ylim(0, 11000)
ax2.grid(axis="y", ls=":", alpha=0.3)

# full-target rate is annotated as text below the axis (information retained without occupying the right-axis scale)
for xi, r in zip(x, rate):
    ax1.annotate(f"full {r:.0f}%", (xi, -650), ha="center", fontsize=8, color="#444")

ax1.grid(axis="y", ls=":", alpha=0.4)
fig.tight_layout()
plt.savefig(OUT, bbox_inches="tight")
plt.savefig(OUT2 / "o3d_comparison.png", bbox_inches="tight")
print("saved:", OUT)

# appendix: small figure of parameter-scale trend (queries for params2/5/7/full19)
fig2, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
sub = [r for r in rows if r["config"] in ("O3D3_params2", "O3D4_params5", "O3D5_params7", "O3D1_full19", "O3D2_sparse")]
names2 = [r["config"].replace("O3D", "P") for r in sub]
q2 = [float(r["mean_queries"]) for r in sub]
cols2 = ["#a3c4f3", "#7aa5d9", "#fb8500", "#c77dff", "#00b4d8"]
ax.bar(names2, q2, color=cols2, edgecolor="black", lw=0.6)
for i, q in enumerate(q2):
    ax.text(i, q + 40, f"{q:.0f}", ha="center", fontsize=10)
ax.set_ylabel("mean number of queries", fontsize=11)
ax.set_title("3D parameter-scale effect: 7 structural parameters are best, full 19 params dragged down by noise", fontsize=11)
ax.grid(axis="y", ls=":", alpha=0.4)
fig2.tight_layout()
plt.savefig(OUT2 / "o3d_param_effect.png", bbox_inches="tight")
plt.savefig(OUT.with_name("o3d_param_effect.png"), bbox_inches="tight")
print("saved:", OUT2 / "o3d_param_effect.png")
