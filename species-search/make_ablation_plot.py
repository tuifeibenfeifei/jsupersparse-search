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
"""make_ablation_plot.py — ablation comparison figure (two species x 9 configs x 4 metrics)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import csv
from pathlib import Path

zh = None
for name in ["Microsoft YaHei", "SimHei", "SimSun"]:
    try:
        p = fm.findfont(fm.FontProperties(family=name))
        if p and not p.lower().endswith("dejavu"):
            zh = name
            break
    except Exception:
        pass
if zh:
    plt.rcParams["font.sans-serif"] = [zh]
    plt.rcParams["axes.unicode_minus"] = False

SRC = Path(REPO_ROOT / "results/species_search/ablation_species")
OUT = Path(REPO_ROOT / "species-search/results")
OUT.mkdir(parents=True, exist_ok=True)

NAME_MAP = {
    "A0_skeleton": "A0 minimal skeleton\n(no guidance/no sparsity/no curiosity)",
    "A1_full_guide": "A1 full-param guidance\n(v1 params19)",
    "A2_sparse_v2base": "A2 auto selection\n(v2 baseline)",
    "A3_sparse_cold150": "A3 + cold start 150",
    "A4_sparse_directed": "A4 + directed long jump",
    "A5_sparse_nocuriosity": "A5 no curiosity",
    "A6_sparse_topk4": "A6 topK=4",
    "A7_sparse_bins4": "A7 binning=4",
    "A8_sparse_bins16": "A8 binning=16",
    "A9_sparse_cluster": "A9 cluster guidance",
    "A10_sparse_cluster_directed": "A10 cluster\n+ directed long jump",
}

data = {}
for sp in ["black_muntjac", "clouded_leopard"]:
    rows = {}
    with open(SRC / f"ablation_species_{sp}_summary.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["config"]] = {
                "points": float(r["mean_points"]),
                "queries": float(r["mean_queries"]),
                "covered": float(r["mean_covered"]),
                "dual": float(r["dual_cluster_rate%"]),
            }
    data[sp] = rows

configs = [c for c in NAME_MAP if all(c in data[sp] for sp in data)]
labels = [NAME_MAP[c] for c in configs]

fig, axes = plt.subplots(1, 4, figsize=(16.5, 5.9))
x = np.arange(len(configs))
w = 0.32
colors = {"black_muntjac": "#4C78A8", "clouded_leopard": "#F2A900"}
names = {"black_muntjac": "black muntjac", "clouded_leopard": "clouded leopard"}

for ax, key, title in [
        (axes[0], "points", "points found (target 20)"),
        (axes[1], "queries", "mean number of queries"),
        (axes[2], "covered", "covered grid cells"),
        (axes[3], "dual", "dual-cluster coverage (%)")]:
    for si, sp in enumerate(["black_muntjac", "clouded_leopard"]):
        vals = [data[sp][c][key] for c in configs]
        ax.bar(x + (si - 0.5) * w, vals, w, label=names[sp], color=colors[sp])
        for xi, v in zip(x + (si - 0.5) * w, vals):
            ax.text(xi, v + max(vals) * 0.02, f"{v:.0f}", ha="center",
                    fontsize=7.2, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0, max(max(data[sp][c][key] for c in configs) for sp in data) * 1.18)
    ax.grid(alpha=0.3, axis="y")
axes[0].legend(fontsize=9.5)
fig.suptitle("Rare-species line ablation: guidance / feature selection / cold start / directed jump / curiosity / topK / number of bins / cluster-aware (mean over 20 seeds)",
             fontsize=13, y=1.03)
fig.tight_layout()
fig.savefig(OUT / "species_ablation_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("ABLATION PLOT OK ->", OUT)
