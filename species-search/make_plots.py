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
"""make_plots.py - generate rare-species automatic feature selection visualization PNGs
Output to species-search/results/:
  feature_selection_evolution.png     feature weight evolution for two species (scatter -> lock-in)
  auto_selection_vs_baselines.png     v1 all params / v2 auto selection / hand-picked 2 params comparison
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

# ---- Chinese font ----
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

OUT = Path(REPO_ROOT / "species-search/results")
OUT.mkdir(parents=True, exist_ok=True)

# ================= Fig 1: Feature selection evolution =================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

# Black Muntjac seed 111 (4330 queries to converge, 20/20)
x1 = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.3])  # k queries
ax1.plot(x1, [0, 0, 0.05, 0.11, 0.16, 0.33, 0.41], "o-", lw=2.2, label="bio6 Min temp of coldest month")
ax1.plot(x1, [0, 0, 0, 0, 0.13, 0.30, 0.39], "s-", lw=2.2, label="bio11 Mean temp of coldest month")
ax1.plot(x1, [0, 0, 0, 0, 0.10, 0.29, 0.39], "^-", lw=2.2, label="bio9 Mean temp of driest quarter")
ax1.plot(x1, [0.45, 0.35, 0.25, 0.23, 0.12, 0, 0], "x--", lw=2.0, color="gray", label="bio12 Annual precipitation")
ax1.plot(x1, [0.41, 0.30, 0.20, 0.18, 0.16, 0, 0], "d--", lw=2.0, color="lightgray", label="bio18 Precipitation of warmest quarter")
ax1.set_title("Black Muntjac · 20/20 points, 4330 queries to converge", fontsize=12)
ax1.set_xlabel("Query progress (k queries)")
ax1.set_ylabel("Feature guidance weight")
ax1.set_ylim(0, 0.52)
ax1.legend(fontsize=8.5, loc="upper right")
ax1.grid(alpha=0.3)

# Clouded Leopard seed 118 (18/20)
x2 = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
ax2.plot(x2, [0.11, 0.49, 0.48, 0.40, 0.32, 0.27, 0.14, 0.12, 0.16, 0.14, 0.09, 0.07, 0.02, 0, 0, 0], "o-", lw=2.2, label="bio2 Mean diurnal temp range")
ax2.plot(x2, [0.28, 0.34, 0.44, 0.26, 0.08, 0.09, 0.08, 0.12, 0.01, 0, 0, 0, 0, 0, 0, 0], "s-", lw=2.2, label="bio3 Isothermality")
ax2.plot(x2, [0.09, 0, 0, 0, 0.13, 0.16, 0.05, 0.13, 0.07, 0, 0, 0.01, 0.01, 0, 0, 0], "^-", lw=2.2, label="bio15 Precipitation seasonality")
ax2.set_title("Clouded Leopard · 18/20 points", fontsize=12)
ax2.set_xlabel("Query progress (k queries)")
ax2.set_ylabel("Feature guidance weight")
ax2.set_ylim(0, 0.55)
ax2.legend(fontsize=8.5, loc="upper right")
ax2.grid(alpha=0.3)

fig.suptitle("Automatic feature selection evolution: input 19 -> machine accumulates data -> automatically identifies significant features -> locks representative features",
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "feature_selection_evolution.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ================= Fig 2: Effect comparison =================
configs = ["Input 19\nno auto-select (v1)", "Input 19\n+auto-select (v2)", "Hand-picked 2 params\n(optimal control)"]
species = ["Black Muntjac", "Clouded Leopard"]
points = np.array([[14.25, 15.50, 18.15],
                   [7.65, 9.35, 9.60]])
queries = np.array([[18775, 16670, 13417],
                    [20000, 20000, 20000]])
dual = np.array([[65, 65, 95],
                 [50, 65, 75]])

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
x = np.arange(len(configs))
w = 0.32
colors = ["#4C78A8", "#F2A900"]

for a, mat, title, ylab in [
        (axes[0], points, "Target points found (target 20)", "Points"),
        (axes[1], queries, "Mean query count", "Queries"),
        (axes[2], dual, "Dual-cluster coverage (%)", "%")]:
    for si, sp in enumerate(species):
        a.bar(x + (si - 0.5) * w, mat[si], w, label=sp, color=colors[si])
        for xi, v in zip(x + (si - 0.5) * w, mat[si]):
            a.text(xi, v + mat.max() * 0.02, f"{v:.0f}", ha="center", fontsize=8.5)
    a.set_xticks(x)
    a.set_xticklabels(configs, fontsize=9)
    a.set_title(title, fontsize=11.5)
    a.set_ylabel(ylab)
    a.grid(alpha=0.3, axis="y")
axes[0].legend(fontsize=9)
fig.suptitle("Automatic feature selection effect: vs all-params-no-select / hand-picked optimal params (mean over 20 seeds)", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "auto_selection_vs_baselines.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("PLOTS OK ->", OUT)
