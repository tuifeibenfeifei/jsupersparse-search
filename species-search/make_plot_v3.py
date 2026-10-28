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
"""make_plot_v3.py — v2 vs v3 optimization comparison figure"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
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

OUT = Path(REPO_ROOT / "species-search/results")
OUT.mkdir(parents=True, exist_ok=True)

# metrics: points / covered grid cells / dual-cluster rate%   (v2, v3)
data = {
    "black muntjac": {"points": (15.50, 15.70), "covered": (1360, 2402), "dual": (65, 65)},
    "clouded leopard": {"points": (9.35, 7.95), "covered": (8449, 4770), "dual": (65, 35)},
}
labels = ["points found", "covered grid cells", "dual-cluster rate (%)"]
keys = ["points", "covered", "dual"]

fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6))
x = np.arange(2)
w = 0.30
colors = {"black muntjac": "#4C78A8", "clouded leopard": "#F2A900"}

for ax, key, lab in zip(axes, keys, labels):
    for si, sp in enumerate(["black muntjac", "clouded leopard"]):
        vals = data[sp][key]
        ax.bar(x + (si - 0.5) * w, vals, w, label=sp, color=colors[sp])
        for xi, v in zip(x + (si - 0.5) * w, vals):
            ax.text(xi, v + max(vals) * 0.03, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["v2\n(auto selection)", "v3\n(+ cold start + directed jump)"], fontsize=9.5)
    ax.set_title(lab, fontsize=12)
    ax.set_ylim(0, max(max(d[key]) for d in data.values()) * 1.25)
    ax.grid(alpha=0.3, axis="y")
axes[0].legend(fontsize=9.5, title="species")
fig.suptitle("v3 three optimizations (cold start + directed long jump + coverage metrics) vs v2 auto selection (mean over 20 seeds, input 19)",
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "v2_vs_v3_optimization.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("V3 PLOT OK ->", OUT)
