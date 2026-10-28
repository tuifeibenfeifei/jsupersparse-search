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
"""make_chl_plot.py — chlorophyll cross-domain experiment comparison figure"""
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

SRC = Path(REPO_ROOT / "results/chl_search")
OUT = Path(REPO_ROOT / "chlorophyll-search/results")
OUT.mkdir(parents=True, exist_ok=True)

NAME_MAP = {
    "CHL0_skeleton": "CHL0 no-parameter skeleton",
    "CHL1_full19": "CHL1 full 19-param guidance\n(7 structural + 12 noise)",
    "CHL2_sparse": "CHL2 AUC auto selection",
    "CHL3_params2": "CHL3 2 structural params\n(sst+sss)",
    "CHL4_params5": "CHL4 5 structural params",
    "CHL5_params7": "CHL5 7 structural params",
    "CHL6_cluster": "CHL6 cluster + dispersed long jump",
    "CHL7_adaptive": "CHL7 adaptive switch\n(online diagnostic)",
}

rows = {}
with open(SRC / "chl_summary.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows[r["config"]] = {
            "points": float(r["mean_points"]),
            "queries": float(r["mean_queries"]),
            "covered": float(r["mean_covered"]),
            "succ": float(r["full_target_rate%"]),
        }

configs = [c for c in NAME_MAP]
labels = [NAME_MAP[c] for c in configs]
x = np.arange(len(configs))

fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.4))
for ax, key, title, color in [
        (axes[0], "points", "points found (target 20, budget 4000)", "#2E7D32"),
        (axes[1], "queries", "mean number of queries", "#1565C0"),
        (axes[2], "succ", "rate of reaching 20 points (%)", "#6A1B9A")]:
    vals = [rows[c][key] for c in configs]
    bars = ax.bar(x, vals, 0.52, color=color)
    for xi, v in zip(x, vals):
        ax.text(xi, v + max(vals) * 0.03, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0, max(vals) * 1.2)
    ax.grid(alpha=0.3, axis="y")
    bars[0].set_label("")
fig.suptitle("Chlorophyll cross-domain experiment: real CCI high-chlorophyll hotspot search (mean over 20 seeds; 7 structural params + 12 noise)",
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "chl_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("CHL PLOT OK ->", OUT)
