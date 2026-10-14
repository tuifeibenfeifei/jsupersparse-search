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
"""make_chl_plot.py — 叶绿素跨领域实验对比图"""
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
    "CHL0_skeleton": "CHL0 无参数骨架",
    "CHL1_full19": "CHL1 全19参数导向\n(7结构+12噪声)",
    "CHL2_sparse": "CHL2 AUC自动选择",
    "CHL3_params2": "CHL3 2结构参数\n(sst+sss)",
    "CHL4_params5": "CHL4 5结构参数",
    "CHL5_params7": "CHL5 7结构参数",
    "CHL6_cluster": "CHL6 聚类+分散远跳",
    "CHL7_adaptive": "CHL7 自适应开关\n(在线诊断)",
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
        (axes[0], "points", "找到点数 (目标20，预算4000)", "#2E7D32"),
        (axes[1], "queries", "平均查询次数", "#1565C0"),
        (axes[2], "succ", "跑满20点率 (%)", "#6A1B9A")]:
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
fig.suptitle("叶绿素跨领域实验：真实CCI高氯热点搜索（20 seeds 平均；7结构参数+12噪声）",
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "chl_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("CHL PLOT OK ->", OUT)
