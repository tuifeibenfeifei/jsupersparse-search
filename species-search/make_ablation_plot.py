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
"""make_ablation_plot.py — 消融对比图（两物种 x 9 配置 x 4 指标）"""
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
    "A0_skeleton": "A0 最小骨架\n(无导向/无稀疏/无好奇)",
    "A1_full_guide": "A1 全参数导向\n(v1 params19)",
    "A2_sparse_v2base": "A2 自动选择\n(v2 基准)",
    "A3_sparse_cold150": "A3 +冷启动150",
    "A4_sparse_directed": "A4 +定向远跳",
    "A5_sparse_nocuriosity": "A5 无好奇心",
    "A6_sparse_topk4": "A6 topK=4",
    "A7_sparse_bins4": "A7 分箱=4",
    "A8_sparse_bins16": "A8 分箱=16",
    "A9_sparse_cluster": "A9 聚类导向",
    "A10_sparse_cluster_directed": "A10 聚类\n+定向远跳",
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
names = {"black_muntjac": "黑麂", "clouded_leopard": "云豹"}

for ax, key, title in [
        (axes[0], "points", "找到点数 (目标20)"),
        (axes[1], "queries", "平均查询次数"),
        (axes[2], "covered", "覆盖网格数"),
        (axes[3], "dual", "双集群覆盖率 (%)")]:
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
fig.suptitle("稀有物种线消融实验：导向 / 特征选择 / 冷启动 / 定向跳 / 好奇心 / topK / 分箱数 / 聚类感知（20 seeds 平均）",
             fontsize=13, y=1.03)
fig.tight_layout()
fig.savefig(OUT / "species_ablation_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("ABLATION PLOT OK ->", OUT)
