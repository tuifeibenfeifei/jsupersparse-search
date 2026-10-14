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
"""make_plots.py — 生成稀有物种自动特征选择可视化 PNG
输出到 species-search\results\：
  feature_selection_evolution.png     两物种特征权重演化（分散→锁定）
  auto_selection_vs_baselines.png     v1全参数 / v2自动选择 / 手工2参数 效果对比
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

# ---- 中文字体 ----
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

# ================= 图1：特征选择演化 =================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

# 黑麂 seed 111（4330 查询收敛，20/20）
x1 = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.3])  # k queries
ax1.plot(x1, [0, 0, 0.05, 0.11, 0.16, 0.33, 0.41], "o-", lw=2.2, label="bio6 最冷月最低温")
ax1.plot(x1, [0, 0, 0, 0, 0.13, 0.30, 0.39], "s-", lw=2.2, label="bio11 最冷月均温")
ax1.plot(x1, [0, 0, 0, 0, 0.10, 0.29, 0.39], "^-", lw=2.2, label="bio9 最干季均温")
ax1.plot(x1, [0.45, 0.35, 0.25, 0.23, 0.12, 0, 0], "x--", lw=2.0, color="gray", label="bio12 年降水")
ax1.plot(x1, [0.41, 0.30, 0.20, 0.18, 0.16, 0, 0], "d--", lw=2.0, color="lightgray", label="bio18 最暖季降水")
ax1.set_title("黑麂 (Black Muntjac) · 20/20 点, 4330 查询收敛", fontsize=12)
ax1.set_xlabel("查询进度 (k queries)")
ax1.set_ylabel("特征导向权重")
ax1.set_ylim(0, 0.52)
ax1.legend(fontsize=8.5, loc="upper right")
ax1.grid(alpha=0.3)

# 云豹 seed 118（18/20）
x2 = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
ax2.plot(x2, [0.11, 0.49, 0.48, 0.40, 0.32, 0.27, 0.14, 0.12, 0.16, 0.14, 0.09, 0.07, 0.02, 0, 0, 0], "o-", lw=2.2, label="bio2 年均日温差")
ax2.plot(x2, [0.28, 0.34, 0.44, 0.26, 0.08, 0.09, 0.08, 0.12, 0.01, 0, 0, 0, 0, 0, 0, 0], "s-", lw=2.2, label="bio3 等温性")
ax2.plot(x2, [0.09, 0, 0, 0, 0.13, 0.16, 0.05, 0.13, 0.07, 0, 0, 0.01, 0.01, 0, 0, 0], "^-", lw=2.2, label="bio15 降水季节性")
ax2.set_title("云豹 (Clouded Leopard) · 18/20 点", fontsize=12)
ax2.set_xlabel("查询进度 (k queries)")
ax2.set_ylabel("特征导向权重")
ax2.set_ylim(0, 0.55)
ax2.legend(fontsize=8.5, loc="upper right")
ax2.grid(alpha=0.3)

fig.suptitle("自动特征选择演化：输入 19 个 → 机器积累数据 → 自动判定显著特征 → 锁定代表特征",
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "feature_selection_evolution.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ================= 图2：效果对比 =================
configs = ["输入19\n不自动选 (v1)", "输入19\n+自动选 (v2)", "手工2参数\n(最优对照)"]
species = ["黑麂", "云豹"]
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
        (axes[0], points, "找到目标点数 (目标20)", "点数"),
        (axes[1], queries, "平均查询次数", "查询次数"),
        (axes[2], dual, "双集群覆盖率 (%)", "%")]:
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
fig.suptitle("自动特征选择效果：vs 全参数不选 / 手工最优参数（20 seeds 平均）", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "auto_selection_vs_baselines.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("PLOTS OK ->", OUT)
