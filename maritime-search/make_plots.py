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
"""make_plots.py — 复跑 3 seeds 采集作图数据，出 4 张图（中文 matplotlib）。
图：
  1) 真漂移轨迹 + 发现点 + 已访问格叠图（seed101 为主，并排 seed102）
  2) 累计发现点数 vs 查询数（3 seeds）
  3) AUC 选出的协变量权重条形图
  4) 访问热力图（seed101）
"""
import sys
from pathlib import Path

ROOT = Path(REPO_ROOT)
sys.path.insert(0, str(ROOT / "species-search"))
sys.path.insert(0, str(ROOT / "maritime-search"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from species_agent import SpeciesSearcher            # noqa: E402
from build_maritime_oracle import MaritimeOracle     # noqa: E402
from run_maritime import CFG, SEEDS, NPZ             # noqa: E402

OUT = ROOT / "results" / "maritime_search"
oracle = MaritimeOracle(NPZ)

# ---- 复跑采集 ----
visited, found, conv = {}, {}, {}
for seed in SEEDS:
    s = SpeciesSearcher(oracle, CFG)
    r = s.run_episode(seed)
    vg = np.zeros((oracle.nrows, oracle.ncols), dtype=np.uint8)
    for (i, j) in s.visited_cells:
        if oracle.in_bounds(i, j):
            vg[i, j] = 1
    visited[seed] = vg
    found[seed] = r["found_cells"]
    conv[seed] = r["convergence"]

cov_names = oracle.cov_names
traj = oracle.traj                       # (601,2) km [x,y]
P0 = oracle.P0_cell                      # (i,j)
cell = oracle.cell_km

# ================= 图1：轨迹叠图（左 seed101 / 右 seed102）=================
fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
for ax, seed in zip(axes, [101, 102]):
    # 已访问格（散点，格中心 km）
    vi, vj = np.where(visited[seed] > 0)
    ax.scatter((vj + 0.5) * cell, (vi + 0.5) * cell, s=4, c="lightgray",
               alpha=0.5, label="已访问格")
    # 真漂移轨迹
    ax.plot(traj[:, 0], traj[:, 1], "-", c="tab:blue", lw=1.8, label="真漂移轨迹")
    ax.plot(traj[0, 0], traj[0, 1], "o", c="black", ms=8, label="失事点 P0")
    ax.plot(traj[-1, 0], traj[-1, 1], "D", c="tab:blue", ms=7, label="300h 后位置")
    # 发现点
    if found[seed]:
        fc = np.array(found[seed])
        ax.scatter((fc[:, 1] + 0.5) * cell, (fc[:, 0] + 0.5) * cell,
                   s=60, marker="*", c="red", edgecolors="k", zorder=5,
                   label=f"发现点 n={len(fc)}")
    nf = len(found[seed])
    ax.set_title(f"seed {seed}：发现 {nf} 点", fontsize=12)
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    ax.set_xlim(0, 300); ax.set_ylim(0, 300)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
axes[0].legend(loc="upper left", fontsize=8)
fig.suptitle("海上搜救：真漂移轨迹 + UAV 已访问格 + 发现点（物理仿真）",
             fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "fig1_trajectory_overlay.png", dpi=130)
plt.close(fig)

# ================= 图2：累计发现 vs 查询数 =================
fig, ax = plt.subplots(figsize=(8, 5))
colors = {101: "tab:gray", 102: "tab:red", 103: "tab:green"}
for seed in SEEDS:
    q = [c[0] for c in conv[seed]]
    f = [c[1] for c in conv[seed]]
    ax.plot(q, f, "-o", ms=3, color=colors[seed], label=f"seed {seed}")
ax.axhline(15, ls="--", c="k", alpha=0.5, label="目标 15 点")
ax.set_xlabel("查询数（架次，= 时间 0.5h/架次）")
ax.set_ylabel("累计发现点数")
ax.set_title("累计发现点数 vs 查询数（3 seeds）")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig2_cumulative_found.png", dpi=130)
plt.close(fig)

# ================= 图3：AUC 选出的协变量权重 =================
d = np.load(OUT / "_artifacts.npz", allow_pickle=True)
akeys = d["auc_keys"]; avals = d["auc_vals"]
fig, ax = plt.subplots(figsize=(9, 5))
labels = [cov_names[int(k)] for k in akeys]
order = np.argsort(avals)
bars = ax.barh([labels[i] for i in order], [avals[i] for i in order],
               color="teal")
for i, idx in enumerate(order):
    ax.text(avals[idx] + 0.005, i, f"{avals[idx]:.3f}", va="center", fontsize=9)
ax.set_xlabel("AUC 稀疏选参权重（(AUC-0.5)^2×4，跨 seed 均值）")
ax.set_title("AUC 在线选出的协变量（权重降序；仅 seed102 有足够正样本触发）")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig3_auc_selected.png", dpi=130)
plt.close(fig)

# ================= 图4：访问热力图（seed101）=================
fig, ax = plt.subplots(figsize=(7, 6))
hm = ax.imshow(visited[101], origin="lower", cmap="Blues",
               extent=[0, 300, 0, 300], alpha=0.6)
ax.plot(traj[:, 0], traj[:, 1], "-", c="tab:red", lw=1.8, label="真漂移轨迹")
ax.plot(traj[0, 0], traj[0, 1], "o", c="black", ms=8, label="P0")
ax.set_title("seed101 访问热力图（蓝=已探测格）——盲扫格网与漂移带未相遇")
ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
ax.legend(loc="upper left", fontsize=9)
fig.colorbar(hm, ax=ax, label="已探测")
fig.tight_layout()
fig.savefig(OUT / "fig4_visited_heatmap.png", dpi=130)
plt.close(fig)

print("figures done ->", OUT)
print("seed points:", {s: len(found[s]) for s in SEEDS})
