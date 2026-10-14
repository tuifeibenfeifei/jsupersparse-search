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
"""任务二绘图：基于 diagnose.py 产出的 _artifacts.npz + baseline.csv 画中文图。
图 a) 漂移走廊 + seed101 visited 二值图 + 发现点；
图 b) 逐 seed 命中数随 query 时刻曲线。
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

OUT = Path(REPO_ROOT / "results/maritime_search/bootstrap_analysis_v3")
art = np.load(OUT / "_artifacts.npz", allow_pickle=True)

traj = art["traj"]                 # (MAX_Q+1, 2) km [x, y]
cell_km = float(art["cell_km"])
R_km = float(art["R_km"])
nrows = int(art["nrows"]); ncols = int(art["ncols"])
P0_cell = tuple(int(v) for v in art["P0_cell"])
visited101 = art["visited101"]     # (nrows,ncols) 0/1
found101 = art["found101"]         # (K,2) 行/列
seeds = [int(s) for s in art["seeds"]]

# ---- 图 a: 漂移走廊 + visited101 + 发现点 ----
fig, ax = plt.subplots(figsize=(9, 8))
# visited 二值图（黑色=已访问）
ax.imshow(visited101, cmap="Greys", origin="lower",
          extent=[0, ncols, 0, nrows], alpha=0.55, aspect="equal")
# 漂移轨迹 km -> 格坐标
tj = traj[:, 0] / cell_km - 0.5
ti = traj[:, 1] / cell_km - 0.5
ax.plot(tj, ti, color="tab:red", lw=1.8, label="目标漂移轨迹(走廊)")
# 起点/终点
ax.scatter([tj[0]], [ti[0]], color="tab:green", s=80, zorder=5, marker="o", label="P0 落水点")
ax.scatter([tj[-1]], [ti[-1]], color="tab:orange", s=80, zorder=5, marker="*", label="轨迹末点")
# 探测半径带（沿轨迹 ±R_cell）
R_cell = R_km / cell_km
ax.fill_between(tj, ti - R_cell, ti + R_cell, color="tab:red", alpha=0.12,
                label=f"探测半径带 ±{R_cell:.0f}格(={R_km:.0f}km)")
# 发现点
if len(found101) > 0:
    ax.scatter(found101[:, 1], found101[:, 0], color="cyan", edgecolor="k",
               s=70, zorder=6, marker="X", label=f"seed101 发现点(n={len(found101)})")
ax.set_xlim(0, ncols); ax.set_ylim(0, nrows)
ax.set_xlabel("列 j (格)"); ax.set_ylabel("行 i (格)")
ax.set_title("海上搜救 seed101：Stage1 盲扫列带 vs 漂移走廊几何关系")
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "fig_a_trajectory_overlay.png", dpi=130)
plt.close()
print("fig_a -> fig_a_trajectory_overlay.png", flush=True)

# ---- 图 b: 逐 seed 命中数随时间 ----
fig, ax = plt.subplots(figsize=(10, 6))
for seed in seeds:
    q = art[f"q_{seed}"]; f = art[f"f_{seed}"]
    ax.plot(q, f, lw=1.4, marker=".", ms=4, label=f"seed{seed}")
ax.axhline(15, color="k", ls="--", lw=1, alpha=0.5, label="target=15")
ax.set_xlabel("query 时刻 (=时间步, 每步0.5h)")
ax.set_ylabel("累计发现点数")
ax.set_title("逐 seed 命中数随时间曲线（600 预算撞满 vs 漂移目标）")
ax.legend(ncol=5, fontsize=8, loc="upper center")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "fig_b_convergence_curves.png", dpi=130)
plt.close()
print("fig_b -> fig_b_convergence_curves.png", flush=True)
print("PLOT DONE", flush=True)
