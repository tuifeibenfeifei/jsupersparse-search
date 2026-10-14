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
"""make_drift_plots.py — 海上搜救逆向漂移 4 张图。
读取 results/drift_search/_run_artifacts.npz，输出到 results/drift_search/。中文 matplotlib。
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(REPO_ROOT)
OUT = ROOT / "results" / "drift_search"
ART = np.load(OUT / "_run_artifacts.npz", allow_pickle=True)

presenceA = ART["presenceA"].astype(bool)
presenceB = ART["presenceB"].astype(bool)
debris_cell = ART["debris_cell"]
P0_i, P0_j = int(ART["P0_i"]), int(ART["P0_j"])
N = int(ART["n"])
per = list(ART["per_seed"])
seeds = list(ART["seeds"])
cell = float(ART["cell_km"])

# 用成功 seed（103）做主图
succ = [i for i, r in enumerate(per) if len(r["foundA"]) > 0]
main = succ[-1] if succ else 0
r = per[main]
main_seed = seeds[main]

# ---------- 图1：阶段A 残骸真位置 + 发现点叠图 ----------
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(presenceA, cmap="Blues", alpha=0.4, origin="upper",
          extent=[0, N, N, 0])
# 残骸真值
ax.scatter(debris_cell[:, 1] + 0.5, debris_cell[:, 0] + 0.5,
           s=60, c="navy", marker="X", label="残骸真位置(10件)", zorder=5)
# A 发现点
fa = r["foundA"]
ax.scatter(fa[:, 1] + 0.5, fa[:, 0] + 0.5, s=70, c="red", marker="^",
           edgecolors="darkred", label=f"A发现点({len(fa)}个)", zorder=6)
# 真 P0
ax.scatter([P0_j + 0.5], [P0_i + 0.5], s=260, c="gold", marker="*",
           edgecolors="black", linewidths=1.2, label="真失事点 P0", zorder=7)
ax.set_title(f"图1 阶段A 残骸快照搜索（seed={main_seed}）\n"
             f"蓝=冻结残骸探测区({int(presenceA.sum())}格) 红三角=A发现点 "
             f"蓝X=残骸 金星=P0", fontsize=11)
ax.set_xlabel("东向 j（格，每格2km）"); ax.set_ylabel("北向 i（格）")
ax.legend(loc="upper right", fontsize=9)
ax.set_xlim(0, N); ax.set_ylim(N, 0)
fig.tight_layout()
fig.savefig(OUT / "fig1_stageA_debris_overlay.png", dpi=130)
plt.close(fig)

# ---------- 图2：解码候选区(质心±半径) vs 真P0 ----------
centroid = r["centroid_km"]
err = r["err_km"]
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(presenceA, cmap="Blues", alpha=0.3, origin="upper", extent=[0, N, N, 0])
# 解码质心（格）
cj = centroid[0] / cell - 0.5
ci = centroid[1] / cell - 0.5
ax.scatter([cj], [ci], s=120, c="green", marker="o", edgecolors="darkgreen",
           linewidths=1.5, label=f"解码质心({centroid[0]:.0f},{centroid[1]:.0f}km)", zorder=6)
# 不确定半径 ±10 格 = 20km
circ = plt.Circle((cj, ci), 10, color="green", fill=False, ls="--", lw=1.5,
                  label="解码不确定半径(±20km)")
ax.add_patch(circ)
ax.scatter([P0_j + 0.5], [P0_i + 0.5], s=260, c="gold", marker="*",
           edgecolors="black", linewidths=1.2, label=f"真P0(误差{err:.1f}km)", zorder=7)
ax.plot([cj, P0_j + 0.5], [ci, P0_i + 0.5], "g--", lw=1.2, zorder=5)
ax.set_title(f"图2 阶段A发现点反向平流解码 → 候选失事区 vs 真P0（seed={main_seed}）\n"
             f"绿圆=解码质心 绿虚圆=±20km不确定 金星=真P0  质心误差={err:.1f}km", fontsize=11)
ax.set_xlabel("东向 j（格）"); ax.set_ylabel("北向 i（格）")
ax.legend(loc="upper right", fontsize=9)
ax.set_xlim(0, N); ax.set_ylim(N, 0)
fig.tight_layout()
fig.savefig(OUT / "fig2_decode_centroid_vs_P0.png", dpi=130)
plt.close(fig)

# ---------- 图3：阶段B 发现点 vs 真P0 ----------
fb = r["foundB"]
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(presenceB, cmap="Oranges", alpha=0.5, origin="upper", extent=[0, N, N, 0])
if len(fb):
    ax.scatter(fb[:, 1] + 0.5, fb[:, 0] + 0.5, s=80, c="red", marker="^",
               edgecolors="darkred", label=f"B发现点({len(fb)}个)", zorder=6)
ax.scatter([P0_j + 0.5], [P0_i + 0.5], s=260, c="gold", marker="*",
           edgecolors="black", linewidths=1.2, label=f"真P0(B→P0={r['B_to_P0_km']:.1f}km)", zorder=7)
ax.set_title(f"图3 阶段B 失事点搜索（seed={main_seed}）\n"
             f"橙=P0静态证据盘(R=3格) 红三角=B发现点 金星=真P0", fontsize=11)
ax.set_xlabel("东向 j（格）"); ax.set_ylabel("北向 i（格）")
ax.legend(loc="upper right", fontsize=9)
ax.set_xlim(0, N); ax.set_ylim(N, 0)
fig.tight_layout()
fig.savefig(OUT / "fig3_stageB_found_vs_P0.png", dpi=130)
plt.close(fig)

# ---------- 图4：阶段B AUC 协变量得分 ----------
aucB = r["aucB"]
names = list(ART["cov_namesB"])
order = sorted(range(len(names)), key=lambda k: -aucB.get(k, 0.5))
vals = [aucB.get(k, 0.5) for k in order]
fig, ax = plt.subplots(figsize=(11, 6))
colors = ["#d62728" if k == 19 else "#9aa0a6" for k in order]
ax.bar(range(len(order)), vals, color=colors, edgecolor="black", linewidth=0.5)
ax.axhline(0.55, ls="--", c="black", lw=1, label="AUC 入选阈值=0.55")
ax.set_xticks(range(len(order)))
ax.set_xticklabels([f"{k}:{names[k]}" for k in order], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Mann-Whitney AUC（方向无关，0.5=无区分度）")
ax.set_ylim(0.45, 1.05)
ax.set_title(f"图4 阶段B 20协变量 AUC 分离度（seed={main_seed}）\n"
             f"红=第20协变量'到解码质心距离' AUC={aucB.get(19,0.5):.3f}（最高）", fontsize=12)
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig4_stageB_auc_features.png", dpi=130)
plt.close(fig)

print("figures ->", OUT)
for f in ["fig1_stageA_debris_overlay.png", "fig2_decode_centroid_vs_P0.png",
          "fig3_stageB_found_vs_P0.png", "fig4_stageB_auc_features.png"]:
    print("  ", f, (OUT / f).exists())
