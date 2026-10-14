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
"""finalize_baselines.py — 汇总框架+3基线、画中文柱状图、写 baseline_report.md。
框架数字直接引用各 regress_v2 CSV（不重跑）。"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(REPO_ROOT / "results/baseline_compare")

# 框架数字（来源：各场景 regress_v2 summary CSV，seeds 101-110）
# (显示名, target, 框架 mean_points, 框架 mean_queries, 备注)
FRAMEWORK = {
    "pollution": ("污染溯源", 15, 15.0, 185.4, "regress_v2 points/total_queries"),
    "maritime": ("海上搜救", 15, 1.0, 600.0, "regress_v2 points/total_queries"),
    "driftA": ("逆向漂移·阶段A", 6, 2.8, 397.5, "regress_v2 A_points/A_queries"),
    "mineralA": ("矿化露头·阶段A", 10, 10.0, 255.5, "regress_v2 A_points/A_queries"),
}
ORDER = ["pollution", "maritime", "driftA", "mineralA"]
METHODS = ["random", "grid", "hill_climb"]
M_LABEL = {"framework": "改进框架", "random": "随机基线", "grid": "栅格基线", "hill_climb": "爬山基线"}

# 读基线汇总
base = {}  # (skey, method) -> (mean_points, mean_queries, hit_rate)
with open(OUT / "baseline_summary.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        base[(r["scenario_key"], r["method"])] = (
            float(r["mean_points_found"]), float(r["mean_total_queries"]),
            float(r["hit_rate"]))

# ---- 合并对比 CSV ----
combined = OUT / "combined_compare.csv"
with open(combined, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["scenario_key", "scenario", "target", "method",
                "mean_points_found", "mean_total_queries", "hit_rate", "source"])
    for k in ORDER:
        sname, target, fp, fq, _ = FRAMEWORK[k]
        w.writerow([k, sname, target, "framework", fp, fq,
                    f"{fp/target:.3f}", "regress_v2 (已跑,引用)"])
        for m in METHODS:
            mp, mq, hr = base[(k, m)]
            w.writerow([k, sname, target, m, mp, mq, f"{hr:.3f}", "本次 baseline_compare"])

# ---- 每场景柱状图（4 方法 points_found）----
for k in ORDER:
    sname, target, fp, fq, _ = FRAMEWORK[k]
    names = ["framework"] + METHODS
    vals = [fp] + [base[(k, m)][0] for m in METHODS]
    qvals = [fq] + [base[(k, m)][1] for m in METHODS]
    labels = [M_LABEL[n] for n in names]
    colors = ["#2c7fb8", "#9ecae1", "#bdbdbd", "#fdae6b"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", lw=0.6)
    ax.axhline(target, ls="--", color="red", lw=1, label=f"目标={target}")
    for b, v, q in zip(bars, vals, qvals):
        ax.text(b.get_x() + b.get_width()/2, v + target*0.02,
                f"{v:.1f}\n({q:.0f}查询)", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("平均发现点数 points_found")
    ax.set_title(f"{sname}：4 方法对比（seeds 101-110 均值；柱上为平均查询数）")
    ax.set_ylim(0, target * 1.25)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"bar_{k}.png", dpi=130)
    plt.close(fig)

print("combined + bars done ->", combined)
