# -*- coding: utf-8 -*-
"""plot_timevary_v5.py — 五轮时变重测图（matplotlib 直出，中文字体 Microsoft YaHei/SimHei）。
数据源：results/timevary_search/v5/（per_seed_*.csv、summary.csv、featurelog_*）；
v4 对照：results/timevary_search/v4/fix/summary.csv 数字。
输出：results/timevary_search/v5/fig_*.png
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
TARGET = 20


def load_per_seed(scene, variant):
    rows = []
    with open(HERE / f"per_seed_{scene}_{variant}.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((int(r["seed"]), int(r["points_found"]),
                         int(r["total_queries"])))
    return rows


def fig_moving_queries():
    variants = ["framework_off", "framework_decay300", "framework_fix_motion",
                "framework_fix_follow", "framework_fix_follow_disperse"]
    labels = ["off", "decay300", "fix_motion\n(v4推荐)", "fix+follow", "fix+follow\n+disperse"]
    qs, pts = [], []
    for v in variants:
        rows = load_per_seed("moving_plume", v)
        qs.append(np.mean([r[2] for r in rows]))
        pts.append(np.mean([r[1] for r in rows]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#9aa5b1", "#7fb2d6", "#e8a23a", "#2e9e6b", "#1f7a4d"]
    bars = ax.bar(range(len(variants)), qs, color=colors, width=0.62)
    for i, (q, p) in enumerate(zip(qs, pts)):
        ax.text(i, q + 18, f"q={q:.0f}\npts={p:.1f}", ha="center", va="bottom",
                fontsize=9)
    ax.axhline(377, ls="--", c="#c0392b", lw=1.2)
    ax.text(4.1, 377, "爬山 q377", va="center", fontsize=9, color="#c0392b")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("查询数（10 seeds 均值）")
    ax.set_title("慢漂移(moving_plume)：五轮跟随模式查询压缩 784.8→394.6 (−49.7%)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "fig_moving_queries.png", dpi=150)
    plt.close(fig)


def fig_abrupt_points():
    variants = ["framework_decay300", "framework_fix_motion", "framework_fix_disperse",
                "framework_fix_follow", "framework_fix_follow_disperse"]
    labels = ["decay300", "fix_motion\n(v4推荐)", "fix+disperse", "fix+follow", "fix+follow\n+disperse"]
    means, stds = [], []
    for v in variants:
        rows = load_per_seed("abrupt_jump", v)
        p = np.array([r[1] for r in rows], float)
        means.append(p.mean()); stds.append(p.std())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(variants))
    ax.bar(x, means, yerr=stds, capsize=4, color=["#7fb2d6", "#e8a23a", "#b3a1e0",
                                                  "#5cb8a0", "#1f7a4d"], width=0.62)
    ax.axhline(19, ls="--", c="#c0392b", lw=1.2)
    ax.text(4.15, 19, "验收线 19", va="center", fontsize=9, color="#c0392b")
    for i, m in enumerate(means):
        ax.text(i, m + 0.12, f"{m:.1f}±{stds[i]:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("点数（10 seeds 均值±std，目标20）")
    ax.set_title("突变(abrupt_jump)：五轮组合 19.3→19.5，seed107 修复至 20")
    ax.set_ylim(14, 21)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "fig_abrupt_points.png", dpi=150)
    plt.close(fig)


def fig_seed107_traj():
    # v4 fix_motion 无存档 featurelog；用当前代码 v4 等价配置（默认关=逐位等价）重跑结果代替
    rows_v4 = _read_feat(HERE / "featurelog_abrupt_jump_framework_fix_motion_seed107.csv")
    rows_v5 = _read_feat(HERE / "featurelog_abrupt_jump_framework_fix_follow_disperse_seed107.csv")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot([r[0] for r in rows_v4], [r[1] for r in rows_v4], "o-", ms=3,
            label="v4 fix_motion seed107 (终态15点)", color="#e8a23a")
    ax.plot([r[0] for r in rows_v5], [r[1] for r in rows_v5], "o-", ms=3,
            label="v5 fix+follow+disperse seed107 (终态20点)", color="#1f7a4d")
    ax.axvline(750, ls=":", c="#555", lw=1)
    ax.text(755, 42, "突变点 q=750", fontsize=9, color="#555")
    ax.set_xlabel("查询数"); ax.set_ylabel("n_pos（密度图有效正样本加权和）")
    ax.set_title("seed107 突变后 n_pos 轨迹：v5 组合终态 20 点（v4 fix_motion 15 点）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "fig_seed107_traj.png", dpi=150)
    plt.close(fig)


def _read_feat(p):
    rows = []
    with open(p, encoding="utf-8") as f:
        rd = csv.reader(f)
        next(rd, None)
        for r in rd:
            rows.append((int(r[0]), int(r[1])))
    return rows


def fig_moving_per_seed():
    base = load_per_seed("moving_plume", "framework_fix_motion")
    fol = load_per_seed("moving_plume", "framework_fix_follow")
    seeds = [r[0] for r in base]
    qb = [r[2] for r in base]; qf = [r[2] for r in fol]
    x = np.arange(len(seeds))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - 0.2, qb, width=0.38, label="v4 fix_motion", color="#e8a23a")
    ax.bar(x + 0.2, qf, width=0.38, label="v5 fix+follow", color="#2e9e6b")
    ax.set_xticks(x); ax.set_xticklabels(seeds)
    ax.set_xlabel("seed"); ax.set_ylabel("查询数")
    ax.set_title("慢漂移逐 seed：跟随模式 10/10 保 20 点，查询全压至 ~330-500")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "fig_moving_per_seed.png", dpi=150)
    plt.close(fig)


def main():
    fig_moving_queries()
    fig_abrupt_points()
    fig_seed107_traj()
    fig_moving_per_seed()
    print("[done] figures ->", HERE)


if __name__ == "__main__":
    main()
