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
"""run_ocean3d.py — 3D 跨领域实验（黑盒在线目标搜索框架 → 三维水体）
数据：真实 EMODnet DIVAnd 4D 溶解氧（地中海西部，0.25°，29 层 × 33 纬 × 73 经）
目标：中层氧最低带低氧核（DO<=170 umol/l，1779 格 ≈ 6%）
协变量：3 真实几何 + 4 物理规律合成代理 + 12 噪声 = 19
配置（20 seeds, target=20, 预算 10000）：
  O3D0_skeleton  无导向/无稀疏/无好奇（骨架）
  O3D1_full19    v1 全 19 参数导向
  O3D2_sparse    v2 AUC 稀疏（自动特征选择）
  O3D3_params2   v1 只 2 结构参数
  O3D4_params5   v1 只 5 结构参数
  O3D5_params7   v1 只 7 结构参数
  O3D6_cluster   v2 稀疏 + 聚类感知 + 定向球壳远跳
  O3D7_adaptive  v2 稀疏 + A11 自适应 + 定向球壳远跳
输出：results/ocean3d/o3d_{detail,summary}.csv
"""
import csv
import sys
from pathlib import Path

from oracle3d import Oracle3D
from species_agent_3d import Searcher3D, random_baseline, grid_baseline

OUT_DIR = Path(REPO_ROOT / "results/ocean3d")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ORACLE = Path(REPO_ROOT / "ocean3d-search/data/oracle3d.npz")

BASE = dict(
    target_points=40, max_queries=10000, min_dist_cells=1.0,
    stage2_min=3, stage3_min=5, stage3_jump_interval=30,
    stage3_jump_min_ratio=1.25, stage3_jump_max_ratio=5.0,
    curiosity_weight=1.0, learning_rate=0.1, discount_factor=0.95,
    epsilon_start=0.9, epsilon_end=0.1, epsilon_decay=0.999,
    stage1_grid_step=3, guidance_on=True, cov_in_state=True,
    recalc_every=20, stage2_topk=8,
    sparse_guidance=True, sparse_topk=6, sparse_min_pos=5,
)

ALL = list(range(19))
P2 = [0, 1]
P5 = [0, 1, 2, 3, 4]
P7 = list(range(7))

CONFIGS = [
    ("O3D0_skeleton", dict(guidance_on=False, sparse_guidance=False, curiosity_weight=0.0), ALL),
    ("O3D1_full19", dict(guidance_on=True, sparse_guidance=False), ALL),
    ("O3D2_sparse", dict(guidance_on=True, sparse_guidance=True), ALL),
    ("O3D3_params2", dict(guidance_on=True, sparse_guidance=False), P2),
    ("O3D4_params5", dict(guidance_on=True, sparse_guidance=False), P5),
    ("O3D5_params7", dict(guidance_on=True, sparse_guidance=False), P7),
    ("O3D6_cluster", dict(guidance_on=True, sparse_guidance=True,
                          cluster_guidance=True, directed_jump=True), ALL),
    ("O3D7_adaptive", dict(guidance_on=True, sparse_guidance=True,
                           adaptive=True, directed_jump=True), ALL),
]

BASELINES = [
    ("O3D_RANDOM", "random"),
    ("O3D_GRID", "grid"),
]

SEEDS = list(range(101, 121))


def main():
    oracle = Oracle3D(ORACLE)
    print(f"[oracle] {oracle.K}x{oracle.L}x{oracle.J} = {oracle.total_cells} 单元, "
          f"presence={int(oracle.presence.sum())}, {len(CONFIGS)} 配置 x {len(SEEDS)} seeds")
    detail_all, summary_rows = [], []
    for cname, over, covs in CONFIGS:
        cfg = dict(BASE, **over)
        cfg["cov_indices"] = covs
        pts, qrs, covs_, succ = [], [], [], 0
        for seed in SEEDS:
            res = Searcher3D(oracle, cfg).run_episode(seed)
            pts.append(res["points_found"]); qrs.append(res["total_queries"])
            covs_.append(res["covered_cells"])
            succ += 1 if res["points_found"] >= 40 else 0
            detail_all.append([cname, seed, res["points_found"], res["total_queries"],
                               res["covered_cells"], round(res["covered_ratio"], 4)])
        n = len(SEEDS)
        mean = lambda x: sum(x) / n
        summary_rows.append([cname, round(mean(pts), 2), round(mean(qrs), 0),
                             round(mean(covs_), 0), round(100.0 * succ / n, 1)])
        print(f"  {cname:16s} 点 {summary_rows[-1][1]:6.2f} | 查询 {summary_rows[-1][2]:6.0f} | "
              f"覆盖 {summary_rows[-1][3]:6.0f} | 跑满 {summary_rows[-1][4]:5.1f}%", flush=True)

    # ---- 3D 基线（random / grid） ----
    base_cfg = dict(target_points=40, max_queries=10000, min_dist_cells=1.0,
                    grid_step=int(BASE["stage1_grid_step"]))
    for cname, kind in BASELINES:
        pts, qrs, covs_, succ = [], [], [], 0
        for seed in SEEDS:
            if kind == "random":
                res = random_baseline(oracle, base_cfg, seed)
            else:
                res = grid_baseline(oracle, base_cfg, seed)
            pts.append(res["points_found"]); qrs.append(res["total_queries"])
            covs_.append(res["total_queries"])
            succ += 1 if res["points_found"] >= 40 else 0
            detail_all.append([cname, seed, res["points_found"], res["total_queries"],
                               res["total_queries"], round(res["total_queries"] / oracle.total_cells, 4)])
        n = len(SEEDS)
        mean = lambda x: sum(x) / n
        summary_rows.append([cname, round(mean(pts), 2), round(mean(qrs), 0),
                             round(mean(covs_), 0), round(100.0 * succ / n, 1)])
        print(f"  {cname:16s} 点 {summary_rows[-1][1]:6.2f} | 查询 {summary_rows[-1][2]:6.0f} | "
              f"覆盖 {summary_rows[-1][3]:6.0f} | 跑满 {summary_rows[-1][4]:5.1f}%", flush=True)
    with open(OUT_DIR / "o3d_detail.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "seed", "points_found", "queries", "covered_cells", "covered_ratio"])
        w.writerows(detail_all)
    with open(OUT_DIR / "o3d_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "mean_points", "mean_queries", "mean_covered", "full_target_rate%"])
        w.writerows(summary_rows)
    print(f"[done] -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
