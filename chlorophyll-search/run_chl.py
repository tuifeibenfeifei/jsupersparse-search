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
"""run_chl.py — 叶绿素跨领域实验（黑盒在线目标搜索框架迁移）
数据：真实 CCI 叶绿素（孟加拉湾 2024-06），presence=291 高氯热点（>p99.5）
协变量：7 结构参数（sst_neg/sss/depth/par/wind/mld/coast_dist，物理规律合成）
       + 12 噪声参数（noise1-12）= 19 参数池
配置（20 seeds, target=20, 预算 20000）：
  CHL0_skeleton  无导向/无稀疏/无好奇（骨架）
  CHL1_full19    v1 全 19 参数导向
  CHL2_sparse    v2 AUC 稀疏（自动特征选择）
  CHL3_params2   v1 只 2 结构参数（sst_neg+sss）
  CHL4_params5   v1 只 5 结构参数（sst_neg/sss/depth/par/wind）
  CHL5_cluster   v2 稀疏 + 聚类感知导向 + 定向远跳（A10 口径）
输出：results/chl_search/chl_{config}_{detail,summary}.csv
"""
import csv
from pathlib import Path

from oracle import SpeciesOracle
from species_agent import SpeciesSearcher

OUT_DIR = Path(REPO_ROOT / "results/chl_search")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ORACLE = Path(REPO_ROOT / "chlorophyll-search/data/oracle.npz")

BASE = dict(
    target_points=20, max_queries=4000, min_dist_cells=1.0,
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
    ("CHL0_skeleton", dict(guidance_on=False, sparse_guidance=False, curiosity_weight=0.0), ALL),
    ("CHL1_full19", dict(guidance_on=True, sparse_guidance=False), ALL),
    ("CHL2_sparse", dict(guidance_on=True, sparse_guidance=True), ALL),
    ("CHL3_params2", dict(guidance_on=True, sparse_guidance=False), P2),
    ("CHL4_params5", dict(guidance_on=True, sparse_guidance=False), P5),
    ("CHL5_params7", dict(guidance_on=True, sparse_guidance=False), P7),
    ("CHL6_cluster", dict(guidance_on=True, sparse_guidance=True,
                          cluster_guidance=True, directed_jump=True), ALL),
    ("CHL7_adaptive", dict(guidance_on=True, sparse_guidance=True,
                           adaptive=True, directed_jump=True), ALL),
]

SEEDS = list(range(101, 121))


def main():
    oracle = SpeciesOracle(ORACLE)
    print(f"[oracle] {oracle.nrows}x{oracle.ncols}, presence={int(oracle.presence.sum())}, "
          f"{len(CONFIGS)} 配置 x {len(SEEDS)} seeds")
    detail_all, summary_rows = [], []
    for cname, over, covs in CONFIGS:
        cfg = dict(BASE, **over)
        cfg["cov_indices"] = covs
        pts, qrs, covs_, succ = [], [], [], 0
        for seed in SEEDS:
            res = SpeciesSearcher(oracle, cfg).run_episode(seed)
            pts.append(res["points_found"]); qrs.append(res["total_queries"])
            covs_.append(res["covered_cells"])
            succ += 1 if res["points_found"] >= 20 else 0
            detail_all.append([cname, seed, res["points_found"], res["total_queries"],
                               res["covered_cells"], round(res["covered_ratio"], 4)])
        n = len(SEEDS)
        mean = lambda x: sum(x) / n
        summary_rows.append([cname, round(mean(pts), 2), round(mean(qrs), 0),
                             round(mean(covs_), 0), round(100.0 * succ / n, 1)])
        print(f"  {cname:14s} 点 {summary_rows[-1][1]:6.2f} | 查询 {summary_rows[-1][2]:6.0f} | "
              f"覆盖 {summary_rows[-1][3]:6.0f} | 跑满 {summary_rows[-1][4]:5.1f}%")
    with open(OUT_DIR / "chl_detail.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "seed", "points_found", "queries", "covered_cells", "covered_ratio"])
        w.writerows(detail_all)
    with open(OUT_DIR / "chl_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "mean_points", "mean_queries", "mean_covered", "full_target_rate%"])
        w.writerows(summary_rows)
    print(f"[done] -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
