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
"""run_ablation_species.py — 稀有物种线详细消融实验（两物种 × 9 配置 × 20 seeds）
对比维度：
  A0 skeleton          导向 OFF + 稀疏 OFF + 好奇心 0（最小骨架）
  A1 full_guide       导向 ON + 稀疏 OFF（全 19 参数导向，= v1 params19）
  A2 sparse (v2 基准)  自动特征选择（AUC 稀疏）
  A3 sparse+cold150    冷启动阈值 150
  A4 sparse+directed   定向远跳
  A5 sparse+no_curi    好奇心权重 0
  A6 sparse+topk4      Stage2 候选 topK=4
  A7 sparse+bins4      分箱数 4
  A8 sparse+bins16     分箱数 16
  A9 sparse+cluster    聚类感知导向（多模态候选按簇分散）
  A10 sparse+cluster+directed  聚类感知导向 + 定向远跳（远跳按簇分散）
输出：results/species_search/ablation_species/
  ablation_species_{species}_detail.csv   每 seed 明细
  ablation_species_{species}_summary.csv  每配置聚合（点数/查询/覆盖/双集群/成功率）
用法：python run_ablation_species.py <oracle_path> <species> <A_boxes> <B_boxes> [n_seeds]
"""
import csv
import sys
from pathlib import Path

from oracle import SpeciesOracle
from species_agent import SpeciesSearcher

OUT_DIR = Path(REPO_ROOT / "results/species_search/ablation_species")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE = dict(
    target_points=20, max_queries=20000, min_dist_cells=1.0,
    stage2_min=3, stage3_min=5, stage3_jump_interval=30,
    stage3_jump_min_ratio=1.25, stage3_jump_max_ratio=5.0,
    curiosity_weight=1.0, learning_rate=0.1, discount_factor=0.95,
    epsilon_start=0.9, epsilon_end=0.1, epsilon_decay=0.999,
    stage1_grid_step=2, guidance_on=True, cov_in_state=True,
    recalc_every=20, stage2_topk=8,
    sparse_guidance=True, sparse_topk=6, sparse_min_pos=5,
)

# 9 个消融配置（在 BASE 上改）
CONFIGS = [
    ("A0_skeleton",        dict(guidance_on=False, sparse_guidance=False, curiosity_weight=0.0)),
    ("A1_full_guide",      dict(guidance_on=True,  sparse_guidance=False)),
    ("A2_sparse_v2base",   dict(guidance_on=True,  sparse_guidance=True)),
    ("A3_sparse_cold150",  dict(guidance_on=True,  sparse_guidance=True, cold_start_samples=150)),
    ("A4_sparse_directed", dict(guidance_on=True,  sparse_guidance=True, directed_jump=True)),
    ("A5_sparse_nocuriosity", dict(guidance_on=True, sparse_guidance=True, curiosity_weight=0.0)),
    ("A6_sparse_topk4",    dict(guidance_on=True,  sparse_guidance=True, stage2_topk=4)),
    ("A7_sparse_bins4",    dict(guidance_on=True,  sparse_guidance=True, n_bins=4)),
    ("A8_sparse_bins16",   dict(guidance_on=True,  sparse_guidance=True, n_bins=16)),
    ("A9_sparse_cluster",  dict(guidance_on=True,  sparse_guidance=True, cluster_guidance=True)),
    ("A10_sparse_cluster_directed", dict(guidance_on=True, sparse_guidance=True,
                                         cluster_guidance=True, directed_jump=True)),
]


def in_cluster(cells, boxes):
    for i, j in cells:
        for (imin, imax, jmin, jmax) in boxes:
            if imin <= i <= imax and jmin <= j <= jmax:
                return True
    return False


def main():
    if len(sys.argv) < 5:
        print("usage: run_ablation_species.py <oracle_path> <species> <A_boxes> <B_boxes> [n_seeds]")
        return 1
    oracle_path = Path(sys.argv[1])
    species = sys.argv[2]
    A_boxes = [tuple(int(v) for v in b.split(",")) for b in sys.argv[3].split("|")]
    B_boxes = [tuple(int(v) for v in b.split(",")) for b in sys.argv[4].split("|")]
    n_seeds = int(sys.argv[5]) if len(sys.argv) > 5 else 20
    seeds = list(range(101, 101 + n_seeds))

    oracle = SpeciesOracle(oracle_path)
    print(f"[oracle] {oracle.nrows}x{oracle.ncols}, presence={int(oracle.presence.sum())}, "
          f"研究区 {oracle.nrows * oracle.ncols} 单元, {len(CONFIGS)} 配置 x {n_seeds} seeds")

    detail_all = []
    summary_rows = []
    for cname, over in CONFIGS:
        cfg = dict(BASE, **over)
        cfg["cov_indices"] = list(range(oracle.covs.shape[0]))
        pts, qrs, covs, duals, succ = [], [], [], [], 0
        for seed in seeds:
            res = SpeciesSearcher(oracle, cfg).run_episode(seed)
            dual = 1 if (in_cluster(res["found_cells"], A_boxes)
                         and in_cluster(res["found_cells"], B_boxes)) else 0
            pts.append(res["points_found"]); qrs.append(res["total_queries"])
            covs.append(res["covered_cells"]); duals.append(dual)
            succ += 1 if res["points_found"] >= 20 else 0
            detail_all.append([cname, seed, res["points_found"], res["total_queries"],
                               res["covered_cells"], round(res["covered_ratio"], 4), dual])
        n = len(seeds)
        mean = lambda x: sum(x) / n
        summary_rows.append([cname, round(mean(pts), 2), round(mean(qrs), 0),
                             round(mean(covs), 0), round(100.0 * sum(duals) / n, 1),
                             round(100.0 * succ / n, 1)])
        print(f"  {cname:22s} 点 {summary_rows[-1][1]:6.2f} | 查询 {summary_rows[-1][2]:6.0f} | "
              f"覆盖 {summary_rows[-1][3]:6.0f} | 双集群 {summary_rows[-1][4]:5.1f}% | "
              f"跑满 {summary_rows[-1][5]:5.1f}%")

    with open(OUT_DIR / f"ablation_species_{species}_detail.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "seed", "points_found", "queries", "covered_cells", "covered_ratio", "dual_cluster"])
        w.writerows(detail_all)
    with open(OUT_DIR / f"ablation_species_{species}_summary.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "mean_points", "mean_queries", "mean_covered", "dual_cluster_rate%", "full_target_rate%"])
        w.writerows(summary_rows)
    print(f"[done] -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
