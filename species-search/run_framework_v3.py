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
"""run_framework_v3.py — v3 优化实验：冷启动 + 定向远跳 + 覆盖指标显式化
输入全部 19 个 BIO 变量，sparse_guidance 自动特征选择。
输出（results/species_search/）：
  framework_species_v3_{species}_params19.csv      每 seed 明细（点数/查询/覆盖/双集群）
  framework_species_v3_summary_{species}.csv       v3 聚合 + v2 对比
用法：
  python run_framework_v3.py <oracle_path> <species_tag> <cluster_A> <cluster_B> [n_seeds]
"""
import csv
import sys
from pathlib import Path

from oracle import SpeciesOracle
from species_agent import SpeciesSearcher

OUT_DIR = Path(REPO_ROOT / "results/species_search")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_CONFIG = dict(
    target_points=20,
    max_queries=20000,
    min_dist_cells=1.0,
    stage2_min=3,
    stage3_min=5,
    stage3_jump_interval=30,
    stage3_jump_min_ratio=1.25,
    stage3_jump_max_ratio=5.0,
    curiosity_weight=1.0,
    learning_rate=0.1,
    discount_factor=0.95,
    epsilon_start=0.9,
    epsilon_end=0.1,
    epsilon_decay=0.999,
    stage1_grid_step=2,
    guidance_on=True,
    cov_in_state=True,
    recalc_every=20,
    stage2_topk=8,
    sparse_guidance=True,
    sparse_topk=6,
    sparse_min_pos=5,
    # v3 优化开关
    cold_start_samples=300,   # 冷启动：总样本 < 300 时全量加权，防 AUC 小样本噪声
    directed_jump=True,       # 定向远跳：向未覆盖高分带
)


def in_cluster(cells, boxes):
    for i, j in cells:
        for (imin, imax, jmin, jmax) in boxes:
            if imin <= i <= imax and jmin <= j <= jmax:
                return True
    return False


def main():
    if len(sys.argv) < 5:
        print("usage: run_framework_v3.py <oracle_path> <species> <A_boxes> <B_boxes> [n_seeds]")
        print("  boxes format: 'i_lo,i_hi,j_lo,j_hi' (multiple separated by '|')")
        return 1
    oracle_path = Path(sys.argv[1])
    species = sys.argv[2]
    A_boxes = [tuple(int(v) for v in b.split(",")) for b in sys.argv[3].split("|")]
    B_boxes = [tuple(int(v) for v in b.split(",")) for b in sys.argv[4].split("|")]
    n_seeds = int(sys.argv[5]) if len(sys.argv) > 5 else 20
    seeds = list(range(101, 101 + n_seeds))

    oracle = SpeciesOracle(oracle_path)
    print(f"[oracle] {oracle.nrows}x{oracle.ncols} 网格, presence={int(oracle.presence.sum())}, "
          f"研究区 {oracle.nrows * oracle.ncols} 单元")

    cfg = dict(BASE_CONFIG, cov_indices=list(range(oracle.covs.shape[0])))
    detail = []
    tot = dict(points=[], queries=[], covered=[], cov_ratio=[], dual=[], succ=0)
    for seed in seeds:
        res = SpeciesSearcher(oracle, cfg).run_episode(seed)
        dual_hit = 1 if (in_cluster(res["found_cells"], A_boxes)
                         and in_cluster(res["found_cells"], B_boxes)) else 0
        detail.append([seed, res["points_found"], res["total_queries"],
                       res["covered_cells"], round(res["covered_ratio"], 4),
                       dual_hit])
        tot["points"].append(res["points_found"])
        tot["queries"].append(res["total_queries"])
        tot["covered"].append(res["covered_cells"])
        tot["cov_ratio"].append(res["covered_ratio"])
        tot["dual"].append(dual_hit)
        tot["succ"] += 1 if res["points_found"] >= 20 else 0
        print(f"[seed {seed}] 点 {res['points_found']}/20 | 查询 {res['total_queries']} | "
              f"覆盖 {res['covered_cells']} ({res['covered_ratio']:.3f}) | 双集群 {dual_hit}")

    with open(OUT_DIR / f"framework_species_v3_{species}_params19.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "points_found", "queries", "covered_cells", "covered_ratio", "dual_cluster"])
        w.writerows(detail)

    n = len(seeds)
    mean = lambda x: sum(x) / len(x)
    summary = {
        "mean_points": round(mean(tot["points"]), 2),
        "mean_queries": round(mean(tot["queries"]), 0),
        "mean_covered": round(mean(tot["covered"]), 0),
        "mean_covered_ratio": round(mean(tot["cov_ratio"]), 4),
        "dual_cluster_rate": round(100.0 * sum(tot["dual"]) / n, 1),
        "success_runs": tot["succ"],
    }
    with open(OUT_DIR / f"framework_species_v3_summary_{species}.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "v3_value"])
        for k, v in summary.items():
            w.writerow([k, v])
    print(f"\n[v3 汇总] {species}（{n} seeds，输入19+自动选择+冷启动+定向远跳）")
    for k, v in summary.items():
        print(f"  {k:22s} {v}")
    print(f"\n[done] -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
