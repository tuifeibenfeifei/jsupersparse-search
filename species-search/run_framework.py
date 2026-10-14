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
"""run_framework.py — 稀有物种数据集：运行用户框架（物种版），参数数量对比 2/5/19
目标：找到 20 个分散出现网格（分散约束 1 cell ≈ 2.8km）；预算 20000 查询。
输出到 REPO_ROOT/results/species_search/
"""
import csv
import sys
from pathlib import Path

from oracle import SpeciesOracle
from species_agent import SpeciesSearcher

# 结果输出目录：项目主 results 目录下（非 species-search 内部）
OUT_DIR = Path(REPO_ROOT / "results/species_search")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 默认 oracle：黑麂；可用 argv 指定其他物种 oracle
ORACLE_PATH = Path(__file__).resolve().parent / "data" / "black_muntjac" / "oracle.npz"

# 集群判定（跨集群覆盖指标）：不同物种按分布定义两个主要集群（网格坐标）
CLUSTERS = {
    # 黑麂：浙皖赣交界
    "black_muntjac": ([(30, 38, 36, 46)], [(103, 104, 19, 24)]),
    # 云豹：藏东南墨脱 + 滇西北-滇西
    "clouded_leopard": ([(40, 70, 0, 55)], [(0, 50, 125, 270)]),
}

# 框架配置：目标 20 个分散出现网格，预算 20000
BASE_CONFIG = dict(
    target_points=20,
    max_queries=20000,
    min_dist_cells=1.0,      # 分散约束 1 cell（≈2.8km；0.025°）
    stage2_min=3,
    stage3_min=5,
    stage3_jump_interval=30,
    stage3_jump_min_ratio=1.25,
    stage3_jump_max_ratio=5.0,
    curiosity_weight=1.0,
    guidance_on=True,
    cov_in_state=True,
)

# 参数数量对比：2 / 5 / 19 个气候变量
# 19 变量索引：0=bio1 ... 11=bio12 ... 18=bio19
COV_GROUPS = {
    "params2": [0, 11],                         # bio1 年均温 + bio12 年降水
    "params5": [0, 3, 4, 5, 11],                # bio1,bio4,bio5,bio6,bio12（温度3+降水2）
    "params19": list(range(19)),                # 全部 19 个 BIO 变量
}

N_REPEATS = 20
SEEDS = list(range(101, 101 + N_REPEATS))


def main():
    # 可选：python run_framework.py [params5,params19] [oracle_path] [species_tag]
    only = None
    oracle_path = ORACLE_PATH
    species = "black_muntjac"
    if len(sys.argv) > 1:
        only = set(sys.argv[1].split(","))
    if len(sys.argv) > 2:
        oracle_path = Path(sys.argv[2])
        # 由路径推断物种
        sp = str(oracle_path).replace("\\", "/").split("/")[-2]
        if sp in CLUSTERS:
            species = sp
    if len(sys.argv) > 3:
        species = sys.argv[3]
    clusters = CLUSTERS.get(species, CLUSTERS["black_muntjac"])

    oracle = SpeciesOracle(oracle_path)
    print(f"[oracle] {oracle.nrows}x{oracle.ncols} = {oracle.total_cells} 网格, "
          f"presence={int(oracle.presence.sum())} ({oracle.presence.mean()*100:.2f}%)")
    print(f"[config] species={species} target=20, max_queries=20000, min_dist=1 cell\n")

    summary, conv_rows = [], []
    for name, idx in COV_GROUPS.items():
        if only and name not in only:
            continue
        found_sum, q_sum, area_sum, cross_sum = 0, 0, 0, 0
        rows = []
        for seed in SEEDS:
            cfg = dict(BASE_CONFIG, cov_indices=idx)
            res = SpeciesSearcher(oracle, cfg).run_episode(seed)
            found_sum += res["points_found"]
            q_sum += res["total_queries"]
            # 覆盖广度：found 网格包围盒面积（网格单位）+ 双集群覆盖
            cs = [(int(a), int(b)) for a, b in res["found_cells"]]
            if cs:
                area = (max(c[0] for c in cs) - min(c[0] for c in cs) + 1) * \
                       (max(c[1] for c in cs) - min(c[1] for c in cs) + 1)
            else:
                area = 0
            def in_cluster(cs, boxes):
                return any(lo_i <= i <= hi_i and lo_j <= j <= hi_j
                           for i, j in cs for (lo_i, hi_i, lo_j, hi_j) in boxes)
            inA = in_cluster(cs, clusters[0])
            inB = in_cluster(cs, clusters[1])
            cross = 1 if (inA and inB) else 0
            area_sum += area
            cross_sum += cross
            rows.append([seed, res["points_found"], res["total_queries"], res["final_stage"],
                         area, cross])
            for q, n, st in res["convergence"]:
                conv_rows.append([name, seed, q, n, st])
        n = N_REPEATS
        avg_f, avg_q = found_sum / n, q_sum / n
        avg_area, cross_rate = area_sum / n, cross_sum / n
        summary.append([name, avg_f, avg_q, avg_area, cross_rate])
        print(f"[{name}] 平均点数 {avg_f:.1f}/20 | 平均查询 {avg_q:.0f} | "
              f"覆盖面积 {avg_area:.0f} 网格 | 双集群覆盖率 {cross_rate:.0%}")
        for r in rows[:5]:
            print(f"    seed {r[0]}: {r[1]}/20 点, {r[2]} 查询, 覆盖{r[4]}网格, 跨集群{r[5]}")
        with open(OUT_DIR / f"framework_species_{species}_{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["seed", "points_found", "total_queries", "final_stage",
                        "cov_area_cells", "cross_cluster"])
            w.writerows(rows)

    with open(OUT_DIR / f"framework_species_summary_{species}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["algo", "avg_points_found", "avg_total_queries",
                    "avg_cov_area_cells", "cross_cluster_rate"])
        w.writerows(summary)
    with open(OUT_DIR / f"framework_species_convergence_{species}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["algo", "seed", "query", "points", "stage"])
        w.writerows(conv_rows)
    print("\n[done] 结果已保存 ->", OUT_DIR)


if __name__ == "__main__":
    main()
