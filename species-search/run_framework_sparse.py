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
"""run_framework_sparse.py — v2 在线稀疏化导向实验
验证目标：参数越多越强（19 参数稀疏化后 ≥ 2 参数）
用法：
  python run_framework_sparse.py <oracle_path> <species_tag> [params2,params19]
"""
import csv
import sys
from pathlib import Path

from oracle import SpeciesOracle
from species_agent import SpeciesSearcher

OUT_DIR = Path(REPO_ROOT / "results/species_search")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLUSTERS = {
    "black_muntjac": ([(30, 38, 36, 46)], [(103, 104, 19, 24)]),
    "clouded_leopard": ([(40, 70, 0, 55)], [(0, 50, 125, 270)]),
}

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
    # ---- v2 稀疏化导向 ----
    sparse_guidance=True,
    sparse_topk=6,
    sparse_min_pos=5,
)

COV_GROUPS = {
    "params2": [0, 11],                       # bio1 + bio12
    "params5": [0, 3, 4, 5, 11],              # bio1,bio4,bio5,bio6,bio12
    "params19": list(range(19)),              # 全部 BIO
}
SEEDS = list(range(101, 121))
N_REPEATS = len(SEEDS)


def main():
    if len(sys.argv) < 3:
        print("usage: python run_framework_sparse.py <oracle_path> <species_tag> [params2,params19]")
        return 1
    oracle_path = Path(sys.argv[1])
    species = sys.argv[2]
    only = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
    clusters = CLUSTERS.get(species, CLUSTERS["black_muntjac"])

    oracle = SpeciesOracle(oracle_path)
    print(f"[oracle] {oracle.nrows}x{oracle.ncols} = {oracle.total_cells} 网格, "
          f"presence={int(oracle.presence.sum())} ({oracle.presence.mean()*100:.2f}%)")
    print(f"[v2-sparse] species={species} sparse_topk=6 min_pos=5\n")

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
        with open(OUT_DIR / f"framework_species_sparse_{species}_{name}.csv", "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["seed", "points_found", "total_queries", "final_stage",
                        "cov_area_cells", "cross_cluster"])
            w.writerows(rows)

    with open(OUT_DIR / f"framework_species_sparse_summary_{species}.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["algo", "avg_points_found", "avg_total_queries",
                    "avg_cov_area_cells", "cross_cluster_rate"])
        w.writerows(summary)
    with open(OUT_DIR / f"framework_species_sparse_convergence_{species}.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["algo", "seed", "query", "points", "stage"])
        w.writerows(conv_rows)
    print("\n[done] v2-sparse 结果已保存 ->", OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
