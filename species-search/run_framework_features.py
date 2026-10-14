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
"""run_framework_features.py — 自动特征选择诊断实验
输入全部 19 个 BIO 协变量，机器在搜索中积累样本，AUC 稀疏化自动选出显著代表特征。
输出：
  - framework_species_sparse_features_{species}.csv     每轮选中的特征明细
  - framework_species_sparse_features_summary_{species}.csv  20 seeds 汇总：各特征选中率/平均权重
用法：
  python run_framework_features.py <oracle_path> <species_tag> [n_seeds]
"""
import csv
import sys
from collections import Counter, defaultdict
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
    sparse_guidance=True,     # 输入 19，自动稀疏选特征
    sparse_topk=6,
    sparse_min_pos=5,
)


def main():
    if len(sys.argv) < 3:
        print("usage: python run_framework_features.py <oracle_path> <species_tag> [n_seeds]")
        return 1
    oracle_path = Path(sys.argv[1])
    species = sys.argv[2]
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    seeds = list(range(101, 101 + n_seeds))

    oracle = SpeciesOracle(oracle_path)
    names = oracle.cov_names
    print(f"[oracle] {oracle.nrows}x{oracle.ncols} 网格, presence={int(oracle.presence.sum())}, "
          f"协变量 {len(names)} 个: {names}")

    cfg = dict(BASE_CONFIG, cov_indices=list(range(oracle.covs.shape[0])))  # 输入全部 19
    detail_rows = []
    # 汇总统计（按 seed 去重）：变量索引 -> 被选中的 seed 集合
    seed_sel = defaultdict(set)
    weight_sum = defaultdict(float)
    first_sel_q = defaultdict(list)
    round_cnt = Counter()

    for seed in seeds:
        res = SpeciesSearcher(oracle, cfg).run_episode(seed)
        seen_in_seed = set()
        for n_q, n_pos, feats in res["feature_log"]:
            feats = list(feats)
            names_s = ";".join(f"{names[k]}={w:.3f}" for k, w in feats)
            detail_rows.append([seed, n_q, n_pos, names_s])
            for k, w in feats:
                seen_in_seed.add(k)
                weight_sum[k] += w
                round_cnt[k] += 1
                first_sel_q[k].append(n_q)
        for k in seen_in_seed:
            seed_sel[k].add(seed)
        print(f"[seed {seed}] 点 {res['points_found']}/20 | 查询 {res['total_queries']} | "
              f"特征选择轮次 {len(res['feature_log'])}")

    with open(OUT_DIR / f"framework_species_sparse_features_{species}.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "query", "n_pos", "selected_features(idx=weight)"])
        w.writerows(detail_rows)

    # 汇总：选中率 = 选中该变量的 seed 数占比；平均权重；平均首次选中查询
    n_seeds_used = len(seeds)
    rows = []
    for k in range(len(names)):
        n_sel = len(seed_sel[k])
        if n_sel == 0:
            rows.append([names[k], 0, 0.0, "-"])
            continue
        avg_w = weight_sum[k] / round_cnt[k]
        avg_first = sum(first_sel_q[k]) / len(first_sel_q[k])
        rows.append([names[k], round(100.0 * n_sel / n_seeds_used, 1), round(avg_w, 4),
                     round(avg_first, 0)])
    rows.sort(key=lambda x: -x[1])
    with open(OUT_DIR / f"framework_species_sparse_features_summary_{species}.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature", "selected_seed_rate(%)", "avg_weight", "avg_first_select_query"])
        w.writerows(rows)

    print(f"\n[自动特征选择汇总] {species}（{n_seeds} seeds）按选中率降序：")
    for r in rows:
        print(f"  {r[0]:6s}  选中率 {r[1]:5.1f}%  平均权重 {r[2]:.4f}  首次选中 @~{r[3]}")
    print(f"\n[done] 特征诊断已保存 -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
