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
"""run_cross_scenario_ablation.py — 跨场景关键机制消融。

三个场景各跑 3 个配置（full / AUC_off / multicluster_off）：
  - MC 深海: 20 world seeds x 20 algo seeds, oracle_npz
  - 云豹: 1 oracle x 20 algo seeds
  - 叶绿素: 1 oracle x 20 algo seeds

输出 results/ablation_cross/ablation_detail.csv
"""
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(REPO_ROOT)
sys.path.insert(0, str(ROOT / "species-search"))
sys.path.insert(0, str(ROOT / "chlorophyll-search"))
sys.path.insert(0, str(ROOT))

from oracle import SpeciesOracle  # noqa: E402
from species_agent import SpeciesSearcher  # noqa: E402

OUT = ROOT / "results" / "ablation_cross"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "progress.log"
DETAIL = OUT / "ablation_detail.csv"

SEEDS = list(range(101, 121))


def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---- 各场景 base cfg（与五轮最终配置对齐）----
MC_CFG = dict(
    target_points=40, max_queries=10000, min_dist_cells=2.0,
    stage2_min=3, stage3_min=5, stage3_jump_interval=30,
    stage3_jump_min_ratio=1.25, stage3_jump_max_ratio=5.0,
    curiosity_weight=1.0, guidance_on=True, cov_in_state=True,
    cov_indices=[0, 1, 2, 3, 4],
    multicluster=True, cluster_ring_ratio=10.0, cluster_idle_limit=12,
    blank_jump=True, coverage_budget_ratio=0.12,
    time_decay=True, decay_mode="exp", decay_tau=300.0,
    mutation_detect=True, mutation_streak_thresh=40,
    follow_mode=True, mutation_disperse_jump=True, mutation_jump_interval=15,
)

SPP_CFG = dict(
    target_points=20, max_queries=20000, min_dist_cells=1.0,
    stage2_min=3, stage3_min=5, stage3_jump_interval=30,
    stage3_jump_min_ratio=1.25, stage3_jump_max_ratio=5.0,
    curiosity_weight=1.0, learning_rate=0.1, discount_factor=0.95,
    epsilon_start=0.9, epsilon_end=0.1, epsilon_decay=0.999,
    stage1_grid_step=2, guidance_on=True, cov_in_state=True,
    recalc_every=20, stage2_topk=8,
    sparse_guidance=True, sparse_topk=6, sparse_min_pos=5,
    multicluster=True, cluster_ring_ratio=10.0, cluster_idle_limit=12,
    blank_jump=True, coverage_budget_ratio=0.12,
    blank_far_ratio=0.3, post_jump_grace=6, saturate_mask=False,
    cov_indices=list(range(19)), adaptive=True, directed_jump=True,
)

CHL_CFG = dict(
    target_points=20, max_queries=4000, min_dist_cells=1.0,
    stage2_min=3, stage3_min=5, stage3_jump_interval=30,
    stage3_jump_min_ratio=1.25, stage3_jump_max_ratio=5.0,
    curiosity_weight=1.0, learning_rate=0.1, discount_factor=0.95,
    epsilon_start=0.9, epsilon_end=0.1, epsilon_decay=0.999,
    stage1_grid_step=3, guidance_on=True, cov_in_state=True,
    recalc_every=20, stage2_topk=8,
    sparse_guidance=True, sparse_topk=6, sparse_min_pos=5,
    multicluster=True, cluster_ring_ratio=10.0, cluster_idle_limit=12,
    blank_jump=True, coverage_budget_ratio=0.12,
    blank_far_ratio=0.3, post_jump_grace=6, saturate_mask=False,
    cov_indices=list(range(19)), adaptive=True, directed_jump=True,
)


CONFIGS = [
    # (scenario, variant, cfg_overrides)
    ("minecraft",     "full",             {}),
    ("minecraft",     "AUC_off",         {"guidance_on": False}),
    ("minecraft",     "multicluster_off", {"multicluster": False}),
    ("clouded_leopard","full",            {}),
    ("clouded_leopard","AUC_off",         {"guidance_on": False, "sparse_guidance": False}),
    ("clouded_leopard","multicluster_off",{"multicluster": False}),
    ("chlorophyll",    "full",             {}),
    ("chlorophyll",    "AUC_off",         {"guidance_on": False, "sparse_guidance": False}),
    ("chlorophyll",    "multicluster_off",{"multicluster": False}),
]

BASE_CFG = {
    "minecraft": MC_CFG,
    "clouded_leopard": SPP_CFG,
    "chlorophyll": CHL_CFG,
}

ORACLE_PATHS = {
    "clouded_leopard": ROOT / "species-search" / "data" / "clouded_leopard" / "oracle.npz",
    "chlorophyll": ROOT / "chlorophyll-search" / "data" / "oracle.npz",
}

MC_WORLD_SEEDS = list(range(1, 21))


def run_scenario(scenario, variant, overrides):
    cfg = dict(BASE_CFG[scenario], **overrides)
    rows = []
    t0 = time.time()

    if scenario == "minecraft":
        for ws in MC_WORLD_SEEDS:
            npz = ROOT / "minecraft-search" / "oracle_npz" / f"mc_world_seed{ws}.npz"
            oracle = SpeciesOracle(npz)
            for ag in SEEDS:
                res = SpeciesSearcher(oracle, cfg).run_episode(ag)
                rows.append([scenario, variant, ws, ag,
                             res["points_found"], res["total_queries"]])
    else:
        oracle = SpeciesOracle(ORACLE_PATHS[scenario])
        for ag in SEEDS:
            res = SpeciesSearcher(oracle, cfg).run_episode(ag)
            rows.append([scenario, variant, 0, ag,
                         res["points_found"], res["total_queries"]])

    mp = sum(r[4] for r in rows) / len(rows)
    mq = sum(r[5] for r in rows) / len(rows)
    dt = time.time() - t0
    log(f"  {scenario:16s} {variant:18s} pts={mp:.2f} q={mq:.0f}  ({dt:.1f}s, n={len(rows)})")
    return rows


def main():
    t_all = time.time()
    log("=== 跨场景消融: full / AUC_off / multicluster_off ===")

    all_rows = []
    for scenario, variant, overrides in CONFIGS:
        rows = run_scenario(scenario, variant, overrides)
        all_rows.extend(rows)

    with open(DETAIL, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "variant", "world_seed", "algo_seed",
                    "points_found", "total_queries"])
        w.writerows(all_rows)

    # 汇总
    print("\n=== Cross-Scenario Ablation Summary (mean points) ===")
    for sc in ["minecraft", "clouded_leopard", "chlorophyll"]:
        print(f"\n[{sc}]")
        for v in ["full", "AUC_off", "multicluster_off"]:
            vals = [r[4] for r in all_rows if r[0] == sc and r[1] == v]
            print(f"  {v:18s} mean={sum(vals)/len(vals):.2f}  (n={len(vals)})")

    log(f"=== 完成 -> {DETAIL} ({(time.time()-t_all)/60:.1f} min)")


if __name__ == "__main__":
    main()
