# -*- coding: utf-8 -*-
"""Small hyperparameter-sensitivity grid: n_bins in {4,8,16} (chlorophyll oracle, CHL_CFG as-is with only n_bins overridden).

Seeds algo_seed 101..105; one episode per (n_bins, seed).
Outputs supplement_stats/peerj_nbins_scan.csv (n_bins,algo_seed,points_found,total_queries).
"""
import csv
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\19264\Desktop\deep-sea-exploration")
sys.path.insert(0, str(ROOT / "species-search"))
sys.path.insert(0, str(ROOT / "chlorophyll-search"))
sys.path.insert(0, str(ROOT))

from oracle import SpeciesOracle  # noqa: E402
from species_agent import SpeciesSearcher  # noqa: E402

# identical to run_cross_ablation.py CHL_CFG (copied verbatim, with only the n_bins override added)
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

ORACLE = ROOT / "chlorophyll-search" / "data" / "oracle.npz"
OUT = ROOT / "supplement_stats" / "peerj_nbins_scan.csv"
N_BINS_GRID = [4, 8, 16]
SEEDS = [101, 102, 103, 104, 105]


def main():
    oracle = SpeciesOracle(ORACLE)
    rows = []
    for nb in N_BINS_GRID:
        cfg = dict(CHL_CFG, n_bins=nb)
        for ag in SEEDS:
            res = SpeciesSearcher(oracle, cfg).run_episode(ag)
            rows.append([nb, ag, res["points_found"], res["total_queries"]])
            print(f"  n_bins={nb:2d} seed={ag} -> pts={res['points_found']:2d} q={res['total_queries']}", flush=True)

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n_bins", "algo_seed", "points_found", "total_queries"])
        w.writerows(rows)
    print(f"[saved] {OUT} ({len(rows)} rows)")

    # mean summary
    print("\n=== n_bins scan mean summary ===")
    import numpy as np
    for nb in N_BINS_GRID:
        qs = [r[3] for r in rows if r[0] == nb]
        pts = [r[2] for r in rows if r[0] == nb]
        print(f"  n_bins={nb:2d}: queries mean={np.mean(qs):8.1f} sd={np.std(qs, ddof=1):7.1f}  "
              f"points mean={np.mean(pts):.1f} (n={len(qs)})")


if __name__ == "__main__":
    main()
