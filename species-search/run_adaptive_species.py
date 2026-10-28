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
"""run_adaptive_species.py — A11 adaptive config comparison experiment (black muntjac + clouded leopard)
A11_adaptive: guidance_on + AUC sparsity + adaptive (online diagnosis: multi-cluster -> cluster-dispersed guidance,
              single-cluster -> global guidance) + directed long jump. Does not rely on manually specifying cluster_guidance.
Comparison baselines: read A0 (skeleton) / A2 (sparsity) / A10 (cluster + directed) from the existing ablation summary.
Output: results/species_search/adaptive/{species}_adaptive_summary.csv
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(REPO_ROOT / "species-search")))
from oracle import SpeciesOracle
from species_agent import SpeciesSearcher

ROOT = Path(REPO_ROOT)
OUT = ROOT / "results" / "species_search" / "adaptive"
OUT.mkdir(parents=True, exist_ok=True)

SPECIES = [
    ("black_muntjac", ROOT / "species-search" / "data" / "black_muntjac" / "oracle.npz"),
    ("clouded_leopard", ROOT / "species-search" / "data" / "clouded_leopard" / "oracle.npz"),
]

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
A11 = dict(BASE, adaptive=True, directed_jump=True)
A11["cov_indices"] = list(range(19))   # all 19 covariates as input
SEEDS = list(range(101, 121))


def two_cluster_rate(oracle, found_cells, n_bins=8):
    """rough dual-cluster determination: after binning, whether >=2 non-adjacent bin bands are covered (consistent with the ablation convention)"""
    if not found_cells:
        return 0.0
    bs_i = max(1, oracle.nrows // n_bins)
    bs_j = max(1, oracle.ncols // n_bins)
    bins = set()
    for (i, j) in found_cells:
        bins.add((i // bs_i, j // bs_j))
    return 1.0 if len(bins) >= 2 else 0.0


def main():
    print(f"[adaptive] A11 x {len(SEEDS)} seeds, two species", flush=True)
    for name, npz in SPECIES:
        oracle = SpeciesOracle(npz)
        pts, qrs, covs, two = [], [], [], 0
        for seed in SEEDS:
            res = SpeciesSearcher(oracle, A11).run_episode(seed)
            pts.append(res["points_found"]); qrs.append(res["total_queries"])
            covs.append(res["covered_cells"])
            two += two_cluster_rate(oracle, res["found_cells"])
        n = len(SEEDS)
        mean = lambda x: sum(x) / n
        print(f"  {name:16s} pts {mean(pts):6.2f} | queries {mean(qrs):6.0f} | covered {mean(covs):6.0f} | "
              f"dual cluster {100.0*two/n:5.1f}%", flush=True)
        with open(OUT / f"{name}_adaptive_summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["config", "mean_points", "mean_queries", "mean_covered", "two_cluster_rate%"])
            w.writerow(["A11_adaptive", round(mean(pts), 2), round(mean(qrs), 0),
                        round(mean(covs), 0), round(100.0 * two / n, 1)])
    print("[done] ->", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
