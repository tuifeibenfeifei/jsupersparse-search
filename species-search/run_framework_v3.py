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
"""run_framework_v3.py — v3 optimization experiment: cold start + directed long jump + explicit coverage metrics
All 19 BIO variables are fed in; sparse_guidance does automatic feature selection.
Outputs (results/species_search/):
  framework_species_v3_{species}_params19.csv      per-seed details (points/queries/coverage/dual cluster)
  framework_species_v3_summary_{species}.csv       v3 aggregation + v2 comparison
Usage:
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
    # v3 optimization switches
    cold_start_samples=300,   # cold start: when total samples < 300, weight all samples fully to prevent AUC small-sample noise
    directed_jump=True,       # directed long jump: jump toward uncovered high-score bands
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
    print(f"[oracle] {oracle.nrows}x{oracle.ncols} grid, presence={int(oracle.presence.sum())}, "
          f"{oracle.nrows * oracle.ncols} cells in the study area")

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
        print(f"[seed {seed}] points {res['points_found']}/20 | queries {res['total_queries']} | "
              f"covered {res['covered_cells']} ({res['covered_ratio']:.3f}) | dual cluster {dual_hit}")

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
    print(f"\n[v3 summary] {species} ({n} seeds, input 19 + auto selection + cold start + directed long jump)")
    for k, v in summary.items():
        print(f"  {k:22s} {v}")
    print(f"\n[done] -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
