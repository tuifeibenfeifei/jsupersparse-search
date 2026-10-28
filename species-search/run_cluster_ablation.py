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
"""run_cluster_ablation.py - run only the new cluster-aware configs (A9/A10) and merge them into the existing ablation CSV.
Usage: python run_cluster_ablation.py <oracle_path> <species> <A_boxes> <B_boxes> [n_seeds]
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

CONFIGS = [
    ("A9_sparse_cluster", dict(guidance_on=True, sparse_guidance=True, cluster_guidance=True)),
    ("A10_sparse_cluster_directed", dict(guidance_on=True, sparse_guidance=True,
                                         cluster_guidance=True, directed_jump=True)),
]


def in_cluster(cells, boxes):
    for i, j in cells:
        for (imin, imax, jmin, jmax) in boxes:
            if imin <= i <= imax and jmin <= j <= jmax:
                return True
    return False


def load_existing(species):
    detail, summary = [], []
    dp, sp = OUT_DIR / f"ablation_species_{species}_detail.csv", OUT_DIR / f"ablation_species_{species}_summary.csv"
    if dp.exists():
        with open(dp, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                detail.append([r["config"], int(r["seed"]), int(r["points_found"]),
                               int(r["queries"]), int(r["covered_cells"]),
                               float(r["covered_ratio"]), int(r["dual_cluster"])])
    if sp.exists():
        with open(sp, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                summary.append([r["config"], float(r["mean_points"]), float(r["mean_queries"]),
                                float(r["mean_covered"]), float(r["dual_cluster_rate%"]),
                                float(r["full_target_rate%"])])
    return detail, summary


def main():
    if len(sys.argv) < 5:
        print("usage: run_cluster_ablation.py <oracle_path> <species> <A_boxes> <B_boxes> [n_seeds]")
        return 1
    oracle_path = Path(sys.argv[1])
    species = sys.argv[2]
    A_boxes = [tuple(int(v) for v in b.split(",")) for b in sys.argv[3].split("|")]
    B_boxes = [tuple(int(v) for v in b.split(",")) for b in sys.argv[4].split("|")]
    n_seeds = int(sys.argv[5]) if len(sys.argv) > 5 else 20
    seeds = list(range(101, 101 + n_seeds))

    oracle = SpeciesOracle(oracle_path)
    detail_all, summary_rows = load_existing(species)
    existing = {r[0] for r in summary_rows}

    for cname, over in CONFIGS:
        if cname in existing:
            print(f"  {cname} already present, skipping")
            continue
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
        print(f"  {cname:26s} pts {summary_rows[-1][1]:6.2f} | queries {summary_rows[-1][2]:6.0f} | "
              f"covered {summary_rows[-1][3]:6.0f} | dual cluster {summary_rows[-1][4]:5.1f}% | "
              f"target reached {summary_rows[-1][5]:5.1f}%")

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
