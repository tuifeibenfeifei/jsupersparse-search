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
"""Task 2: root-cause diagnosis of the maritime search-and-rescue cold start (motion bootstrap).
- import SpeciesSearcher from the [snapshot] directory (concurrency-safe, second-round baseline);
- MaritimeOracle 150x150, budget 600 queries; seeds 101-110, single process, sequential;
- produces baseline.csv + artifacts.npz (for plotting).
"""
import sys, csv, time
from pathlib import Path
import numpy as np

ROOT = Path(REPO_ROOT)
sys.path.insert(0, str(SNAP))                       # snapshot species-search
sys.path.insert(1, str(ROOT / "maritime-search"))

from species_agent import SpeciesSearcher          # noqa: E402
from build_maritime_oracle import MaritimeOracle    # noqa: E402

OUT = ROOT / "results" / "maritime_search" / "bootstrap_analysis_v3"
OUT.mkdir(parents=True, exist_ok=True)
NPZ = ROOT / "maritime-search" / "data" / "oracle.npz"

SEEDS = list(range(101, 111))   # 101..110

# config copied verbatim from run_maritime.py CFG (task-specified values)
CFG = dict(
    target_points=15, max_queries=600, min_dist_cells=1.0,
    stage1_grid_step=4,
    cov_indices=list(range(19)),
    sparse_guidance=True, sparse_topk=6, sparse_min_pos=5, recalc_every=20,
    stage2_min=3, stage3_min=5, stage3_jump_interval=30,
    multicluster=True, cluster_ring_ratio=10.0, cluster_idle_limit=12,
    blank_jump=True, coverage_budget_ratio=0.12, blank_far_ratio=0.3,
    post_jump_grace=6, saturate_mask=False, idle_blank_mix=8,
)

def main():
    print(f"=== MARITIME BOOTSTRAP DIAG v3 seeds={SEEDS} budget={CFG['max_queries']} ===", flush=True)
    oracle = MaritimeOracle(NPZ)
    print(f"oracle {oracle.nrows}x{oracle.ncols} R={oracle.R_km}km "
          f"P0_cell={oracle.P0_cell} cell_km={oracle.cell_km}", flush=True)

    rows = []
    conv_by_seed = {}
    artifacts = {"traj": oracle.traj,
                "P0_cell": np.array(oracle.P0_cell),
                "cell_km": np.array(oracle.cell_km),
                "R_km": np.array(oracle.R_km),
                "nrows": np.array(oracle.nrows),
                "ncols": np.array(oracle.ncols)}
    visited101 = np.zeros((oracle.nrows, oracle.ncols), dtype=np.uint8)
    found101 = None

    for seed in SEEDS:
        t0 = time.time()
        searcher = SpeciesSearcher(oracle, CFG)
        res = searcher.run_episode(seed)
        dt = time.time() - t0
        # query time of the first hit
        found_t = list(searcher._found_t)
        first_hit_q = int(found_t[0]) if found_t else -1
        n_found_ts = len(found_t)
        rows.append([seed, res["points_found"], res["total_queries"],
                     res["covered_cells"], res["final_stage"],
                     first_hit_q, n_found_ts,
                     round(res.get("covered_ratio", 0.0), 4),
                     round(dt, 1)])
        conv_by_seed[seed] = res["convergence"]
        print(f"seed={seed} pts={res['points_found']} q={res['total_queries']} "
              f"covered={res['covered_cells']}({res.get('covered_ratio',0):.3f}) "
              f"stage={res['final_stage']} first_hit_q={first_hit_q} "
              f"n_found_ts={n_found_ts} t={dt:.1f}s", flush=True)
        if seed == 101:
            for (i, j) in searcher.visited_cells:
                if oracle.in_bounds(i, j):
                    visited101[i, j] = 1
            found101 = list(res["found_cells"])

    # baseline.csv
    with open(OUT / "baseline.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "points_found", "total_queries", "covered_cells",
                    "final_stage", "first_hit_q", "n_found_ts",
                    "covered_ratio", "dt_s"])
        w.writerows(rows)

    # artifacts.npz
    save = {"traj": oracle.traj,
            "P0_cell": np.array(oracle.P0_cell),
            "cell_km": np.array(oracle.cell_km),
            "R_km": np.array(oracle.R_km),
            "nrows": np.array(oracle.nrows),
            "ncols": np.array(oracle.ncols),
            "visited101": visited101,
            "found101": (np.array(found101) if found101 else np.zeros((0, 2), dtype=int)),
            "seeds": np.array(SEEDS)}
    for seed, conv in conv_by_seed.items():
        save[f"q_{seed}"] = np.array([c[0] for c in conv])
        save[f"f_{seed}"] = np.array([c[1] for c in conv])
        save[f"s_{seed}"] = np.array([c[2] for c in conv])
    np.savez(OUT / "_artifacts.npz", **save)
    print(f"baseline.csv -> {OUT / 'baseline.csv'}", flush=True)
    print(f"_artifacts.npz -> {OUT / '_artifacts.npz'}", flush=True)
    print("=== DONE ===", flush=True)

if __name__ == "__main__":
    main()
