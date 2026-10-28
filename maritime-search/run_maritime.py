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
"""run_maritime.py — maritime search-and-rescue scenario: reuses SpeciesSearcher to run seeds 101/102/103.

- imports the species-search framework (SpeciesSearcher) and this directory's oracle (MaritimeOracle) via sys.path.
- does not modify the framework itself. Episodes run sequentially in a single process (ThreadPoolExecutor is forbidden).
- outputs: results/maritime_search/maritime_detail.csv / maritime_summary.csv
        + _artifacts.npz (for plotting).
"""
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(REPO_ROOT)
sys.path.insert(0, str(ROOT / "species-search"))
sys.path.insert(0, str(ROOT / "maritime-search"))

import numpy as np  # noqa: E402
from species_agent import SpeciesSearcher          # noqa: E402
from build_maritime_oracle import MaritimeOracle  # noqa: E402

OUT = ROOT / "results" / "maritime_search"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "progress.log"
NPZ = ROOT / "maritime-search" / "data" / "oracle.npz"

SEEDS = [101, 102, 103]

# ---- experiment config (as specified by the task; sparse_guidance enabled to turn on AUC sparse parameter selection) ----
CFG = dict(
    target_points=15, max_queries=600, min_dist_cells=1.0,
    stage1_grid_step=4,
    cov_indices=list(range(19)),
    multicluster=True, cluster_ring_ratio=10.0, cluster_idle_limit=12,
    blank_jump=True, coverage_budget_ratio=0.12, blank_far_ratio=0.3,
    post_jump_grace=6, saturate_mask=False, idle_blank_mix=8,
    # AUC sparse parameter selection (framework defaults, consistent with the final version)
    sparse_guidance=True, sparse_topk=6, sparse_min_pos=5, recalc_every=20,
    stage2_min=3, stage3_min=5, stage3_jump_interval=30,
)


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    log("=" * 64)
    log(f"maritime search-and-rescue experiment started, seeds={SEEDS}, sequential single-process run, budget={CFG['max_queries']}")
    oracle = MaritimeOracle(NPZ)
    log(f"oracle {oracle.nrows}x{oracle.ncols}, covs={oracle.covs.shape}, "
        f"R={oracle.R_km}km, P0={oracle.P0_cell}")

    detail_rows = []
    artifacts = {"traj": oracle.traj, "cov_names": np.array(oracle.cov_names)}
    conv_by_seed = {}
    visited101 = np.zeros((oracle.nrows, oracle.ncols), dtype=np.uint8)
    found101 = None
    feature_snapshots = []   # the last AUC parameter-selection record of every seed

    for seed in SEEDS:
        t0 = time.time()
        searcher = SpeciesSearcher(oracle, CFG)
        res = searcher.run_episode(seed)
        dt = time.time() - t0
        detail_rows.append([seed, res["points_found"], res["total_queries"],
                            res["covered_cells"], res["final_stage"]])
        conv_by_seed[seed] = res["convergence"]
        log(f"seed={seed} pts={res['points_found']} q={res['total_queries']} "
            f"covered={res['covered_cells']} stage={res['final_stage']} "
            f"t={dt:.1f}s")
        # snapshot of the last feature selection
        fl = res["feature_log"]
        if fl:
            feature_snapshots.append((seed, fl[-1]))
        if seed == 101:
            for (i, j) in searcher.visited_cells:
                if oracle.in_bounds(i, j):
                    visited101[i, j] = 1
            found101 = list(res["found_cells"])
            artifacts["visited101"] = visited101
            artifacts["found101"] = np.array(found101) if found101 else np.zeros((0, 2))

    # ---- detail CSV ----
    with open(OUT / "maritime_detail.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "points_found", "total_queries",
                    "covered_cells", "final_stage"])
        w.writerows(detail_rows)

    # ---- summary CSV (mean +/- std) ----
    n = len(detail_rows)
    pts = [r[1] for r in detail_rows]
    qs = [r[2] for r in detail_rows]
    cov = [r[3] for r in detail_rows]
    with open(OUT / "maritime_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "std", "min", "max"])
        w.writerow(["points_found", round(np.mean(pts), 2), round(np.std(pts), 2),
                    min(pts), max(pts)])
        w.writerow(["total_queries", round(np.mean(qs), 1), round(np.std(qs), 1),
                    min(qs), max(qs)])
        w.writerow(["covered_cells", round(np.mean(cov), 1), round(np.std(cov), 1),
                    min(cov), max(cov)])
    log(f"summary: pts={np.mean(pts):.2f}+/-{np.std(pts):.2f}  "
        f"q={np.mean(qs):.0f}  covered={np.mean(cov):.0f}")

    # ---- AUC parameter-selection conclusion (aggregate the last snapshot across seeds) ----
    log("--- last AUC parameter-selection snapshot (n_q, n_pos, [(cov_idx, weight)...]) ---")
    agg = {}
    for seed, (nq, npos, keep) in feature_snapshots:
        log(f"  seed={seed} n_q={nq} n_pos={npos} keep={keep}")
        for k, wv in keep:
            agg.setdefault(k, []).append(wv)
    auc_table = {k: float(np.mean(v)) for k, v in agg.items()}
    log(f"  cross-seed average weights: {sorted(auc_table.items(), key=lambda x: -x[1])}")

    # ---- save plotting artifacts ----
    conv_arr = {}
    for seed, conv in conv_by_seed.items():
        conv_arr[f"q_{seed}"] = np.array([c[0] for c in conv])
        conv_arr[f"f_{seed}"] = np.array([c[1] for c in conv])
    save = dict(conv_arr)
    save["traj"] = oracle.traj
    save["P0_cell"] = np.array(oracle.P0_cell)
    save["cell_km"] = np.array(oracle.cell_km)
    save["R_km"] = np.array(oracle.R_km)
    save["visited101"] = visited101
    save["found101"] = (np.array(found101) if found101 else np.zeros((0, 2)))
    save["cov_names"] = np.array(oracle.cov_names)
    save["auc_keys"] = np.array(list(auc_table.keys()))
    save["auc_vals"] = np.array(list(auc_table.values()))
    np.savez(OUT / "_artifacts.npz", **save)
    log(f"artifacts -> {OUT / '_artifacts.npz'}")
    log("DONE.")


if __name__ == "__main__":
    main()
