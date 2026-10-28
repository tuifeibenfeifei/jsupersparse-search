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
"""run_mc_gpucb.py — MC scenario GP-UCB baseline (downsampled 50x50=2500 cells).

Approach:
  - Downsample the 100x100 study area to 50x50 (take 1 cell every 2), totaling 2500 candidate cells
  - Initially query 10 random points
  - Then every 20 steps, train a GP (RBF kernel) on queried points, compute on all unqueried cells
    UCB = mean + kappa * std, select the highest for query
  - presence is 0/1; GP regresses to continuous values
Output results/minecraft_search/baseline_mc_gpucb.csv
"""
import csv
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

ROOT = Path(REPO_ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "minecraft-search"))

from mc_oracle import MCGridOracle  # noqa: E402

OUT = ROOT / "results" / "minecraft_search"
LOG = OUT / "progress.log"

WORLD_SEEDS = list(range(1, 21))
ALGO_SEEDS = list(range(101, 106))
TARGET = 40
MAX_Q = 10000
MIN_D = 2.0
DOWNSAMPLE = 2  # Take 1 cell every 2 -> 50x50=2500
KAPPA = 2.0
INIT_RANDOM = 10
RETRAIN_EVERY = 20


def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _mindist_ok(i, j, found, min_d):
    return all(math.hypot(i - fi, j - fj) >= min_d for fi, fj in found)


def run_gpucb(oracle, algo_seed):
    rng = np.random.RandomState(algo_seed)
    nrows, ncols = oracle.nrows, oracle.ncols  # 100, 100

    # Downsampled candidate cells (center-aligned)
    cand_i = np.arange(1, nrows, DOWNSAMPLE)
    cand_j = np.arange(1, ncols, DOWNSAMPLE)
    ci_grid, cj_grid = np.meshgrid(cand_i, cand_j, indexing="ij")
    candidates = np.column_stack([ci_grid.ravel(), cj_grid.ravel()]).astype(float)  # (N, 2)
    N = len(candidates)

    queried = np.zeros(N, dtype=bool)
    y_obs = np.zeros(N, dtype=float)
    found = []
    oracle.reset_counter()

    def query_by_idx(idx):
        i, j = int(candidates[idx, 0]), int(candidates[idx, 1])
        p, _ = oracle.query_cell(i, j)
        queried[idx] = True
        y_obs[idx] = 1.0 if p else 0.0
        if p and _mindist_ok(i, j, found, MIN_D):
            found.append((i, j))

    # Initially query INIT_RANDOM random points
    init_idx = rng.choice(N, size=INIT_RANDOM, replace=False)
    for idx in init_idx:
        if len(found) >= TARGET or oracle.query_count >= MAX_Q:
            break
        query_by_idx(idx)

    gp = None
    while oracle.query_count < MAX_Q and len(found) < TARGET:
        # Train GP
        known = queried
        if known.sum() >= 5:
            Xtr = candidates[known]
            ytr = y_obs[known]
            if gp is None or oracle.query_count % RETRAIN_EVERY == 0:
                kernel = C(1.0) * RBF(length_scale=5.0)
                gp = GaussianProcessRegressor(
                    kernel=kernel, alpha=0.1, n_restarts_optimizer=0, random_state=42
                )
                try:
                    gp.fit(Xtr, ytr)
                except Exception:
                    gp = None

        # Compute UCB on unqueried cells
        unqueried = ~queried
        if unqueried.sum() == 0:
            break  # All downsampled cells queried, stop
        if gp is not None and unqueried.sum() > 0:
            Xte = candidates[unqueried]
            mu, sigma = gp.predict(Xte, return_std=True)
            ucb = mu + KAPPA * sigma
            # Avoid selecting near already-discovered points (min_dist)
            best = np.argmax(ucb)
            idx = np.where(unqueried)[0][best]
        else:
            idx = rng.choice(np.where(unqueried)[0])

        query_by_idx(idx)

    return len(found), oracle.query_count


def main():
    t_all = time.time()
    log("==== MC GP-UCB baseline (downsampled 50x50)====")
    rows = []
    for ws in WORLD_SEEDS:
        oracle = MCGridOracle(world_seed=ws)
        for ag in ALGO_SEEDS:
            pts, qs = run_gpucb(oracle, ag)
            rows.append([ws, ag, pts, qs])
            if len(rows) % 10 == 0:
                avg = sum(r[2] for r in rows) / len(rows)
                log(f"  gpucb progress {len(rows)}/{len(WORLD_SEEDS)*len(ALGO_SEEDS)}  avg points={avg:.1f}")

    out = OUT / "baseline_mc_gpucb.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["world_seed", "algo_seed", "points_found", "total_queries"])
        w.writerows(rows)

    mp = sum(r[2] for r in rows) / len(rows)
    mq = sum(r[3] for r in rows) / len(rows)
    hit = sum(1 for r in rows if r[2] >= TARGET) / len(rows)
    log(f"  gpucb  pts={mp:.2f}/{TARGET} q={mq:.0f}  hit rate={hit*100:.0f}%  ({(time.time()-t_all)/60:.1f}min)")
    log(f"==== Done -> {out}")


if __name__ == "__main__":
    main()
