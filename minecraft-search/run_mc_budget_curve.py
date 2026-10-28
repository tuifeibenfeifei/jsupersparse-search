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
"""run_mc_budget_curve.py — MC budget-recall curve sweep.

For each method, run it under budgets B in {500, 1000, 2000, 5000, 10000},
recording the mean number of deep-sea points found at that budget.
Output: results/minecraft_search/budget_curve.csv
"""
import csv
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(REPO_ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "species-search"))
sys.path.insert(0, str(ROOT / "minecraft-search"))

from oracle import SpeciesOracle  # noqa: E402
from species_agent import SpeciesSearcher  # noqa: E402
from mc_oracle import MCGridOracle  # noqa: E402

OUT = ROOT / "results" / "minecraft_search"
LOG = OUT / "progress.log"

WORLD_SEEDS = list(range(1, 21))
TARGET = 40
MIN_D = 2.0
BUDGETS = [500, 1000, 2000, 5000, 10000]

_D8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1),
       (1, -1), (1, 0), (1, 1)]


def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _mindist_ok(i, j, found, min_d):
    return all(math.hypot(i - fi, j - fj) >= min_d for fi, fj in found)


def run_random(oracle, algo_seed, budget):
    rng = random.Random(algo_seed)
    nrows, ncols = oracle.nrows, oracle.ncols
    found = []
    oracle.reset_counter()
    while oracle.query_count < budget and len(found) < TARGET:
        i, j = rng.randrange(nrows), rng.randrange(ncols)
        p, _ = oracle.query_cell(i, j)
        if p and _mindist_ok(i, j, found, MIN_D):
            found.append((i, j))
    return len(found)


def run_hill(oracle, algo_seed, budget):
    rng = random.Random(algo_seed)
    nrows, ncols = oracle.nrows, oracle.ncols
    stuck_limit = 6
    found = []
    oracle.reset_counter()
    cur = (rng.randrange(nrows), rng.randrange(ncols))
    climbed = 0
    while oracle.query_count < budget and len(found) < TARGET:
        i, j = cur
        p, _ = oracle.query_cell(i, j)
        if p is None:
            cur = (rng.randrange(nrows), rng.randrange(ncols))
            continue
        if p:
            if _mindist_ok(i, j, found, MIN_D):
                found.append((i, j))
            order = list(_D8)
            rng.shuffle(order)
            moved = False
            for di, dj in order:
                ni, nj = i + di, j + dj
                if not (0 <= ni < nrows and 0 <= nj < ncols):
                    continue
                p2, _ = oracle.query_cell(ni, nj)
                if p2:
                    if _mindist_ok(ni, nj, found, MIN_D):
                        found.append((ni, nj))
                    cur = (ni, nj)
                    moved = True
                    break
            if not moved:
                climbed += 1
                if climbed >= stuck_limit:
                    cur = (rng.randrange(nrows), rng.randrange(ncols))
                    climbed = 0
                else:
                    d = rng.choice(_D8)
                    cur = (min(nrows - 1, max(0, i + d[0])),
                           min(ncols - 1, max(0, j + d[1])))
        else:
            cur = (rng.randrange(nrows), rng.randrange(ncols))
    return len(found)


def framework_cfg(budget):
    return dict(
        target_points=TARGET, max_queries=budget, min_dist_cells=MIN_D,
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


def main():
    t_all = time.time()
    rows = []  # method, budget, world_seed, algo_seed, points

    # Framework: 20 algo seeds per world
    log("==== budget sweep: framework")
    for b in BUDGETS:
        cfg = framework_cfg(b)
        n_done = 0
        for ws in WORLD_SEEDS:
            npz = ROOT / "minecraft-search" / "oracle_npz" / f"mc_world_seed{ws}.npz"
            oracle = SpeciesOracle(npz)
            for ag in range(101, 121):
                res = SpeciesSearcher(oracle, cfg).run_episode(ag)
                rows.append(["framework", b, ws, ag, res["points_found"]])
                n_done += 1
        log(f"  budget={b:5d} done={n_done}")

    # Random: 5 algo seeds per world
    log("==== budget sweep: random")
    for b in BUDGETS:
        for ws in WORLD_SEEDS:
            oracle = MCGridOracle(world_seed=ws)
            for ag in range(101, 106):
                pts = run_random(oracle, ag, b)
                rows.append(["random", b, ws, ag, pts])
        log(f"  budget={b:5d} done")

    # Hill climb: 5 algo seeds per world
    log("==== budget sweep: hill_climb")
    for b in BUDGETS:
        for ws in WORLD_SEEDS:
            oracle = MCGridOracle(world_seed=ws)
            for ag in range(101, 106):
                pts = run_hill(oracle, ag, b)
                rows.append(["hill_climb", b, ws, ag, pts])
        log(f"  budget={b:5d} done")

    out = OUT / "budget_curve.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "budget", "world_seed", "algo_seed", "points_found"])
        w.writerows(rows)
    log(f"==== done -> {out} ({(time.time()-t_all)/60:.1f} min)")

    # print summary
    print("\n=== Budget-Recall Summary (mean points) ===")
    print(f"{'Method':12s}", end="")
    for b in BUDGETS:
        print(f"  B={b:5d}", end="")
    print()
    for m in ["framework", "random", "hill_climb"]:
        print(f"{m:12s}", end="")
        for b in BUDGETS:
            vals = [r[4] for r in rows if r[0] == m and r[1] == b]
            print(f"  {sum(vals)/len(vals):6.2f}", end="")
        print()


if __name__ == "__main__":
    main()
