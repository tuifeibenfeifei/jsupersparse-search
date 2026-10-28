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
"""run_mc_framework.py — Run the latest 5-round optimized framework on the Minecraft deep-sea scenario.

Convention (fully aligned with the three baselines run_mc_baselines.py):
  - World seeds 1..20
  - Each world seed × each algo_seed runs 1 episode
  - target=40, max_queries=10000, min_dist_cells=2.0
  - cov_indices=[0,1,2,3,4] (use all 5; let Presence-AUC sparsify automatically)

Output results/minecraft_search/:
  framework_mc_detail.csv   per-run detail
  framework_mc_summary.csv   summary
"""
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(REPO_ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "species-search"))

from oracle import SpeciesOracle  # noqa: E402
from species_agent import SpeciesSearcher  # noqa: E402

OUT = ROOT / "results" / "minecraft_search"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "progress.log"

WORLD_SEEDS = list(range(1, 21))
ALGO_SEEDS = list(range(101, 121))  # 20 repetitions
TARGET = 40
MAX_Q = 10000
MIN_D = 2.0

# 5-round recommended config: all mechanisms enabled, but time_decay/mutation/follow will not trigger in static scenarios
CFG = dict(
    target_points=TARGET,
    max_queries=MAX_Q,
    min_dist_cells=MIN_D,
    stage2_min=3,
    stage3_min=5,
    stage3_jump_interval=30,
    stage3_jump_min_ratio=1.25,
    stage3_jump_max_ratio=5.0,
    curiosity_weight=1.0,
    guidance_on=True,
    cov_in_state=True,
    cov_indices=[0, 1, 2, 3, 4],
    # Multi-cluster split pursuit (A11)
    multicluster=True,
    cluster_ring_ratio=10.0,
    cluster_idle_limit=12,
    blank_jump=True,
    coverage_budget_ratio=0.12,
    # 5-round mechanisms (do not trigger in static scenarios, but kept for config consistency)
    time_decay=True,
    decay_mode="exp",
    decay_tau=300.0,
    mutation_detect=True,
    mutation_streak_thresh=40,
    follow_mode=True,
    mutation_disperse_jump=True,
    mutation_jump_interval=15,
)


def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    t_all = time.time()
    log(f"==== Framework Minecraft deep-sea comparison: world_seeds={WORLD_SEEDS[0]}-{WORLD_SEEDS[-1]} "
        f"algo_seeds={ALGO_SEEDS[0]}-{ALGO_SEEDS[-1]} target={TARGET} budget={MAX_Q}")

    rows = []
    for ws in WORLD_SEEDS:
        npz = ROOT / "minecraft-search" / "oracle_npz" / f"mc_world_seed{ws}.npz"
        oracle = SpeciesOracle(npz)
        for ag in ALGO_SEEDS:
            t0 = time.time()
            res = SpeciesSearcher(oracle, CFG).run_episode(ag)
            pts = res["points_found"]
            qs = res["total_queries"]
            stage = res["final_stage"]
            dt = time.time() - t0
            rows.append([ws, ag, pts, qs, stage, round(dt, 2)])
            if len(rows) % 20 == 0:
                avg_pts = sum(r[2] for r in rows) / len(rows)
                avg_q = sum(r[3] for r in rows) / len(rows)
                log(f"  Progress {len(rows)}/{len(WORLD_SEEDS)*len(ALGO_SEEDS)}  "
                    f"current avg pts={avg_pts:.1f} avg queries={avg_q:.0f}")

    # Per-run detail
    det = OUT / "framework_mc_detail.csv"
    with open(det, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["world_seed", "algo_seed", "points_found",
                    "total_queries", "final_stage", "wall_time_s"])
        w.writerows(rows)

    n = len(rows)
    avg_pts = sum(r[2] for r in rows) / n
    avg_q = sum(r[3] for r in rows) / n
    hit = sum(1 for r in rows if r[2] >= TARGET) / n
    min_p = min(r[2] for r in rows)
    max_p = max(r[2] for r in rows)

    with open(OUT / "framework_mc_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "mean_points", "mean_queries",
                    "hit_rate", "min_points", "max_points", "n_runs"])
        w.writerow(["framework_v5", round(avg_pts, 2), round(avg_q, 1),
                    round(hit, 3), min_p, max_p, n])

    log(f"  framework  pts={avg_pts:.2f}/{TARGET} q={avg_q:.0f} "
        f"hit rate={hit*100:.0f}%  (total time {(time.time()-t_all)/60:.1f} min)")
    log(f"==== Done -> {OUT}")


if __name__ == "__main__":
    main()
