# -*- coding: utf-8 -*-
import os
from pathlib import Path
REPO_ROOT = Path(__file__).resolve()
for _ in range(8):
    if (REPO_ROOT / 'configs').is_dir():
        break
    REPO_ROOT = REPO_ROOT.parent
"""mutation_streak_thresh sensitivity sweep (moving_plume + abrupt_jump scenes).

Threshold levels 10/20/30/40(default)/60/80; protocol N=60, MAX_Q=1500, TARGET=20, MIN_D=2.0, seeds 101-110.
Base config FIX_MOTION (see results/timevary_search/v5/_diag_v5.py, decay_tau=300, sparse_topk=5).
Output: per-variant CSV + summary CSV under supplement_stats/mutation_sweep/.
"""
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "mutation_sweep"
OUTDIR.mkdir(parents=True, exist_ok=True)
PROJECT = Path(REPO_ROOT)
sys.path.insert(0, str(PROJECT / "species-search"))
sys.path.insert(0, str(PROJECT / "results" / "timevary_search"))

from species_agent import SpeciesSearcher  # noqa: E402
from tv_oracles import MovingPlumeOracle, AbruptJumpOracle  # noqa: E402

N = 60
MAX_Q = 1500
TARGET = 20
MIN_D = 2.0
SEEDS = list(range(101, 111))

FRAME = dict(target_points=TARGET, max_queries=MAX_Q, min_dist_cells=MIN_D,
             sparse_guidance=True, sparse_min_pos=5, sparse_topk=5, recalc_every=20)
DECAY = dict(time_decay=True, decay_mode="exp", decay_tau=300.0)
BASE = dict(FRAME, **DECAY, mutation_detect=True, mutation_auto_motion=True)

THRS = [10, 20, 30, 40, 60, 80]
LOG = OUTDIR / "mutation_sweep.log"
if LOG.exists():
    LOG.unlink()


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_scene(scene_cls, tag):
    log(f"--- {tag}: mutation_streak_thresh = {THRS} ---")
    summary_rows = []
    per_seed_rows = []
    for thr in THRS:
        cfg = dict(BASE, mutation_streak_thresh=float(thr))
        rows = []
        for seed in SEEDS:
            o = scene_cls(nrows=N, ncols=N, sim_seed=1234)
            res = SpeciesSearcher(o, cfg).run_episode(algo_seed=seed)
            rows.append(dict(scene=tag, thr=thr, seed=seed,
                             pts=res["points_found"], q=res["total_queries"],
                             n_hits=len(res["found_cells"]),
                             stage=res["final_stage"]))
        pts = np.array([r["pts"] for r in rows], float)
        qs = np.array([r["q"] for r in rows], float)
        reach = float(np.mean(pts >= TARGET))
        summary_rows.append(dict(scene=tag, thr=thr, pts_mean=round(pts.mean(), 3),
                                 q_mean=round(qs.mean(), 1), reach=round(reach, 3),
                                 q_min=int(qs.min()), q_max=int(qs.max())))
        log(f"  thr={thr:3d} pts={pts.mean():5.2f} q={qs.mean():7.1f} "
            f"reach={reach:.2f}")
        per_seed_rows.extend(rows)
    # sensitivity: Spearman(thr vs q_mean) and trend delta
    if len(THRS) > 3:
        rho, p_rho = stats.spearmanr(THRS, [r["q_mean"] for r in summary_rows])
        log(f"  Spearman(thr, q) = {rho:+.3f} p={p_rho:.3f}")
    with open(OUTDIR / f"mutation_{tag}_summary.csv", "w", encoding="utf-8",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    with open(OUTDIR / f"mutation_{tag}_perseed.csv", "w", encoding="utf-8",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_seed_rows[0].keys()))
        w.writeheader(); w.writerows(per_seed_rows)
    return summary_rows


def main():
    t0 = time.time()
    mp = run_scene(MovingPlumeOracle, "moving_plume")
    aj = run_scene(AbruptJumpOracle, "abrupt_jump")
    log(f"[done] total {time.time() - t0:.1f}s")
    with open(OUTDIR / "mutation_combined_summary.csv", "w", encoding="utf-8",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mp[0].keys()))
        w.writeheader()
        w.writerows(mp + aj)


if __name__ == "__main__":
    main()
