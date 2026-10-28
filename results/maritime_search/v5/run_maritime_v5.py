# -*- coding: utf-8 -*-
"""run_maritime_v4.py — fourth round: maritime search-and-rescue Stage1 prior-focus (stage1_prior) cold-start validation.

- oracle: maritime-search/data/oracle.npz (150x150, 600 budget, SAR prior = covariate index 8).
- single process, sequential (WORKERS=1). 3 variants x 10 seeds (101-110).
- variants:
    prior_off       third-round original CFG (control; should be point-for-point identical to the v3 baseline)
    prior_on        + stage1_prior=True, stage1_prior_cov=8
    prior_on_motion + stage1_prior + motion_extrapolation=True
- outputs (in this script's v4/ directory): per_seed.csv, summary.csv, progress.log.
"""
import csv
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(r"C:\Users\19264\Desktop\deep-sea-exploration")
sys.path.insert(0, str(ROOT / "species-search"))
sys.path.insert(0, str(ROOT / "maritime-search"))

import numpy as np  # noqa: E402
from species_agent import SpeciesSearcher          # noqa: E402
from build_maritime_oracle import MaritimeOracle    # noqa: E402

HERE = Path(__file__).resolve().parent              # results/maritime_search/v4
HERE.mkdir(parents=True, exist_ok=True)
LOG = HERE / "progress.log"
NPZ = ROOT / "maritime-search" / "data" / "oracle.npz"
SEEDS = list(range(101, 111))                       # 10 seeds

CFG = dict(
    target_points=15, max_queries=600, min_dist_cells=1.0,
    stage1_grid_step=4,
    cov_indices=list(range(19)),
    multicluster=True, cluster_ring_ratio=10.0, cluster_idle_limit=12,
    blank_jump=True, coverage_budget_ratio=0.12, blank_far_ratio=0.3,
    post_jump_grace=6, saturate_mask=False, idle_blank_mix=8,
    sparse_guidance=True, sparse_topk=6, sparse_min_pos=5, recalc_every=20,
    stage2_min=3, stage3_min=5, stage3_jump_interval=30,
)

VARIANTS = {
    "prior_off":       dict(CFG),
    "prior_on":        dict(CFG, stage1_prior=True, stage1_prior_cov=8),
    "prior_on_motion": dict(CFG, stage1_prior=True, stage1_prior_cov=8,
                            motion_extrapolation=True),
}

logf = open(LOG, "w", encoding="utf-8")


def log(msg):
    print(msg, flush=True)
    logf.write(msg + "\n")
    logf.flush()


def first_hit_q(conv):
    """convergence: [(q, n_found, stage)...]; return the first q where n_found >= 1 (approximate, sampled every 100)"""
    for (q, n, st) in conv:
        if n >= 1:
            return q
    return -1


def main():
    t0 = time.time()
    oracle = MaritimeOracle(NPZ)
    log(f"oracle {oracle.nrows}x{oracle.ncols} covs={oracle.covs.shape} "
        f"R={oracle.R_km}km P0={oracle.P0_cell} variants={list(VARIANTS)} seeds=101-110")
    for vname, cfg in VARIANTS.items():
        rows = []
        for seed in SEEDS:
            ts = time.time()
            try:
                s = SpeciesSearcher(oracle, cfg)
                res = s.run_episode(seed)
                fhq = first_hit_q(res["convergence"])
                rows.append(dict(
                    seed=seed,
                    points_found=int(res["points_found"]),
                    total_queries=int(res["total_queries"]),
                    covered_cells=int(res["covered_cells"]),
                    final_stage=int(res["final_stage"]),
                    first_hit_q=fhq,
                    stage1_end_coverage=(res["stage1_end_coverage"]
                                         if res["stage1_end_coverage"] is not None else ""),
                ))
                log(f"[{vname}] seed={seed} pts={res['points_found']} "
                    f"q={res['total_queries']} cov={res['covered_cells']} "
                    f"stage={res['final_stage']} fhq={fhq} "
                    f"s1cov={res['stage1_end_coverage']} ({time.time()-ts:.1f}s)")
            except Exception:
                log(f"[ERROR] {vname}/seed{seed}")
                log(traceback.format_exc())
                rows.append(dict(seed=seed, points_found=-1, total_queries=-1,
                                 covered_cells=-1, final_stage=-1, first_hit_q=-1,
                                 stage1_end_coverage=""))
        with open(HERE / f"per_seed_{vname}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        pts = np.array([r["points_found"] for r in rows], float)
        qs = np.array([r["total_queries"] for r in rows], float)
        cov = np.array([r["covered_cells"] for r in rows], float)
        stg = [r["final_stage"] for r in rows]
        fhq = [r["first_hit_q"] for r in rows]
        log(f"[{vname}] mean pts={pts.mean():.2f} q={qs.mean():.0f} "
            f"cov={cov.mean():.0f} n_stage2plus={sum(1 for s in stg if s >= 2)}/{len(stg)} "
            f"fhq={fhq}")

    # summary.csv
    with open(HERE / "summary.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "n_seeds", "points_mean", "queries_mean",
                    "covered_mean", "n_stage2plus", "first_hit_q_list"])
        for vname in VARIANTS:
            with open(HERE / f"per_seed_{vname}.csv", encoding="utf-8") as rf:
                rows = list(csv.DictReader(rf))
            pts = np.array([int(r["points_found"]) for r in rows], float)
            qs = np.array([int(r["total_queries"]) for r in rows], float)
            cov = np.array([int(r["covered_cells"]) for r in rows], float)
            stg = [int(r["final_stage"]) for r in rows]
            fhq = [int(r["first_hit_q"]) for r in rows]
            w.writerow([vname, len(rows), round(pts.mean(), 2), round(qs.mean(), 1),
                        round(cov.mean(), 0), sum(1 for s in stg if s >= 2), fhq])
    log(f"[done] total time {time.time()-t0:.1f}s -> {HERE}")


if __name__ == "__main__":
    main()
