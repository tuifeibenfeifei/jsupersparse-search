# -*- coding: utf-8 -*-
"""补跑代理增长微基准 + 汇总统计。"""
import csv, statistics, sys, time
from pathlib import Path
ROOT = Path(r"C:\Users\19264\Desktop\deep-sea-exploration")
sys.path.insert(0, str(ROOT / "species-search"))
sys.path.insert(0, str(ROOT / "chlorophyll-search"))
sys.path.insert(0, str(ROOT))
import numpy as np
from oracle import SpeciesOracle  # noqa: E402
from species_agent import SpeciesSearcher  # noqa: E402
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

CHL_CFG = dict(
    target_points=20, max_queries=4000, min_dist_cells=1.0,
    stage2_min=3, stage3_min=5, stage3_jump_interval=30,
    stage3_jump_min_ratio=1.25, stage3_jump_max_ratio=5.0,
    curiosity_weight=1.0, learning_rate=0.1, discount_factor=0.95,
    epsilon_start=0.9, epsilon_end=0.1, epsilon_decay=0.999,
    stage1_grid_step=3, guidance_on=True, cov_in_state=True,
    recalc_every=20, stage2_topk=8, sparse_guidance=True, sparse_topk=6, sparse_min_pos=5,
    multicluster=True, cluster_ring_ratio=10.0, cluster_idle_limit=12,
    blank_jump=True, coverage_budget_ratio=0.12, blank_far_ratio=0.3, post_jump_grace=6,
    saturate_mask=False, cov_indices=list(range(19)), adaptive=True, directed_jump=True,
)

OUT = ROOT / "supplement_stats"

def main():
    rng = np.random.RandomState(0)
    oracle = SpeciesOracle(ROOT / "chlorophyll-search" / "data" / "oracle.npz")
    # build Xgrid like EpisodeCtx
    import sys as _s; _s.path.insert(0, str(ROOT / "chlorophyll-search" / "compare_oldnew"))
    from run_compare import EpisodeCtx
    ctx = EpisodeCtx(oracle)
    Xgrid = ctx.Xgrid.reshape(-1, ctx.Xgrid.shape[-1])

    rows = []
    for n in [100, 500, 1000, 2000, 4000]:
        idx = rng.choice(len(Xgrid), size=n, replace=False)
        Xtr = Xgrid[idx].astype(np.float64)
        ytr = np.random.RandomState(n).randn(n) * 0.1
        cand_idx = rng.choice(len(Xgrid), size=2000, replace=False)
        Xcand = Xgrid[cand_idx].astype(np.float64)
        times = []
        for rep in range(3):
            t0 = time.perf_counter()
            g = GaussianProcessRegressor(kernel=RBF(length_scale=0.5), optimizer=None,
                                         alpha=1e-3, normalize_y=False)
            g.fit(Xtr, ytr)
            _ = g.predict(Xcand, return_std=True)
            times.append(time.perf_counter() - t0)
        med = statistics.median(times)
        rows.append(dict(n=n, gp_fit_plus_predict_s=round(med, 5),
                         reps=",".join(f"{t:.5f}" for t in times)))
        print(f"n={n:5d} gp_step_median={med*1000:.2f} ms  reps={[round(t*1000,1) for t in times]}", flush=True)

    # framework per-query
    oracle2 = SpeciesOracle(ROOT / "chlorophyll-search" / "data" / "oracle.npz")
    t0 = time.perf_counter()
    res = SpeciesSearcher(oracle2, CHL_CFG).run_episode(101)
    fw_wall = time.perf_counter() - t0
    fw_per_q = fw_wall / max(res["total_queries"], 1)
    print(f"framework: wall={fw_wall:.3f}s q={res['total_queries']} per_q={fw_per_q*1000:.3f} ms", flush=True)
    for r in rows:
        r["framework_per_query_s"] = round(fw_per_q, 6)
        r["ratio_gp_over_framework"] = round(r["gp_fit_plus_predict_s"] / fw_per_q, 1)

    with open(OUT / "peerj_proxy_growth.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["n","gp_fit_plus_predict_s","reps","framework_per_query_s","ratio_gp_over_framework"])
        w.writeheader(); w.writerows(rows)
    print("proxy done", flush=True)

if __name__ == "__main__":
    main()
