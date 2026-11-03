# -*- coding: utf-8 -*-
"""run_pollution.py - abrupt pollution tracing scene: transfer experiment for the black-box online target-search framework.

Imported through sys.path to reuse species-search/species_agent.py::SpeciesSearcher (the framework itself is not modified).
Black-box oracle = pollution-search/pollution_oracle.py::PollutionOracle (physics-simulated plume).

Configuration (as specified by the task brief):
  cov_indices=19, budget=800, target_points=15, stage1_grid_step=3,
  min_dist_cells=1.0; all other final-F default keys are the same as in the species regression.
Runs seeds 101/102/103 sequentially in one process; ThreadPoolExecutor is not used.

Outputs:
  results/pollution_search/pollution_detail.csv   per seed
  results/pollution_search/pollution_summary.csv   3-seed mean
  results/pollution_search/_run_artifacts.npz     per-seed detail for plotting and reporting
"""
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve()
while not (REPO_ROOT / "configs").is_dir():
    if REPO_ROOT.parent == REPO_ROOT:
        raise RuntimeError("repository root (containing configs/) not found")
    REPO_ROOT = REPO_ROOT.parent
ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "species-search"))   # framework core
sys.path.insert(0, str(ROOT / "pollution-search"))  # scene oracle

from species_agent import SpeciesSearcher   # noqa: E402
from pollution_oracle import PollutionOracle  # noqa: E402

OUT = ROOT / "results" / "pollution_search"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "progress.log"

SEEDS = [101, 102, 103]

# Final-F default keys (same BASE as results/species_search/regress_opt_F/run_species_regress_F.py),
# overridden per the task brief: budget=800, target=15, stage1_grid_step=3
CFG = dict(
    target_points=15, max_queries=800, min_dist_cells=1.0,
    stage2_min=3, stage3_min=5, stage3_jump_interval=30,
    stage3_jump_min_ratio=1.25, stage3_jump_max_ratio=5.0,
    curiosity_weight=1.0, learning_rate=0.1, discount_factor=0.95,
    epsilon_start=0.9, epsilon_end=0.1, epsilon_decay=0.999,
    stage1_grid_step=3, guidance_on=True, cov_in_state=True,
    recalc_every=20, stage2_topk=8,
    sparse_guidance=True, sparse_topk=6, sparse_min_pos=5,
    # Final-F optimisation keys (listed explicitly by the task brief)
    multicluster=True, cluster_ring_ratio=10.0, cluster_idle_limit=12,
    blank_jump=True, coverage_budget_ratio=0.12, blank_far_ratio=0.3,
    post_jump_grace=6, saturate_mask=False, idle_blank_mix=8,
    # High-dimensional noise pool: all 19 covariates are fed in and sparse AUC-based selection rejects the noise automatically
    cov_indices=list(range(19)),
)


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def auc_scores_from_queries(queries, cov_indices):
    """Reproduce the framework Mann-Whitney AUC (direction-free, max(auc,1-auc)),
    computing the separability of each covariate from the final queried sample. Returns {k: auc}."""
    import numpy as np
    import math
    covs = np.array([q[0] for q in queries], dtype=float)
    pres = np.array([q[1] for q in queries], dtype=float)
    out = {}
    pos_idx = np.where(pres > 0.5)[0]
    neg_idx = np.where(pres < 0.5)[0]
    for k in cov_indices:
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            out[k] = 0.5
            continue
        pv = covs[pos_idx, k]
        nv = covs[neg_idx, k]
        x = np.concatenate([pv, nv])
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty(len(x), dtype=float)
        ranks[order] = np.arange(1, len(x) + 1)
        xs = x[order]
        i, L = 0, len(xs)
        while i < L:
            j = i
            while j + 1 < L and xs[j + 1] == xs[i]:
                j += 1
            if j > i:
                ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        n, m = len(pv), len(nv)
        rsum = float(ranks[:n].sum())
        u = rsum - n * (n + 1) / 2.0
        auc = u / (n * m)
        out[k] = max(auc, 1.0 - auc)
    return out


def main():
    open(LOG, "w", encoding="utf-8").close()
    oracle = PollutionOracle()
    log(f"oracle loaded: {oracle.nrows}x{oracle.ncols}, "
        f"presence={int(oracle.presence.sum())} ({oracle.plume_frac*100:.1f}%), "
        f"src=({oracle.source_i},{oracle.source_j})")
    log(f"cfg: budget=800 target=15 stage1_step=3 cov=19, seeds={SEEDS}")

    detail_rows = []
    artifacts = {"seeds": []}
    sx, sy = oracle.source_i, oracle.source_j
    _d = np.load(ROOT / "pollution-search/data/oracle.npz")
    DI_F = float(_d["flow_di"]); DJ_F = float(_d["flow_dj"])

    for seed in SEEDS:
        t0 = time.time()
        searcher = SpeciesSearcher(oracle, CFG)
        res = searcher.run_episode(seed)
        dt = time.time() - t0

        found = res["found_cells"]
        visited = np.zeros((oracle.nrows, oracle.ncols), dtype=np.uint8)
        for (i, j) in searcher.visited_cells:
            if oracle.in_bounds(i, j):
                visited[i, j] = 1

        # tracing error: Euclidean distance (cells) from the most upstream detected point to the true source
        if found:
            s_found = [(i - sx) * DI_F + (j - sy) * DJ_F for (i, j) in found]
            up_idx = int(np.argmin(s_found))
            up = found[up_idx]
            trace_err = float(((up[0] - sx) ** 2 + (up[1] - sy) ** 2) ** 0.5)
        else:
            up = (-1, -1)
            trace_err = float("nan")

        # final AUC selection (separability of all 19 covariates)
        auc = auc_scores_from_queries(searcher._queries, list(range(19)))
        # features finally selected by the framework (last entry of feature_log)
        kept = []
        if searcher._feature_log:
            kept = searcher._feature_log[-1][2]   # [(k, weight)...]

        log(f"seed={seed} pts={res['points_found']}/15 q={res['total_queries']} "
            f"cov={res['covered_cells']} stage={res['final_stage']} "
            f"upstream={up} trace_err={trace_err:.2f} cells  elapsed {dt:.1f}s")

        detail_rows.append([
            seed, res["points_found"], res["total_queries"],
            res["covered_cells"], res["final_stage"],
            up[0], up[1], round(trace_err, 3),
        ])
        artifacts["seeds"].append({
            "seed": seed,
            "found_cells": np.array(found, dtype=int).reshape(-1, 2),
            "visited": visited,
            "convergence": np.array(res["convergence"], dtype=float),
            "auc": auc,
            "kept": kept,
            "n_pos_final": int(sum(1 for q in searcher._queries if q[1])),
            "n_q_final": len(searcher._queries),
        })

    # ---- detail.csv ----
    with open(OUT / "pollution_detail.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "points_found", "total_queries", "covered_cells",
                     "final_stage", "upstream_i", "upstream_j", "trace_err_cells"])
        w.writerows(detail_rows)

    # ---- summary.csv（3-seed mean）----
    n = len(detail_rows)
    mp = sum(r[1] for r in detail_rows) / n
    mq = sum(r[2] for r in detail_rows) / n
    mc = sum(r[3] for r in detail_rows) / n
    mt = sum(r[7] for r in detail_rows if not (r[7] != r[7])) / max(1, sum(1 for r in detail_rows if r[7] == r[7]))
    with open(OUT / "pollution_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "seed101", "seed102", "seed103"])
        w.writerow(["points_found", round(mp, 2), detail_rows[0][1], detail_rows[1][1], detail_rows[2][1]])
        w.writerow(["total_queries", round(mq, 1), detail_rows[0][2], detail_rows[1][2], detail_rows[2][2]])
        w.writerow(["covered_cells", round(mc, 1), detail_rows[0][3], detail_rows[1][3], detail_rows[2][3]])
        w.writerow(["trace_err_cells", round(mt, 2), detail_rows[0][7], detail_rows[1][7], detail_rows[2][7]])
        w.writerow(["trace_err_km", round(mt * 0.5, 2),
                     round(detail_rows[0][7] * 0.5, 2),
                     round(detail_rows[1][7] * 0.5, 2),
                     round(detail_rows[2][7] * 0.5, 2)])

    # ---- artifacts.npz (for plotting) ----
    np.savez(
        OUT / "_run_artifacts.npz",
        allow_pickle=True,
        presence=oracle.presence,
        source_i=sx, source_j=sy,
        cov_names=np.array(oracle.cov_names),
        n_real=np.array([8]),
        seeds=np.array(SEEDS),
        per_seed=np.array(artifacts["seeds"], dtype=object),
    )
    log(f"mean: pts={mp:.2f}/15 q={mq:.0f} cov={mc:.0f} trace_err={mt:.2f} cells ({mt*0.5:.1f} km)")
    log("done.")


if __name__ == "__main__":
    main()
