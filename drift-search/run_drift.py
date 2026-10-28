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
"""run_drift.py - maritime search and rescue reverse-drift sequential decoding two-stage A->B experiment.

sys.path imports reuse species-search/species_agent.py::SpeciesSearcher (framework itself is not modified).
Black-box oracle = drift-search/drift_oracle.py::DriftOracle (physics simulation).

Pipeline (one full A->B per seed, run sequentially in a single process):
  Stage A: current space-time, SpeciesSearcher searches the frozen debris snapshot (budget=400, target=6, 19 cov).
  Decode wrapper: take A's discovered points, backward advect each point by -(C + L_hat*W)*T_A to get candidate wreck points,
            centroid + uncertainty radius (+/-10 cells); construct a "distance to decoded centroid" covariate.
  Stage B: past space-time, a new SpeciesSearcher + augmented oracle (20 cov, presence=P0 static evidence)
            searches for wreck points (budget=300, target=3).

Outputs:
  results/drift_search/drift_detail.csv   per seed
  results/drift_search/drift_summary.csv  mean over 3 seeds
  results/drift_search/_run_artifacts.npz for plotting/reporting
"""
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(REPO_ROOT)
sys.path.insert(0, str(ROOT / "species-search"))        # framework itself
sys.path.insert(0, str(ROOT / "drift-search"))          # this scenario's oracle

from species_agent import SpeciesSearcher     # noqa: E402
from drift_oracle import DriftOracle          # noqa: E402

OUT = ROOT / "results" / "drift_search"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "progress.log"

SEEDS = [101, 102, 103]

# F-final default keys (same as species/regress_opt_F BASE); override budget/target/dims per task spec
CFG_A = dict(
    target_points=6, max_queries=400, min_dist_cells=2.0,
    stage2_min=3, stage3_min=5, stage3_jump_interval=30,
    stage3_jump_min_ratio=1.25, stage3_jump_max_ratio=5.0,
    curiosity_weight=1.0, learning_rate=0.1, discount_factor=0.95,
    epsilon_start=0.9, epsilon_end=0.1, epsilon_decay=0.999,
    stage1_grid_step=8, guidance_on=True, cov_in_state=True,
    recalc_every=10, stage2_topk=8,
    sparse_guidance=True, sparse_topk=6, sparse_min_pos=5,
    multicluster=True, cluster_ring_ratio=10.0, cluster_idle_limit=12,
    blank_jump=True, coverage_budget_ratio=0.12, blank_far_ratio=0.3,
    post_jump_grace=6, saturate_mask=False, idle_blank_mix=8,
    cov_indices=list(range(19)),
)
CFG_B = dict(
    target_points=3, max_queries=300, min_dist_cells=2.0,
    stage2_min=3, stage3_min=5, stage3_jump_interval=30,
    stage3_jump_min_ratio=1.25, stage3_jump_max_ratio=5.0,
    curiosity_weight=1.0, learning_rate=0.1, discount_factor=0.95,
    epsilon_start=0.9, epsilon_end=0.1, epsilon_decay=0.999,
    stage1_grid_step=8, guidance_on=True, cov_in_state=True,
    recalc_every=10, stage2_topk=8,
    sparse_guidance=True, sparse_topk=6, sparse_min_pos=3,
    multicluster=True, cluster_ring_ratio=10.0, cluster_idle_limit=12,
    blank_jump=True, coverage_budget_ratio=0.12, blank_far_ratio=0.3,
    post_jump_grace=6, saturate_mask=False, idle_blank_mix=8,
    cov_indices=list(range(20)),
)

DECODE_RADIUS_CELL = 10   # decode uncertainty radius (cells), +/-10 cells = 20 km


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def auc_scores_from_queries(queries, cov_indices):
    """Reproduce the framework's Mann-Whitney AUC (direction-agnostic), measuring per-covariate separability from the final queried samples."""
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
    # Stage A oracle (covs/presence are constant across seeds; reuse one instance)
    oracleA = DriftOracle(ROOT / "drift-search" / "data" / "stageA.npz")
    cell = oracleA.cell_km
    P0 = (oracleA.P0_x, oracleA.P0_y)
    # Decoded representative quantities
    cu_p0, cv_p0 = oracleA.cu_p0, oracleA.cv_p0
    wu_m, wv_m = oracleA.wu_mean, oracleA.wv_mean
    L_hat = oracleA.leeway_hat
    T_A = oracleA.T_A_h
    disp_dec = np.array([cu_p0 + L_hat * wu_m * 3.6,
                         cv_p0 + L_hat * wv_m * 3.6]) * T_A   # km
    log(f"StageA oracle {oracleA.nrows}x{oracleA.ncols} presence={int(oracleA.presence.sum())} "
        f"({oracleA.presence.mean()*100:.2f}%)  P0=({P0[0]:.1f},{P0[1]:.1f})km")
    log(f"Decoded displacement=({disp_dec[0]:.2f},{disp_dec[1]:.2f})km  L_hat={L_hat} T_A={T_A}h")

    # Coordinate grid (km) for constructing Stage B's 20th covariate
    ii, jj = np.meshgrid(np.arange(oracleA.nrows), np.arange(oracleA.ncols), indexing="ij")
    Xg = (jj + 0.5) * cell
    Yg = (ii + 0.5) * cell

    detail_rows = []
    artifacts = {"seeds": []}

    for seed in SEEDS:
        t0 = time.time()
        # ---------- Stage A ----------
        searcherA = SpeciesSearcher(oracleA, CFG_A)
        resA = searcherA.run_episode(seed)
        foundA = resA["found_cells"]

        # ---------- Decode wrapper: backward advect discovered points ----------
        if foundA:
            fk = np.array(foundA, dtype=float)
            fx = (fk[:, 1] + 0.5) * cell   # east km
            fy = (fk[:, 0] + 0.5) * cell   # north km
            cand = np.stack([fx, fy], axis=1) - disp_dec[None, :]
            centroid = cand.mean(axis=0)   # (cx_km, cy_km)
            err_km = float(np.hypot(centroid[0] - P0[0], centroid[1] - P0[1]))
            cx_c = centroid[0] / cell - 0.5
            cy_c = centroid[1] / cell - 0.5
        else:
            centroid = np.array([np.nan, np.nan])
            err_km = float("nan")
            cx_c = cy_c = float("nan")

        # ---------- Stage B: augmented oracle (append 20th covariate) ----------
        oracleB = DriftOracle(ROOT / "drift-search" / "data" / "stageB.npz")
        if foundA:
            dist_band = np.sqrt((Xg - centroid[0]) ** 2 + (Yg - centroid[1]) ** 2)
        else:
            dist_band = np.zeros_like(Xg)
        oracleB.append_covariate(dist_band, "distance to decoded centroid(km)")

        searcherB = SpeciesSearcher(oracleB, CFG_B)
        resB = searcherB.run_episode(seed)
        foundB = resB["found_cells"]

        # B hit: whether a point is found inside the P0 static evidence region (presenceB is the P0 disk, so foundB>0 means hit)
        B_hit = bool(len(foundB) > 0)
        # Distance from B's discovered points to the true P0 (km)
        if foundB:
            fb = np.array(foundB, dtype=float)
            fb_x = (fb[:, 1] + 0.5) * cell
            fb_y = (fb[:, 0] + 0.5) * cell
            B_to_P0_km = float(np.hypot(fb_x - P0[0], fb_y - P0[1]).mean())
        else:
            B_to_P0_km = float("nan")

        # Stage B AUC (20 covariates)
        aucB = auc_scores_from_queries(searcherB._queries, list(range(20)))
        keptB = searcherB._feature_log[-1][2] if searcherB._feature_log else []
        # Whether the 20th covariate was selected by the framework
        sel_idx = 19
        sel_weight = dict(keptB).get(sel_idx, 0.0)
        sel_auc = aucB.get(sel_idx, 0.5)
        selected = bool(sel_weight > 0.0)

        dt = time.time() - t0
        log(f"seed={seed} A: pts={resA['points_found']}/6 q={resA['total_queries']} "
            f"| decoded_centroid=({centroid[0]:.1f},{centroid[1]:.1f})km err={err_km:.1f}km "
            f"| B: pts={resB['points_found']}/3 q={resB['total_queries']} hit={B_hit} "
            f"B_to_P0={B_to_P0_km:.1f}km | AUC[dist_to_centroid]={sel_auc:.3f} selected={selected} "
            f"elapsed={dt:.1f}s")

        # Visited-cell heatmap
        visA = np.zeros((oracleA.nrows, oracleA.ncols), dtype=np.uint8)
        for (i, j) in searcherA.visited_cells:
            if oracleA.in_bounds(i, j):
                visA[i, j] = 1
        visB = np.zeros((oracleB.nrows, oracleB.ncols), dtype=np.uint8)
        for (i, j) in searcherB.visited_cells:
            if oracleB.in_bounds(i, j):
                visB[i, j] = 1

        detail_rows.append([
            seed,
            resA["points_found"], resA["total_queries"],
            round(err_km, 2),
            resB["points_found"], resB["total_queries"],
            int(B_hit), round(B_to_P0_km, 2) if B_to_P0_km == B_to_P0_km else "",
            round(sel_auc, 3), int(selected),
        ])
        artifacts["seeds"].append({
            "seed": seed,
            "foundA": np.array(foundA, dtype=int).reshape(-1, 2),
            "visA": visA,
            "convA": np.array(resA["convergence"], dtype=float),
            "foundB": np.array(foundB, dtype=int).reshape(-1, 2),
            "visB": visB,
            "convB": np.array(resB["convergence"], dtype=float),
            "centroid_km": centroid,
            "err_km": err_km,
            "B_to_P0_km": B_to_P0_km,
            "aucB": aucB,
            "keptB": keptB,
            "n_posB": int(sum(1 for q in searcherB._queries if q[1])),
            "n_qB": len(searcherB._queries),
        })

    # ---- detail.csv ----
    with open(OUT / "drift_detail.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "A_points", "A_queries", "decode_centroid_err_km",
                    "B_points", "B_queries", "B_final_hit", "B_to_P0_km",
                    "auc_dist_centroid", "B_selected_dist_cov"])
        w.writerows(detail_rows)

    # ---- summary.csv (mean over 3 seeds) ----
    def _mean(idx):
        vals = [r[idx] for r in detail_rows if isinstance(r[idx], (int, float)) and r[idx] == r[idx]]
        return sum(vals) / len(vals) if vals else float("nan")
    with open(OUT / "drift_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "seed101", "seed102", "seed103"])
        w.writerow(["A_points(/6)", round(_mean(1), 2), *[r[1] for r in detail_rows]])
        w.writerow(["A_queries", round(_mean(2), 1), *[r[2] for r in detail_rows]])
        w.writerow(["decode_centroid_err_km", round(_mean(3), 2), *[r[3] for r in detail_rows]])
        w.writerow(["B_points(/3)", round(_mean(4), 2), *[r[4] for r in detail_rows]])
        w.writerow(["B_queries", round(_mean(5), 1), *[r[5] for r in detail_rows]])
        w.writerow(["B_final_hit", int(sum(r[6] for r in detail_rows)), *[r[6] for r in detail_rows]])
        w.writerow(["B_to_P0_km", round(_mean(7), 2), *[r[7] for r in detail_rows]])
        w.writerow(["auc_dist_centroid", round(_mean(8), 3), *[r[8] for r in detail_rows]])
        w.writerow(["B_selected_dist_cov", int(sum(r[9] for r in detail_rows)),
                    *[r[9] for r in detail_rows]])

    # ---- artifacts.npz ----
    np.savez(
        OUT / "_run_artifacts.npz",
        allow_pickle=True,
        presenceA=oracleA.presence,
        presenceB=oracleB.presence,
        debris_cell=oracleA.debris_cell,
        P0_i=oracleA.P0_i, P0_j=oracleA.P0_j, P0_x=oracleA.P0_x, P0_y=oracleA.P0_y,
        cell_km=cell, n=oracleA.nrows,
        disp_dec=disp_dec,
        cov_namesB=np.array(oracleB.cov_names),
        seeds=np.array(SEEDS),
        per_seed=np.array(artifacts["seeds"], dtype=object),
    )
    log(f"mean: Apts={_mean(1):.2f}/6 Aq={_mean(2):.0f} err={_mean(3):.1f}km "
        f"Bpts={_mean(4):.2f}/3 hit={sum(r[6] for r in detail_rows)}/3")
    log("done.")


if __name__ == "__main__":
    main()




