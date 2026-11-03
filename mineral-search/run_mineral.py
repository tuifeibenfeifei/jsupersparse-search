# -*- coding: utf-8 -*-
"""run_mineral.py - two-stage sequential-decoding experiment, mineralized outcrop to subsurface ore body
================================================================================
Stage 1 (A, 2-D surface): SpeciesSearcher searches the 120x120 surface mineralized outcrop.
Decoding (inside this wrapper): RANSAC line fit on the stage-A discoveries gives the main-structure strike theta_hat and the trace centroid;
  a 3-D subsurface structural plane is inferred with the same strike and a 70 deg dip, producing the stage-B covariate "distance to the predicted structural plane".
Stage 2 (B, 3-D subsurface): the decoded guidance channel is injected into Oracle3D and Searcher3D searches the 80x80x40 ore body.
A and B are different space-time blocks and their maps are never fused; block B is shifted 20 km south-west of A (see build_mineral_oracle).
The framework itself is not modified; it is imported and reused via sys.path, and all decoding logic lives in this wrapper.
Seeds 101/102/103 each run one complete A to B pass, sequentially in a single process.
"""
import csv
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve()
while not (REPO_ROOT / "configs").is_dir():
    if REPO_ROOT.parent == REPO_ROOT:
        raise RuntimeError("repository root (containing configs/) not found")
    REPO_ROOT = REPO_ROOT.parent
ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "species-search"))
sys.path.insert(0, str(ROOT / "ocean3d-search"))
sys.path.insert(0, str(ROOT / "mineral-search"))

from species_agent import SpeciesSearcher        # noqa: E402
from species_agent_3d import Searcher3D        # noqa: E402
from build_mineral_oracle import (MineralOracle2D, MineralOracle3D,  # noqa: E402
                                  signed_dist_to_plane_3d, DIP_DEG, CX_SHIFT)

MIN_DIR = ROOT / "mineral-search"
DATA = MIN_DIR / "data"
OUT = ROOT / "results" / "mineral_search"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "progress.log"

SEEDS = [101, 102, 103]

# Stage-A configuration (final F defaults; budget and target per the task)
CFG_A = dict(
    target_points=10, max_queries=500, min_dist_cells=1.0,
    stage2_min=3, stage3_min=5, stage3_jump_interval=30,
    stage3_jump_min_ratio=1.25, stage3_jump_max_ratio=5.0,
    curiosity_weight=1.0, learning_rate=0.1, discount_factor=0.95,
    epsilon_start=0.9, epsilon_end=0.1, epsilon_decay=0.999,
    stage1_grid_step=8, guidance_on=True, cov_in_state=True,
    recalc_every=20, stage2_topk=8,
    sparse_guidance=True, sparse_topk=6, sparse_min_pos=5,
    multicluster=True, cluster_ring_ratio=10.0, cluster_idle_limit=12,
    blank_jump=True, coverage_budget_ratio=0.12, blank_far_ratio=0.3,
    post_jump_grace=6, saturate_mask=False, idle_blank_mix=8,
)
# Stage-B configuration (final F defaults; budget, target and stage-1 step per the task)
CFG_B = dict(
    target_points=20, max_queries=1500, min_dist_cells=1.0,
    stage2_min=3, stage3_min=5, stage3_jump_interval=30,
    stage3_jump_min_ratio=1.25, stage3_jump_max_ratio=5.0,
    curiosity_weight=1.0, learning_rate=0.1, discount_factor=0.95,
    epsilon_start=0.9, epsilon_end=0.1, epsilon_decay=0.999,
    stage1_grid_step=3, guidance_on=True, cov_in_state=True,
    recalc_every=20, stage2_topk=8,
    sparse_guidance=True, sparse_topk=6, sparse_min_pos=5,
    multicluster=True, cluster_ring_ratio=10.0, cluster_idle_limit=12,
    blank_jump=True, coverage_budget_ratio=0.12, blank_far_ratio=0.3,
    post_jump_grace=6, saturate_mask=False, idle_blank_mix=8,
)

TRUE_STRIKE = 75.0     # True strike of the main mineralizing structure (META)
RANSAC_INLIER_KM = 2.0


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ==================== Decoding wrapper: stage-A discoveries -> strike theta_hat + predicted plane ====================
def ransac_strike(found_cells_A, n_iter=300, seed=0):
    """RANSAC line fit on the stage-A discoveries; returns (strike_deg, cx_B, cy_B, inlier_mask).
    found_cells_A: [(i,j)] 2-D cell coordinates. World coordinates x = j+0.5 (east), y = 119.5-i (north).
    The returned cx_B,cy_B are the trace centroid mapped into block-B world coordinates (CX_SHIFT subtracted)."""
    pts = np.array([(j + 0.5, 119.5 - (i + 0.5)) for (i, j) in found_cells_A],
                   dtype=np.float64)
    n = len(pts)
    if n < 2:
        return TRUE_STRIKE, 40.0, 40.0, np.zeros(n, bool)
    rng = np.random.default_rng(seed)
    best = None  # (n_inlier, rms, vx, vy, cx, cy, mask)
    for _ in range(n_iter):
        a, b = rng.choice(n, size=2, replace=False)
        p1, p2 = pts[a], pts[b]
        v = p2 - p1
        norm = np.hypot(*v)
        if norm < 1e-6:
            continue
        v = v / norm
        # perpendicular distance to the line p1 + t*v
        d = np.abs((pts[:, 0] - p1[0]) * v[1] - (pts[:, 1] - p1[1]) * v[0])
        inl = d <= RANSAC_INLIER_KM
        n_inl = int(inl.sum())
        if n_inl < 2:
            continue
        # projected span along the line (degeneracy guard)
        t = (pts[inl, 0] - p1[0]) * v[0] + (pts[inl, 1] - p1[1]) * v[1]
        span = t.max() - t.min()
        if span < 3.0:
            continue
        rms = float(np.sqrt(np.mean(d[inl] ** 2)))
        if best is None or (n_inl, -rms) > (best[0], -best[1]):
            # trace centroid = mean of the inlier points
            centroid = pts[inl].mean(axis=0)
            best = (n_inl, rms, v[0], v[1], centroid[0], centroid[1], inl.copy())
    if best is None:
        # degenerate fallback: principal axis via PCA
        c = pts.mean(axis=0)
        cov = np.cov((pts - c).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        v = eigvecs[:, np.argmax(eigvals)]
        best = (n, 0.0, v[0], v[1], c[0], c[1], np.ones(n, bool))
    _, _, vx, vy, cxA, cyA, mask = best
    # direction (vx, vy) = (east, north) -> strike theta = 90 deg - atan2(vy,vx), normalised to [0,180)
    alpha = np.rad2deg(np.arctan2(vy, vx))
    strike = (90.0 - alpha) % 180.0
    # block-A world centroid -> block-B world coordinates (south-west shift CX_SHIFT)
    cx_B = cxA - CX_SHIFT
    cy_B = cyA - CX_SHIFT
    return float(strike), float(cx_B), float(cy_B), mask


def strike_err_deg(hat, true=TRUE_STRIKE):
    d = abs(hat - true) % 180.0
    return float(min(d, 180.0 - d))


def build_predicted_plane_map(oracleB, strike_hat, cx_B, cy_B):
    """Build the stage-B covariate 'distance to the predicted structural plane' over the full KxLxJ grid."""
    jj, ii = np.meshgrid(np.arange(oracleB.J) + 0.5, np.arange(oracleB.L) + 0.5)
    x2d = jj
    y2d = (oracleB.L - 1 - ii + 0.5)
    z3 = (np.arange(oracleB.K) + 0.5) * oracleB.dz
    XX = np.broadcast_to(x2d[None, :, :], (oracleB.K, oracleB.L, oracleB.J))
    YY = np.broadcast_to(y2d[None, :, :], (oracleB.K, oracleB.L, oracleB.J))
    ZZ = np.broadcast_to(z3[:, None, None], (oracleB.K, oracleB.L, oracleB.J))
    sd = signed_dist_to_plane_3d(XX, YY, ZZ, cx_B, cy_B, strike_hat, DIP_DEG)
    # Distance to the plane is non-negative (absolute value); a signed version makes the background symmetric about 0 and collapses the AUC to 0.5
    return np.abs(sd).astype(np.float32)


# ==================== Offline full-grid AUC (true ore body vs background) ====================
def offline_auc_per_cov(oracleB):
    """Mann-Whitney AUC for each covariate (ore=True vs ore=False). Direction-free: take max(auc, 1-auc)."""
    ore = oracleB.ore_mask           # (K,L,J) bool
    neg = ~ore
    aucs = []
    for k in range(oracleB.covs.shape[0]):
        v = oracleB.covs[k]
        a = v[ore].ravel()
        b = v[neg].ravel()
        if len(a) == 0 or len(b) == 0:
            aucs.append(0.5)
            continue
        # sub-sampling control
        if len(a) > 500:
            a = a[np.random.default_rng(0).choice(len(a), 500, replace=False)]
        if len(b) > 2000:
            b = b[np.random.default_rng(1).choice(len(b), 2000, replace=False)]
        x = np.concatenate([a, b])
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty(len(x)); ranks[order] = np.arange(1, len(x) + 1)
        rsum = ranks[:len(a)].sum()
        u = rsum - len(a) * (len(a) + 1) / 2.0
        auc = u / (len(a) * len(b))
        aucs.append(max(auc, 1.0 - auc))
    return aucs


# ==================== Main flow ====================
def main():
    if LOG.exists():
        LOG.unlink()
    log("=== Two-stage sequential decoding, mineralized outcrop to subsurface ore body: START (synthetic geological model / physics simulation) ===")
    log(f"stage-A oracle = {DATA/'stageA_surface.npz'}")
    oraA = MineralOracle2D(DATA / "stageA_surface.npz")
    log(f"A: {oraA.nrows}x{oraA.ncols}, presence={int(oraA.presence.sum())} "
        f"({oraA.presence.mean()*100:.2f}%)")

    detail_rows = []
    artifacts = {}
    t_start = time.time()
    for seed in SEEDS:
        log(f"--- seed={seed} stage A: start ---")
        resA = SpeciesSearcher(oraA, CFG_A).run_episode(seed)
        A_pts = int(resA["points_found"])
        A_q = int(resA["total_queries"])
        foundA = resA["found_cells"]
        log(f"  A done: found={A_pts}/{CFG_A['target_points']}, queries={A_q}, "
            f"final_stage={resA['final_stage']}")

        # ---- decoding: strike estimation ----
        strike_hat, cx_B, cy_B, inl = ransac_strike(foundA, seed=seed)
        serr = strike_err_deg(strike_hat, TRUE_STRIKE)
        log(f"  decode: theta_hat={strike_hat:.1f} deg (true {TRUE_STRIKE} deg, error {serr:.1f} deg), "
            f"trace centroid B=({cx_B:.1f},{cy_B:.1f}), RANSAC inliers={int(inl.sum())}/{len(foundA)}")

        # ---- stage B: inject the predicted-plane channel, then run Searcher3D ----
        # Predicted plane: strike from the decoded theta_hat; position anchored at block-B centre (40,40) (regional geological prior:
        # outcrop decoding mainly yields the strike, and the trace position inside the survey block is anchored at the block centre;
        # the biased discovery centroid is not carried over directly). Dip follows the geological prior of 70 deg.
        oraB = MineralOracle3D(DATA / "stageB_subsurface.npz")
        pred_map = build_predicted_plane_map(oraB, strike_hat, 40.0, 40.0)
        oraB.inject_predicted_plane(pred_map)
        resB = Searcher3D(oraB, CFG_B).run_episode(seed)
        B_pts = int(resB["points_found"])
        B_q = int(resB["total_queries"])
        foundB = resB["found_cells"]

        # B_on_structure: fraction of discoveries landing on the true ore sheet
        on_struct = [bool(oraB.ore_mask[c[0], c[1], c[2]]) for c in foundB]
        b_on_struct = float(np.mean(on_struct)) if on_struct else 0.0
        # fraction landing on random weak anomalies (false positives)
        on_noise = [bool(oraB.noise_anom[c[0], c[1], c[2]]) for c in foundB]
        b_on_noise = float(np.mean(on_noise)) if on_noise else 0.0

        # offline AUC after injecting the predicted plane for this seed
        aucs = offline_auc_per_cov(oraB)
        auc_predplane = aucs[1]
        top_cov = int(np.argmax(aucs))
        log(f"  B done: found={B_pts}/{CFG_B['target_points']}, queries={B_q}, "
            f"B_on_structure={b_on_struct:.2f}, on_noise={b_on_noise:.2f}, "
            f"AUC[dist_pred_plane]={auc_predplane:.3f}, top_cov={top_cov}"
            f"({oraB.cov_names[top_cov]})")

        detail_rows.append(dict(
            seed=seed, A_points=A_pts, A_queries=A_q,
            strike_hat=round(strike_hat, 1), strike_err_deg=round(serr, 1),
            B_points=B_pts, B_queries=B_q,
            B_on_structure=round(b_on_struct, 3), B_on_noise=round(b_on_noise, 3),
            AUC_predplane=round(auc_predplane, 3),
            top_cov=top_cov, top_cov_name=oraB.cov_names[top_cov],
        ))
        artifacts[seed] = dict(
            foundA=foundA, inlier_mask=inl, strike_hat=strike_hat,
            foundB=foundB, on_struct=on_struct, on_noise=on_noise,
            aucs=aucs, cov_names=list(oraB.cov_names),
            ore_mask=oraB.ore_mask, pred_map=pred_map,
            strike_err=serr,
        )
        log(f"  seed={seed} elapsed {(time.time()-t_start)/60:.1f} min")

    # ---- write the detail CSV ----
    detail_csv = OUT / "mineral_detail.csv"
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        w.writeheader()
        w.writerows(detail_rows)
    log(f"detail CSV -> {detail_csv}")

    # ---- summary CSV ----
    n = len(detail_rows)
    def mean(k): return sum(r[k] for r in detail_rows) / n
    summary = dict(
        A_points_mean=mean("A_points"), A_queries_mean=mean("A_queries"),
        strike_err_mean=mean("strike_err_deg"),
        B_points_mean=mean("B_points"), B_queries_mean=mean("B_queries"),
        B_on_structure_mean=mean("B_on_structure"),
        B_on_noise_mean=mean("B_on_noise"),
        AUC_predplane_mean=mean("AUC_predplane"),
    )
    with open(OUT / "mineral_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean_over_seeds"])
        for k, v in summary.items():
            w.writerow([k, round(v, 3)])
    log(f"summary CSV -> {OUT/'mineral_summary.csv'}")

    # ---- save artefacts for plotting ----
    np.savez(OUT / "_run_artifacts.npz",
             seeds=np.array(SEEDS),
             detail=np.array([[r[k] for k in r] for r in detail_rows], dtype=object),
             detail_keys=np.array(list(detail_rows[0].keys())),
             allow_pickle=True)
    # per-seed key visualisation data
    vis = {}
    for seed in SEEDS:
        a = artifacts[seed]
        vis[f"foundA_{seed}"] = np.array(a["foundA"])
        vis[f"foundB_{seed}"] = np.array(a["foundB"])
        vis[f"on_struct_{seed}"] = np.array(a["on_struct"])
        vis[f"aucs_{seed}"] = np.array(a["aucs"])
        vis[f"strike_hat_{seed}"] = a["strike_hat"]
        vis[f"strike_err_{seed}"] = a["strike_err"]
    # store ore_mask once (geologically fixed)
    vis["ore_mask"] = artifacts[SEEDS[0]]["ore_mask"]
    vis["cov_names"] = np.array(artifacts[SEEDS[0]]["cov_names"])
    np.savez(OUT / "_vis.npz", **vis)
    log(f"artifacts -> {OUT/'_vis.npz'}")
    log(f"=== ALL DONE, elapsed {(time.time()-t_start)/60:.1f} min ===")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        log(traceback.format_exc())
        raise
