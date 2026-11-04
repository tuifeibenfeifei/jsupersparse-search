# -*- coding: utf-8 -*-
"""run_compare.py - shared environment and GP-based baselines for the supplementary checks.

This module is imported by the scripts in supplement_stats/ that need a common,
budget-parameterised search environment plus two surrogate-model baselines:

  Env         - thin wrapper around a SpeciesOracle, tracks found points, visited
                cells and query history, and honours the module-level budget
                knobs TARGET / MAX_Q / MIN_D (several callers reassign them
                before running: e.g. peerj_resume_mc sets TARGET=40, MAX_Q=10000,
                MIN_D=2.0 for the Minecraft scenes).
  EpisodeCtx  - precomputed normalised structural covariate fields, used as the
                feature space of the GP baselines.
  algo_gp_ucb - batched GP-UCB (mu + kappa * sigma) over a random candidate pool.
  algo_turbo  - simplified TuRBO: trust region around the best point, shrink on
                failure, expand and re-centre on a new discovery.

The implementations mirror the greedy/hill-climb baselines in
harmful_arms_paired.py so that the framework and the baselines share identical
query, visibility and minimum-distance semantics.
"""
from __future__ import annotations

import math
import random

import numpy as np

# Module-level budget knobs. Callers that target a different scene reassign
# these before constructing an Env; defaults match the chlorophyll scene.
TARGET = 20
MAX_Q = 4000
MIN_D = 1.0
STRUCT = list(range(7))


# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------


class Env:
    """Budget-aware wrapper around an oracle with recording of the search history."""

    def __init__(self, oracle):
        self.o = oracle
        self.reset()

    def reset(self):
        self.o.reset_counter()
        self.found = []
        self.visited = set()
        self.pos_cov = []
        self.X_hist = []
        self.y_hist = []

    @property
    def q(self):
        return self.o.query_count

    @property
    def done(self):
        return self.q >= MAX_Q or len(self.found) >= TARGET

    def query(self, i, j):
        i = int(i)
        j = int(j)
        pres, cov = self.o.query_cell(i, j)
        if pres is None:
            return None
        self.visited.add((i, j))
        is_new = False
        if pres:
            if all(math.hypot(i - fi, j - fj) >= MIN_D for fi, fj in self.found):
                self.found.append((i, j))
                self.pos_cov.append(cov.copy())
                is_new = True
        self.X_hist.append((i, j))
        self.y_hist.append(1.0 if pres else 0.0)
        return pres, cov, is_new

    def result(self):
        total = self.o.nrows * self.o.ncols
        return dict(
            points_found=len(self.found),
            total_queries=self.q,
            covered_cells=len(self.visited),
            covered_ratio=len(self.visited) / total if total else 0.0,
            success=len(self.found) >= TARGET,
        )


# ----------------------------------------------------------------------------
# Feature space
# ----------------------------------------------------------------------------


def build_feature_fields(o):
    """Normalise the structural covariates into an (H, W, len(STRUCT)) float32 field."""
    H, W = o.nrows, o.ncols
    covf = np.zeros((H, W, len(STRUCT)), dtype=np.float32)
    for kk, k in enumerate(STRUCT):
        v = o.covs[k].astype(np.float32)
        lo, hi = float(v.min()), float(v.max())
        if hi - lo < 1e-9:
            hi = lo + 1.0
        covf[..., kk] = (v - lo) / (hi - lo)
    return covf


class EpisodeCtx:
    """Normalised covariate grid plus the raw min/max needed to normalise discoveries."""

    def __init__(self, o):
        self.Xgrid = build_feature_fields(o)
        self.H, self.W = o.nrows, o.ncols
        mins, maxs = [], []
        for k in STRUCT:
            v = o.covs[k].astype(np.float32)
            mins.append(float(v.min()))
            maxs.append(float(v.max()))
        self._cov_min = np.array(mins, dtype=np.float32)
        self._cov_max = np.array(maxs, dtype=np.float32)

    def norm_pos_cov(self, pos_cov_raw_list):
        if not pos_cov_raw_list:
            return np.zeros((0, len(STRUCT)), dtype=np.float32)
        arr = np.array(pos_cov_raw_list)[:, STRUCT]
        arr = (arr - self._cov_min) / (self._cov_max - self._cov_min + 1e-9)
        return arr.astype(np.float32)


# ----------------------------------------------------------------------------
# GP helpers
# ----------------------------------------------------------------------------


def _fit_gp(Xtr, ytr):
    """Fit a zero-optimiser RBF GP; returns None when there is nothing to learn."""
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF
    except Exception:
        return None
    Xtr = np.asarray(Xtr, dtype=np.float64)
    ytr = np.asarray(ytr, dtype=np.float64)
    if Xtr.ndim != 2 or Xtr.shape[0] < 2:
        return None
    if not np.isfinite(Xtr).all() or not np.isfinite(ytr).all():
        return None
    g = GaussianProcessRegressor(
        kernel=RBF(length_scale=0.5), optimizer=None, alpha=1e-3, normalize_y=False
    )
    try:
        g.fit(Xtr, ytr)
    except Exception:
        return None
    return g


def _hist_features(ctx, env, n=None):
    """Build the training set from the query history (cell coordinates + covariates)."""
    Xg = ctx.Xgrid
    ncov = Xg.shape[-1]
    rows = []
    for (i, j) in env.X_hist:
        coord = (i / max(ctx.H - 1, 1), j / max(ctx.W - 1, 1))
        rows.append(np.concatenate([np.asarray(coord, dtype=np.float32), Xg[i, j]]))
    X = np.asarray(rows, dtype=np.float64).reshape(-1, 2 + ncov)
    y = np.asarray(env.y_hist, dtype=np.float64)[: X.shape[0]]
    if n is not None and X.shape[0] > n:
        X = X[-n:]
        y = y[-n:]
    return X, y


def _gp_candidates(ctx, env, rng, pool=2000):
    """Sample unvisited cells as GP candidates; returns (coords, features)."""
    H, W = ctx.H, ctx.W
    ncov = ctx.Xgrid.shape[-1]
    cand = []
    feats = []
    tries = 0
    limit = pool * 20
    while len(cand) < pool and tries < limit:
        tries += 1
        i = rng.randrange(H)
        j = rng.randrange(W)
        if (i, j) in env.visited:
            continue
        coord = (i / max(H - 1, 1), j / max(W - 1, 1))
        cand.append((i, j))
        feats.append(np.concatenate([np.asarray(coord, dtype=np.float32), ctx.Xgrid[i, j]]))
    if not cand:
        return [], np.zeros((0, 2 + ncov), dtype=np.float64)
    return cand, np.asarray(feats, dtype=np.float64).reshape(-1, 2 + ncov)


# ----------------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------------


def algo_gp_ucb(env, rng, ctx, cold=12, kappa=2.0, pool=2000, batch=20):
    """Batched GP-UCB: cold-start random probes, then mu + kappa * sigma over a pool."""
    H, W = ctx.H, ctx.W
    while not env.done and env.q < cold:
        env.query(rng.randrange(H), rng.randrange(W))
    queued = []
    while not env.done:
        if not queued:
            Xtr, ytr = _hist_features(ctx, env, n=300)
            g = _fit_gp(Xtr, ytr)
            if g is None:
                queued = [(rng.randrange(H), rng.randrange(W)) for _ in range(batch)]
            else:
                cand, Xc = _gp_candidates(ctx, env, rng, pool=pool)
                if not cand:
                    break
                mu, std = g.predict(Xc, return_std=True)
                order = np.argsort(mu + kappa * std)[::-1][:batch]
                queued = [cand[k] for k in order]
        i, j = queued.pop(0)
        env.query(i, j)
    return env.result()


def algo_turbo(env, rng, ctx, cold=12, pool=300, batch=8, init_radius=0.25):
    """Simplified TuRBO with a shrinking/expanding trust region around the incumbent."""
    H, W = ctx.H, ctx.W
    while not env.done and env.q < cold:
        env.query(rng.randrange(H), rng.randrange(W))
    center = np.array([rng.uniform(0, H - 1), rng.uniform(0, W - 1)])
    radius = init_radius * max(H, W)
    queued = []
    while not env.done:
        if not queued:
            Xtr, ytr = _hist_features(ctx, env, n=250)
            g = _fit_gp(Xtr, ytr)
            n_samp = pool * 3
            offs = np.array([rng.gauss(0, radius) for _ in range(n_samp * 2)]).reshape(n_samp, 2)
            pts = center[None, :] + offs
            cands = []
            for pt in pts:
                i = int(round(pt[0]))
                j = int(round(pt[1]))
                if 0 <= i < H and 0 <= j < W and (i, j) not in env.visited:
                    cands.append((i, j))
                    if len(cands) >= pool:
                        break
            if len(cands) < batch:
                for _ in range(batch - len(cands)):
                    i = rng.randrange(H)
                    j = rng.randrange(W)
                    if (i, j) not in env.visited:
                        cands.append((i, j))
            if g is None or not cands:
                queued = cands[:batch]
            else:
                # Feature layout must match _hist_features: [coord_i, coord_j, covariates...]
                Xc = np.array([
                    np.concatenate([
                        np.asarray((i / max(H - 1, 1), j / max(W - 1, 1)), dtype=np.float32),
                        ctx.Xgrid[i, j],
                    ])
                    for (i, j) in cands
                ], dtype=np.float64)
                mu, std = g.predict(Xc, return_std=True)
                order = np.argsort(mu + 1.5 * std)[::-1][:batch]
                queued = [cands[k] for k in order]
            if not queued:
                break
        i, j = queued.pop(0)
        out = env.query(i, j)
        if out is None:
            break
        pres, cov, is_new = out
        if is_new:
            center = np.array([float(i), float(j)])
            radius = min(radius * 1.5, 0.5 * max(H, W))
        else:
            radius *= 0.98
        if radius < 2.0:
            center = np.array([rng.uniform(0, H - 1), rng.uniform(0, W - 1)])
            radius = init_radius * max(H, W)
    return env.result()
