# -*- coding: utf-8 -*-
"""common.py - shared framework for the rare-species black-box search comparison experiment.

Protocol (hard version: highly dispersed multi-target hard comparison):
  target=80, max_queries=60000, min_dist_cells=4.0, seeds=101..120。
Black-box semantics: one oracle.query_cell(i,j) = 1 unit of budget (out-of-bounds still counts).
Target criterion: presence=True AND grid Euclidean distance to every already-found point >= min_dist before it counts as found.
No cheating: an agent may observe presence only through query_cell; it may use the global covariates oracle.covs
for guidance / model inputs (the framework also builds its guidance map from covs, so the protocol matches).
"""
from __future__ import annotations
import sys, time, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SPECIES_DIR = HERE.parent  # species-search/
sys.path.insert(0, str(SPECIES_DIR))
from oracle import SpeciesOracle  # noqa: E402

DATA = SPECIES_DIR / "data"
ORACLE_PATHS = {
    "black_muntjac": DATA / "black_muntjac" / "oracle.npz",
    "clouded_leopard": DATA / "clouded_leopard" / "oracle.npz",
}

TARGET = 80
MAX_QUERIES = 60000
MIN_DIST = 4.0
SEEDS = list(range(101, 121))  # 20 seeds


def make_oracle(species: str) -> SpeciesOracle:
    return SpeciesOracle(ORACLE_PATHS[species])


# ---------------------------------------------------------------------------
# Global covariate standardisation (used only for agent scoring/modelling, costs no budget)
# ---------------------------------------------------------------------------
class CovGuide:
    """Score guidance from the covariate centroid of discovered positives; also exposes whole-grid standardised covariates."""

    def __init__(self, oracle: SpeciesOracle):
        self.oracle = oracle
        # z-score the whole-grid covariates (over all grid cells, including ocean / non-occurrence areas, for normalisation only)
        covs = oracle.covs.astype(np.float64)  # (K, R, C)
        K, R, C = covs.shape
        flat = covs.reshape(K, -1)
        self.mu = flat.mean(axis=1)
        self.sd = flat.std(axis=1) + 1e-9
        # standardised whole grid (R, C, K)
        self.zcov = ((covs - self.mu[:, None, None]) / self.sd[:, None, None]).transpose(1, 2, 0)
        self.found_centroid = None  # (K,) centroid of positives in standardised space
        self.found_covs_z = []

    def note_found(self, i: int, j: int):
        v = self.zcov[i, j]
        self.found_covs_z.append(v)
        self.found_centroid = np.mean(self.found_covs_z, axis=0)

    def guide(self, i: int, j: int) -> float:
        """Guidance score in 0..1: higher the closer to the covariate centroid of positives; returns 0.5 when there are no positives yet."""
        if self.found_centroid is None:
            return 0.5
        d = np.mean(np.abs(self.zcov[i, j] - self.found_centroid))
        return float(math.exp(-d / 1.0))


# ---------------------------------------------------------------------------
# Agent base class
# ---------------------------------------------------------------------------
class BaseAgent:
    name = "base"
    kind = "old"  # old / new

    def __init__(self, oracle: SpeciesOracle, rng: np.random.Generator, guide: CovGuide):
        self.oracle = oracle
        self.rng = rng
        self.guide = guide
        self.R = oracle.nrows
        self.C = oracle.ncols
        self.visited = set()  # in-bounds (i,j) already queried
        self.found = []       # (i,j) points already counted as found

    def in_bounds(self, i, j):
        return 0 <= i < self.R and 0 <= j < self.C

    def propose(self):
        raise NotImplementedError

    def update(self, i, j, presence, cov, is_new_found):
        if self.in_bounds(i, j):
            self.visited.add((i, j))
        if is_new_found:
            self.found.append((i, j))
            self.guide.note_found(i, j)


# ---------------------------------------------------------------------------
# Unified experiment loop
# ---------------------------------------------------------------------------
def run_one(oracle: SpeciesOracle, agent_factory, seed: int,
            target: int = TARGET, max_queries: int = MAX_QUERIES,
            min_dist: float = MIN_DIST, verbose: bool = False) -> dict:
    oracle.reset_counter()
    rng = np.random.default_rng(seed)
    guide = CovGuide(oracle)
    agent = agent_factory(oracle, rng, guide)

    found = []
    visited = set()
    t0 = time.time()
    while oracle.query_count < max_queries and len(found) < target:
        i, j = agent.propose()
        if i is None or j is None:
            # agent has no candidates left: fall back to random
            i = int(rng.integers(0, oracle.nrows))
            j = int(rng.integers(0, oracle.ncols))
        presence, cov = oracle.query_cell(i, j)
        is_new_found = False
        if presence is True:
            ok = all(math.hypot(i - fi, j - fj) >= min_dist - 1e-9 for fi, fj in found)
            if ok:
                found.append((i, j))
                is_new_found = True
        if oracle.in_bounds(i, j):
            visited.add((i, j))
        agent.update(i, j, presence, cov, is_new_found)

    elapsed = time.time() - t0
    total = oracle.nrows * oracle.ncols
    return {
        "algo": agent.name,
        "kind": agent.kind,
        "seed": seed,
        "points_found": len(found),
        "queries": oracle.query_count,
        "covered_cells": len(visited),
        "covered_ratio": len(visited) / total,
        "success": len(found) >= target,
        "elapsed_s": round(elapsed, 2),
    }
