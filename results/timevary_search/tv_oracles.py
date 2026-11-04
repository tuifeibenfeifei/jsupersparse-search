# -*- coding: utf-8 -*-
"""tv_oracles.py - synthetic time-varying oracles for the drift scenarios.

Both oracles expose the same black-box interface as
``species_search.oracle.SpeciesOracle`` so that ``SpeciesSearcher`` can drive them
unchanged:

    oracle.nrows, oracle.ncols, oracle.covs (K, H, W), oracle.cov_names
    oracle.query_count, oracle.reset_counter()
    oracle.in_bounds(i, j)
    oracle.query_cell(i, j) -> (presence: bool, cov: np.ndarray)
    oracle.query(i, j)      -> (presence: bool, cov: np.ndarray)

Time advances by one virtual step every ``dt_queries`` queries. The presence field
is recomputed (lazily, on demand) from the current physical state, so the searcher
sees a genuinely non-stationary target: what it learned about the covariate ->
presence mapping can become stale.

``MovingPlumeOracle``
    A contaminant plume whose source drifts across the grid with a slowly rotating
    advection direction; presence is the plume body, covariates are smooth physical
    fields (temperature, salinity-like gradient, depth, distance to the drifting
    front). The mapping is learnable but the optimal region keeps moving, which is
    what a time-decay / follow strategy is supposed to exploit.

``AbruptJumpOracle``
    A stationary plume that relocates discontinuously at a few pre-scheduled query
    counts. Between jumps the mapping is stable, which is what mutation detection is
    supposed to notice; across a jump the previously learned guidance becomes
    actively misleading.
"""
from __future__ import annotations

import numpy as np


class _TimeVaryingOracle:
    """Shared bookkeeping: query counting, virtual clock and field access."""

    #: number of queries per virtual time step
    dt_queries = 20

    def __init__(self, nrows: int = 60, ncols: int = 60, sim_seed: int = 1234):
        self.nrows = int(nrows)
        self.ncols = int(ncols)
        self.sim_seed = int(sim_seed)
        self.query_count = 0

        ii, jj = np.meshgrid(np.arange(self.nrows), np.arange(self.ncols), indexing="ij")
        self._ii = ii.astype(np.float32)
        self._jj = jj.astype(np.float32)

        # Static covariates: normalised coordinates + a smooth depth-like field.
        # Time-varying structure is injected through the presence field and through
        # the "distance to front" covariate, which is refreshed with the state.
        self.cov_names = [
            "coord_i", "coord_j", "depth", "temp_base", "sal_base",
            "front_dist", "plume_prox",
        ]
        self.covs = np.zeros((len(self.cov_names), self.nrows, self.ncols), dtype=np.float32)
        self.covs[0] = self._ii / max(self.nrows - 1, 1)
        self.covs[1] = self._jj / max(self.ncols - 1, 1)
        rng = np.random.RandomState(self.sim_seed)
        yy = self.covs[0] * np.pi
        xx = self.covs[1] * np.pi
        self.covs[2] = (0.5 + 0.35 * np.sin(yy * 1.7) * np.cos(xx * 1.3)).astype(np.float32)
        self.covs[3] = (0.5 + 0.25 * np.sin(yy * 2.1 + 0.7)).astype(np.float32)
        self.covs[4] = (0.5 + 0.25 * np.cos(xx * 1.9 - 0.4)).astype(np.float32)
        self.covs[5] = 0.0
        self.covs[6] = 0.0

        self.presence = np.zeros((self.nrows, self.ncols), dtype=bool)
        self._state = None
        self._last_step = -1

    # -- clock ---------------------------------------------------------------
    @property
    def step(self) -> int:
        return self.query_count // self.dt_queries

    def reset_counter(self):
        self.query_count = 0
        self._state = None
        self._last_step = -1
        self._refresh(force=True)

    # -- black-box interface -------------------------------------------------
    def in_bounds(self, i, j) -> bool:
        return 0 <= int(i) < self.nrows and 0 <= int(j) < self.ncols

    def query_cell(self, i, j):
        """One grid query; an out-of-bounds call still consumes one budget unit."""
        self.query_count += 1
        if not self.in_bounds(i, j):
            return None, None
        self._maybe_advance()
        i = int(i)
        j = int(j)
        return bool(self.presence[i, j]), self.covs[:, i, j].copy()

    def query(self, i, j):
        return self.query_cell(i, j)

    # -- internals -----------------------------------------------------------
    def _maybe_advance(self):
        st = self.step
        if st != self._last_step:
            self._refresh(force=(self._last_step < 0))
            self._last_step = st

    def _refresh(self, force=False):  # pragma: no cover - overridden
        raise NotImplementedError

    def _write_fields(self, ci, cj, radius, angle, elong):
        """Render a plume and the derived covariates for a given source state."""
        di = self._ii - float(ci)
        dj = self._jj - float(cj)
        ca, sa = np.cos(angle), np.sin(angle)
        # rotate into the plume frame, then elongate along the advection direction
        u = di * ca + dj * sa
        v = -di * sa + dj * ca
        d2 = (u / max(elong, 1e-3)) ** 2 + v ** 2
        r = float(max(radius, 1e-3))
        prox = np.exp(-d2 / (2.0 * r * r)).astype(np.float32)
        self.presence = prox > 0.55
        self.covs[5] = np.sqrt(d2).astype(np.float32) / float(max(self.nrows, self.ncols))
        self.covs[6] = prox


class MovingPlumeOracle(_TimeVaryingOracle):
    """A plume whose source drifts smoothly across the domain (slow, learnable drift)."""

    dt_queries = 20

    def _refresh(self, force=False):
        t = float(self.step)
        # smooth large-scale drift: a Lissajous path across the central region
        ci = self.nrows * (0.5 + 0.28 * np.sin(0.11 * t + 0.4))
        cj = self.ncols * (0.5 + 0.28 * np.cos(0.09 * t - 0.2))
        angle = 0.6 + 0.35 * np.sin(0.05 * t)
        radius = 0.10 * min(self.nrows, self.ncols)
        elong = 2.2 + 0.6 * np.sin(0.07 * t)
        self._state = (ci, cj, radius, angle, elong)
        self._write_fields(ci, cj, radius, angle, elong)


class AbruptJumpOracle(_TimeVaryingOracle):
    """A deterministic plume that teleports to a new site at scheduled query counts."""

    dt_queries = 20
    #: query counts at which the plume relocates discontinuously
    jump_schedule = (300, 700, 1100)

    def _refresh(self, force=False):
        # choose the site index from how many scheduled jumps have already happened
        k = sum(1 for q in self.jump_schedule if self.query_count >= q)
        rng = np.random.RandomState(self.sim_seed + 101 * k)
        ci = rng.uniform(0.15, 0.85) * self.nrows
        cj = rng.uniform(0.15, 0.85) * self.ncols
        angle = rng.uniform(0.0, np.pi)
        radius = 0.09 * min(self.nrows, self.ncols)
        elong = 1.8
        self._state = (ci, cj, radius, angle, elong)
        self._write_fields(ci, cj, radius, angle, elong)
