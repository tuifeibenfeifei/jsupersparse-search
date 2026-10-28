# -*- coding: utf-8 -*-
"""oracle.py — rare-species black-box query interface (species-version cubiomes_wrapper + database)
Black-box semantics: given (lat, lon), return whether the target species is present at that location + the covariate vector at that location.
Every query increments query_count. Isomorphic to the user framework's cubiomes_wrapper.get_temperature/is_deep_ocean:
    cubiomes:  (x,z) -> (temperature, whether deep sea)
    species:   (lat,lon) -> (covariates, whether present)
"""
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data" / "oracle.npz"


class SpeciesOracle:
    """load the pre-built grid oracle and provide black-box queries"""

    def __init__(self, npz_path: Path = DATA):
        d = np.load(npz_path)
        self.lon0 = float(d["lon0"])
        self.lat0 = float(d["lat0"])
        self.cell = float(d["cell_deg"])
        self.ncols = int(d["ncols"])
        self.nrows = int(d["nrows"])
        self.presence = d["presence"]          # (nrows, ncols) bool
        self.covs = d["covs"]                  # (K, nrows, ncols) float32
        self.cov_names = [str(x) for x in d["cov_names"]]
        self.study_poly = d["study_poly"]      # [lon_min, lon_max, lat_min, lat_max]
        self.query_count = 0

    # ---- coordinate conversion ----
    def latlon_to_cell(self, lat: float, lon: float):
        i = int((lat - self.lat0) / self.cell)
        j = int((lon - self.lon0) / self.cell)
        if i < 0 or j < 0 or i >= self.nrows or j >= self.ncols:
            return None
        return i, j

    def cell_to_latlon(self, i: int, j: int):
        return self.lat0 + (i + 0.5) * self.cell, self.lon0 + (j + 0.5) * self.cell

    def in_bounds(self, i: int, j: int) -> bool:
        return 0 <= i < self.nrows and 0 <= j < self.ncols

    # ---- black-box query: one query = one count ----
    def query(self, lat: float, lon: float):
        """return (presence: bool, cov_vector: np.ndarray); return (None, None) when out of bounds"""
        cell = self.latlon_to_cell(lat, lon)
        if cell is None:
            return None, None
        i, j = cell
        self.query_count += 1
        return bool(self.presence[i, j]), self.covs[:, i, j].copy()

    def query_cell(self, i: int, j: int):
        """grid query: an out-of-bounds query also consumes one budget unit (black-box semantics), return (None, None)"""
        self.query_count += 1
        if not self.in_bounds(i, j):
            return None, None
        return bool(self.presence[i, j]), self.covs[:, i, j].copy()

    # ---- statistics used by experiments ----
    @property
    def total_cells(self):
        return self.nrows * self.ncols

    def presence_cells(self):
        return [tuple(c) for c in np.argwhere(self.presence)]

    def reset_counter(self):
        self.query_count = 0


def covariate_guidance_table(oracle: SpeciesOracle, found_cells, n_bins: int = 6):
    """Stage2 data accumulation: compute the covariate percentile interval of found presence cells (parameter -> presence-probability guidance table)
    return {cov_index: (lo, hi)}: the middle interval (25%-75% percentile) of presence cells, used for guidance scoring
    """
    if not found_cells:
        return None
    pts = np.array([[oracle.covs[k, i, j] for i, j in found_cells]
                    for k in range(oracle.covs.shape[0])])  # (K, n)
    table = {}
    for k in range(oracle.covs.shape[0]):
        v = pts[k]
        lo = np.percentile(v, 25)
        hi = np.percentile(v, 75)
        if hi - lo < 1e-9:
            lo, hi = np.min(v), np.max(v)
        table[k] = (float(lo), float(hi))
    return table


def guidance_score(oracle: SpeciesOracle, i: int, j: int, table, cov_indices):
    """guidance score: the fraction of covariates falling inside the presence interval (0-1)"""
    if table is None:
        return 0.0
    hit = 0
    for k in cov_indices:
        lo, hi = table[k]
        v = float(oracle.covs[k, i, j])
        if lo <= v <= hi:
            hit += 1
    return hit / len(cov_indices)
