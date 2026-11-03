# -*- coding: utf-8 -*-
"""pollution_oracle.py - black-box oracle for abrupt pollution source tracing.

Interface aligned with species-search/oracle.py::SpeciesOracle:
  - .nrows / .ncols / .covs (K,nrows,ncols)
  - .in_bounds(i,j)
  - .query_cell(i,j) -> (presence: bool|None, cov_vector: ndarray|None)
  - .query_count / .reset_counter()

Black-box semantics: given a grid cell (i,j), return whether it is above the pollution threshold (0/1) plus its static covariate vector.
Every query is counted. The true source parameters are never exposed to the searcher (only the reporting layer reads them to evaluate tracing error).
"""
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data" / "oracle.npz"


class PollutionOracle:
    def __init__(self, npz_path: Path = DATA):
        d = np.load(npz_path, allow_pickle=True)
        self.nrows = int(d["nrows"])
        self.ncols = int(d["ncols"])
        self.presence = d["presence"].astype(bool)       # (nrows,ncols) bool
        self.covs = d["covs"].astype(np.float32)         # (K,nrows,ncols)
        self.cov_names = [str(x) for x in d["cov_names"]]
        self.query_count = 0
        # ground truth (for reporting only; never fed to the searcher)
        self.source_i = int(d["source_i"])
        self.source_j = int(d["source_j"])
        self.meta = str(d["meta"])
        self.threshold = float(d["threshold"])
        self.plume_frac = float(d["plume_frac"])

    @property
    def total_cells(self):
        return self.nrows * self.ncols

    def in_bounds(self, i, j):
        return 0 <= i < self.nrows and 0 <= j < self.ncols

    def query_cell(self, i, j):
        """Grid query: an out-of-bounds query still consumes one unit of budget (black-box semantics) and returns (None,None)."""
        self.query_count += 1
        if not self.in_bounds(i, j):
            return None, None
        return bool(self.presence[i, j]), self.covs[:, i, j].copy()

    def presence_cells(self):
        return [tuple(c) for c in np.argwhere(self.presence)]

    def reset_counter(self):
        self.query_count = 0
