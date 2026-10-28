# -*- coding: utf-8 -*-
"""drift_oracle.py — maritime search-and-rescue reverse-drift black-box oracle (interface aligned with SpeciesOracle).

  - .nrows / .ncols / .covs (K,nrows,ncols) / .cov_names
  - .in_bounds(i,j)
  - .query_cell(i,j) -> (presence: bool|None, cov_vector: ndarray|None)
  - .query_count / .reset_counter() / .total_cells / .presence_cells

Stage A: presence = frozen debris snapshot (within R_A cells of any debris).
Stage B: presence = static evidence of the wreck point P0 (within R_B cells of P0).
At Stage B runtime, the 20th covariate (distance to the decoded centroid) is appended via append_covariate().
The ground truth (P0 / debris positions / physical parameters) is read only by the reporting and decoding wrapper, and is not fed to the searcher.
"""
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"


class DriftOracle:
    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=True)
        self.nrows = int(d["nrows"])
        self.ncols = int(d["ncols"])
        self.presence = d["presence"].astype(bool)       # (nrows,ncols)
        self.covs = d["covs"].astype(np.float32)         # (K,nrows,ncols)
        self.cov_names = [str(x) for x in d["cov_names"]]
        self.cell_km = float(d["cell_km"])
        self.query_count = 0
        # ground truth (for reporting/decoding only)
        self.P0_i = int(d["P0_i"]); self.P0_j = int(d["P0_j"])
        self.P0_x = float(d["P0_x"]); self.P0_y = float(d["P0_y"])
        self.meta = str(d["meta"])
        # stageA-specific (absent in stageB)
        for k in ("debris_cell", "debris_km", "R_A_cell", "R_B_cell",
                  "T_A_h", "cu_p0", "cv_p0", "wu_mean", "wv_mean",
                  "leeway_hat", "cloud_center_x", "cloud_center_y"):
            if k in d.files:
                v = d[k]
                setattr(self, k, float(v) if v.ndim == 0 else v)

    def append_covariate(self, band: np.ndarray, name: str):
        """append a covariate band at runtime (Stage B's 'distance to decoded centroid')."""
        band = band.astype(np.float32).reshape(1, self.nrows, self.ncols)
        self.covs = np.concatenate([self.covs, band], axis=0)
        self.cov_names.append(name)

    @property
    def total_cells(self):
        return self.nrows * self.ncols

    def in_bounds(self, i, j):
        return 0 <= i < self.nrows and 0 <= j < self.ncols

    def query_cell(self, i, j):
        """grid query: out-of-bounds also consumes one budget unit (black-box semantics); returns (None,None)."""
        self.query_count += 1
        if not self.in_bounds(i, j):
            return None, None
        return bool(self.presence[i, j]), self.covs[:, i, j].copy()

    def presence_cells(self):
        return [tuple(c) for c in np.argwhere(self.presence)]

    def reset_counter(self):
        self.query_count = 0
