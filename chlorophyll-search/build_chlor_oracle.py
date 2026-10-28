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
"""build_chlor_oracle.py — Build chlorophyll experiment Oracle
- Target field: real ESA CCI chlorophyll (Bay of Bengal 2024-06 monthly mean, 241x241 @0.04°)
  presence = chlor > p99.5 (270 high-chlorophyll hotspots)
- Covariates (19, synthesized from physical laws, transparently annotated):
  7 structural parameters: sst_neg (SST negatively correlated with chlorophyll/upwelling), sss (weakly negative correlation with chlorophyll/estuarine low salinity),
    depth (shallow north/deep south geometry), par (PAR positive correlation/photosynthesis), wind (wind positive correlation/upwelling),
    mld (mixed layer negative correlation/dilution), coast_dist (coastal distance geometry, indirect)
  12 noise parameters: noise1-12 (U[0,1] irrelevant variables, simulating parameter pool noise)
  Note: Statistical relationships between covariates and target match established oceanographic facts; values are synthesized from laws and include noise,
      the algorithm's black box cannot see any construction information.
Output: data/oracle.npz (presence uint8[H,W], covs float32[19,H,W], cov_names)
"""
import csv
import numpy as np
from pathlib import Path

OUT = Path(REPO_ROOT / "chlorophyll-search/data")
rng = np.random.RandomState(42)

# 1) Read real chlorophyll field → grid
rows = []
with open(OUT / "raw_chlorophyll.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows.append((float(r["latitude"]), float(r["longitude"]), float(r["chlor_a"])))
lats = np.array([r[0] for r in rows])
lons = np.array([r[1] for r in rows])
chl = np.array([r[2] for r in rows])
uniq_lat = np.sort(np.unique(np.round(lats, 3)))[::-1]   # descending order (north→south)
uniq_lon = np.sort(np.unique(np.round(lons, 3)))
H, W = len(uniq_lat), len(uniq_lon)
lat2i = {v: i for i, v in enumerate(uniq_lat)}
lon2j = {v: j for j, v in enumerate(uniq_lon)}
chl_grid = np.full((H, W), np.nan, dtype=np.float64)
for (la, lo, c) in rows:
    i, j = lat2i[round(la, 3)], lon2j[round(lo, 3)]
    chl_grid[i, j] = c
print("grid", chl_grid.shape, "nan", int(np.isnan(chl_grid).sum()))
med = float(np.nanmedian(chl_grid))
chl_grid = np.nan_to_num(chl_grid, nan=med)   # cloud/land mask → global median

# 2) presence: high-chlorophyll hotspots
valid = chl_grid[chl_grid >= 0]
p995 = float(np.percentile(valid, 99.5))
presence = (chl_grid > p995).astype(np.uint8)
print("presence:", int(presence.sum()), "thr=%.3f" % p995, "grid", H, "x", W)

# 3) Normalize chlorophyll [0,1]
lo, hi = float(valid.min()), float(valid.max())
chl_n = np.clip((chl_grid - lo) / (hi - lo), 0.0, 1.0)

# 4) Grid coordinates (geometric parameters)
ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
lat_n = (ii / max(H - 1, 1)).astype(np.float64)      # 0=north 1=south
lon_n = (jj / max(W - 1, 1)).astype(np.float64)
# Coastal distance (minimum normalized distance to grid edge, approximating offshore distance; north/east nearshore is a Bay of Bengal topographic feature)
coast_d = np.minimum(np.minimum(lat_n, 1 - lat_n), np.minimum(lon_n, 1 - lon_n))
# depth: shallow north/deep south (Ganges shelf) + shallow near coast
depth = (0.70 * (1.0 - lat_n) + 0.25 * (1.0 - coast_d) + 0.05 * rng.rand(H, W))
depth = np.clip(depth, 0, 1)

n1 = rng.rand(H, W); n2 = rng.rand(H, W); n3 = rng.rand(H, W)
n4 = rng.rand(H, W); n5 = rng.rand(H, W); n6 = rng.rand(H, W)

covs = np.zeros((19, H, W), dtype=np.float32)
covs[0] = np.clip(1.0 - 0.55 * chl_n + 0.15 * n1, 0, 1)      # sst_neg
covs[1] = np.clip(1.0 - 0.35 * chl_n + 0.20 * n2, 0, 1)      # sss
covs[2] = depth.astype(np.float32)                            # depth
covs[3] = np.clip(0.50 * chl_n + 0.30 * n3, 0, 1)            # par
covs[4] = np.clip(0.45 * chl_n + 0.40 * n4, 0, 1)            # wind
covs[5] = np.clip(-0.30 * chl_n + 0.50 * n5 + 0.2, 0, 1)     # mld
covs[6] = coast_d.astype(np.float32)                          # coast_dist
for k in range(7, 19):
    covs[k] = rng.rand(H, W)                                   # noise1..12

cov_names = ["sst_neg", "sss", "depth", "par", "wind", "mld", "coast_dist"] + \
            ["noise%d" % (k - 6) for k in range(7, 19)]

# oracle.py compatibility key: flip grid vertically so i=0 corresponds to minimum latitude (south boundary)
presence = presence[::-1].copy()
covs = covs[:, ::-1].copy()
cell_deg = 0.04166666666666667
lat0 = 9.979166666666671      # minimum latitude (after flip at i=0)
lon0 = 85.02083333333331      # minimum longitude

np.savez(OUT / "oracle.npz", presence=presence, covs=covs, cov_names=np.array(cov_names),
         chlor_thr=p995, chlor_grid=chl_grid[::-1].astype(np.float32),
         lat0=lat0, lon0=lon0, cell_deg=cell_deg,
         nrows=H, ncols=W, study_poly=np.array([85.0, 95.0, 10.0, 20.0]))
print("ORACLE SAVED ->", OUT / "oracle.npz")
print("cov:", cov_names)
