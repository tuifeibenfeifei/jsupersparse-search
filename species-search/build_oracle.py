# -*- coding: utf-8 -*-
"""build_oracle.py — Build rare species search black-box oracle (general version, supports multiple species)
Usage:
  python build_oracle.py <species_subdir> <lon_min> <lon_max> <lat_min> <lat_max> <cell_deg> [max_unc]
Example:
  python build_oracle.py black_muntjac 117.5 119.5 27.5 30.5 0.025 5000
  python build_oracle.py clouded_leopard 84 99 25 30 0.05 5000

Input:
  - data/<species_subdir>/occurrences.csv   (GBIF records; columns: lat, lon, uncertainty_m)
  - data/worldclim/wc2.1_5m_bio/            (WorldClim v2.1 bioclim 5m GeoTIFF, unzipped)
Output:
  - data/<species_subdir>/oracle.npz
Black-box semantics: query (lat, lon) → oracle.query() returns (presence, cov_vector)
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# ---- Covariate list (all 19 bioclimatic variables BIO1-19) ----
COV_FILES = {
    "bio1": "wc2.1_5m_bio_1.tif",
    "bio2": "wc2.1_5m_bio_2.tif",
    "bio3": "wc2.1_5m_bio_3.tif",
    "bio4": "wc2.1_5m_bio_4.tif",
    "bio5": "wc2.1_5m_bio_5.tif",
    "bio6": "wc2.1_5m_bio_6.tif",
    "bio7": "wc2.1_5m_bio_7.tif",
    "bio8": "wc2.1_5m_bio_8.tif",
    "bio9": "wc2.1_5m_bio_9.tif",
    "bio10": "wc2.1_5m_bio_10.tif",
    "bio11": "wc2.1_5m_bio_11.tif",
    "bio12": "wc2.1_5m_bio_12.tif",
    "bio13": "wc2.1_5m_bio_13.tif",
    "bio14": "wc2.1_5m_bio_14.tif",
    "bio15": "wc2.1_5m_bio_15.tif",
    "bio16": "wc2.1_5m_bio_16.tif",
    "bio17": "wc2.1_5m_bio_17.tif",
    "bio18": "wc2.1_5m_bio_18.tif",
    "bio19": "wc2.1_5m_bio_19.tif",
}


def load_occurrences(subdir: str, max_unc: float):
    """Read data/<subdir>/occurrences.csv; optionally filter coordinate uncertainty <= max_unc"""
    rows = []
    path = DATA / subdir / "occurrences.csv"
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lat, lon = float(r["lat"]), float(r["lon"])
            if max_unc and "uncertainty_m" in r:
                u = float(r["uncertainty_m"])
                if u > max_unc:
                    continue
            rows.append((lat, lon))
    return rows


def build():
    import rasterio

    if len(sys.argv) < 7:
        print(__doc__)
        return 1
    subdir = sys.argv[1]
    lon_min, lon_max = float(sys.argv[2]), float(sys.argv[3])
    lat_min, lat_max = float(sys.argv[4]), float(sys.argv[5])
    cell_deg = float(sys.argv[6])
    max_unc = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0

    occ = load_occurrences(subdir, max_unc)
    print(f"[info] {subdir}: occurrence count = {len(occ)} (unc<= {max_unc})")

    ncols = int(round((lon_max - lon_min) / cell_deg))
    nrows = int(round((lat_max - lat_min) / cell_deg))
    print(f"[info] grid {ncols}×{nrows} = {ncols * nrows} cells")

    # presence
    presence = np.zeros((nrows, ncols), dtype=bool)
    for lat, lon in occ:
        if not (lon_min <= lon < lon_max and lat_min <= lat < lat_max):
            continue
        j = min(int((lon - lon_min) / cell_deg), ncols - 1)
        i = min(int((lat - lat_min) / cell_deg), nrows - 1)
        presence[i, j] = True
    n_pres = int(presence.sum())
    print(f"[info] presence grid cells = {n_pres} ({n_pres / (ncols * nrows) * 100:.2f}%)")

    # covariates (nearest-neighbor upsampling)
    cov_names = list(COV_FILES.keys())
    covs = np.full((len(cov_names), nrows, ncols), np.nan, dtype=np.float32)
    wc_dir = DATA / "worldclim" / "wc2.1_5m_bio"
    lons_c = lon_min + (np.arange(ncols) + 0.5) * cell_deg
    lats_c = lat_min + (np.arange(nrows) + 0.5) * cell_deg
    for k, name in enumerate(cov_names):
        fn = wc_dir / COV_FILES[name]
        if not fn.exists():
            print(f"[warn] missing {fn.name}, covariate {name} all NaN")
            continue
        with rasterio.open(fn) as ds:
            row_top, col_left = ds.index(lon_min, lat_max)      # top-left (north, west)
            row_bottom, col_right = ds.index(lon_max, lat_min)  # bottom-right (south, east)
            arr = ds.read(1, window=((row_top, row_bottom + 1), (col_left, col_right + 1)))
        h_src, w_src = arr.shape
        src_i = np.clip(
            np.round((lat_max - lats_c) / (lat_max - lat_min) * (h_src - 1)).astype(int), 0, h_src - 1)
        src_j = np.clip(
            np.round((lons_c - lon_min) / (lon_max - lon_min) * (w_src - 1)).astype(int), 0, w_src - 1)
        covs[k] = arr[np.ix_(src_i, src_j)]
        valid = np.isfinite(covs[k])
        print(f"[info] {name}: source pixels {h_src}×{w_src} → target grid {nrows}×{ncols}, "
              f"valid {int(valid.sum())}/{nrows * ncols} "
              f"range[{np.nanmin(covs[k]):.1f},{np.nanmax(covs[k]):.1f}]")

    # Missing value fill (median)
    for k in range(len(cov_names)):
        m = np.nanmedian(covs[k])
        covs[k][~np.isfinite(covs[k])] = m

    out_dir = DATA / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "oracle.npz"
    np.savez_compressed(
        out,
        lon0=lon_min, lat0=lat_min, cell_deg=cell_deg,
        ncols=ncols, nrows=nrows,
        presence=presence, covs=covs,
        cov_names=np.array(cov_names, dtype="U16"),
        study_poly=np.array([lon_min, lon_max, lat_min, lat_max]),
    )
    print(f"[done] oracle saved → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(build())
