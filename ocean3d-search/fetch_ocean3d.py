# -*- coding: utf-8 -*-
"""fetch_ocean3d.py v4 - EMODnet real 4D dissolved oxygen (western Mediterranean subset), no bracketed index syntax.
Source: all_european_seas_DO_4D_6b2d_86a0_45c7 (EMODnet Chemistry DIVAnd 4D, 102 depth levels x 0.125 deg).
Variable: Water_body_dissolved_oxygen_concentration [time=12][depth=102][lat][lon] umol/l.
Subset: 36-44N x -6-12E, time[0], depth 0-1500 m (step 2).
Output: ocean3d-search/data/axes.npz and raw_do_3d.npz

The downloads run only when the file is executed as a script, so importing it (for
example from a repository-wide health check) never touches the network.
"""
import csv
import sys
import urllib.error
import urllib.request
from io import StringIO
from pathlib import Path

import numpy as np

# --- repository root: walk up until the directory that holds configs/ ---
REPO_ROOT = Path(__file__).resolve()
while not (REPO_ROOT / "configs").is_dir():
    if REPO_ROOT.parent == REPO_ROOT:
        raise RuntimeError("repository root (containing configs/) not found")
    REPO_ROOT = REPO_ROOT.parent
# -----------------------------------------------------------------------

DS = "all_european_seas_DO_4D_6b2d_86a0_45c7"
VAR = "Water_body_dissolved_oxygen_concentration"
UA = {"User-Agent": "Mozilla/5.0"}


def fetch(expr, timeout=300):
    u = f"https://erddap.emodnet.eu/erddap/griddap/{DS}.csv?{expr}"
    req = urllib.request.Request(u, headers=UA)
    return urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "ignore")


def parse_axis(txt, col):
    rd = csv.reader(StringIO(txt))
    next(rd, None)
    vals = []
    for r in rd:
        if len(r) < 5:
            continue
        try:
            vals.append(float(r[col]))
        except Exception:
            continue
    return np.array(vals)


def main() -> int:
    """Download the depth/lat/lon axes and the oxygen cube, then save both npz files."""
    out = REPO_ROOT / "ocean3d-search" / "data"
    out.mkdir(parents=True, exist_ok=True)

    try:
        print("lat profile...", flush=True)
        lat = parse_axis(fetch(f"{VAR}[0][0][0:1:236][0]"), 2)
        print("  lat n=%d %0.3f..%0.3f" % (len(lat), lat.min(), lat.max()), flush=True)
        print("lon profile...", flush=True)
        lon = parse_axis(fetch(f"{VAR}[0][0][0][0:1:460]"), 3)
        print("  lon n=%d %0.3f..%0.3f" % (len(lon), lon.min(), lon.max()), flush=True)
        print("depth profile...", flush=True)
        depth = parse_axis(fetch(f"{VAR}[0][0:1:101][0][0]"), 1)
        print("  depth n=%d %0.2f..%0.2f" % (len(depth), depth.min(), depth.max()), flush=True)
    except (urllib.error.URLError, OSError) as exc:
        print(f"[error] axis download failed: {exc}")
        print("        the EMODnet server needs to be reachable; retry later")
        return 1

    np.savez(out / "axes.npz", depth=depth, lat=lat, lon=lon)

    i0, i1 = int(np.searchsorted(lat, 36.0)), int(np.searchsorted(lat, 44.0, "right")) - 1
    j0, j1 = int(np.searchsorted(lon, -6.0)), int(np.searchsorted(lon, 12.0, "right")) - 1
    k0, k1 = int(np.searchsorted(depth, 0.0)), int(np.searchsorted(depth, 1500.0, "right")) - 1
    kd = np.arange(k0, k1 + 1, 2)
    print("subset lat[%d:%d] lon[%d:%d] depth[%d:%d] step2 -> %d layers"
          % (i0, i1, j0, j1, k0, k1, len(kd)), flush=True)

    expr = f"{VAR}[0][{kd[0]}:2:{kd[-1]}][{i0}:1:{i1}][{j0}:1:{j1}]"
    print("downloading", expr, flush=True)
    try:
        txt = fetch(expr, timeout=900)
    except (urllib.error.URLError, OSError) as exc:
        print(f"[error] cube download failed: {exc}")
        print("        the EMODnet server needs to be reachable; retry later")
        return 1
    print("bytes:", len(txt), flush=True)

    rd = csv.reader(StringIO(txt))
    next(rd, None)
    rows = []
    for r in rd:
        if len(r) < 5:
            continue
        try:
            rows.append((float(r[1]), float(r[2]), float(r[3]), float(r[4])))
        except Exception:
            continue
    rows = np.array(rows)
    print("rows:", rows.shape, flush=True)
    depths = np.unique(rows[:, 0]); lats = np.unique(rows[:, 1]); lons = np.unique(rows[:, 2])
    print("grid depth", len(depths), "lat", len(lats), "lon", len(lons), flush=True)
    do3d = np.full((len(depths), len(lats), len(lons)), np.nan)
    di = {v: k for k, v in enumerate(depths)}
    li = {v: k for k, v in enumerate(lats)}
    oi = {v: k for k, v in enumerate(lons)}
    for d, a, o, v in rows:
        if v > 1e30:
            continue
        do3d[di[d], li[a], oi[o]] = v
    print("do3d:", do3d.shape, "non-nan:", int(np.isfinite(do3d).sum()), flush=True)
    np.savez(out / "raw_do_3d.npz", do3d=do3d.astype(np.float32),
             depth=depths, lat=lats, lon=lons)
    print("SAVED ->", out / "raw_do_3d.npz", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
