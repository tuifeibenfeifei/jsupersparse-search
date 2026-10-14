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
"""fetch_ocean3d.py v4 — EMODNET 真实 4D 溶解氧（西地中海子集），无括号索引语法
源：all_european_seas_DO_4D_6b2d_86a0_45c7（EMODnet Chemistry DIVAnd 4D，102 深度层 × 0.125°）
变量：Water_body_dissolved_oxygen_concentration [time=12][depth=102][lat][lon] umol/l
子集：36-44N × -6-12E，time[0]，depth 0-1500m（步长 2）
"""
import csv
import urllib.request
from io import StringIO
from pathlib import Path

import numpy as np

OUT = Path(REPO_ROOT / "ocean3d-search/data")
OUT.mkdir(parents=True, exist_ok=True)
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


print("lat profile...", flush=True)
lat = parse_axis(fetch(f"{VAR}[0][0][0:1:236][0]"), 2)
print("  lat n=%d %0.3f..%0.3f" % (len(lat), lat.min(), lat.max()), flush=True)
print("lon profile...", flush=True)
lon = parse_axis(fetch(f"{VAR}[0][0][0][0:1:460]"), 3)
print("  lon n=%d %0.3f..%0.3f" % (len(lon), lon.min(), lon.max()), flush=True)
print("depth profile...", flush=True)
depth = parse_axis(fetch(f"{VAR}[0][0:1:101][0][0]"), 1)
print("  depth n=%d %0.2f..%0.2f" % (len(depth), depth.min(), depth.max()), flush=True)

np.savez(OUT / "axes.npz", depth=depth, lat=lat, lon=lon)

i0, i1 = int(np.searchsorted(lat, 36.0)), int(np.searchsorted(lat, 44.0, "right")) - 1
j0, j1 = int(np.searchsorted(lon, -6.0)), int(np.searchsorted(lon, 12.0, "right")) - 1
k0, k1 = int(np.searchsorted(depth, 0.0)), int(np.searchsorted(depth, 1500.0, "right")) - 1
kd = np.arange(k0, k1 + 1, 2)
print("subset lat[%d:%d] lon[%d:%d] depth[%d:%d] step2 -> %d layers" % (i0, i1, j0, j1, k0, k1, len(kd)), flush=True)

expr = f"{VAR}[0][{kd[0]}:2:{kd[-1]}][{i0}:1:{i1}][{j0}:1:{j1}]"
print("downloading", expr, flush=True)
txt = fetch(expr, timeout=900)
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
np.savez(OUT / "raw_do_3d.npz", do3d=do3d.astype(np.float32),
         depth=depths, lat=lats, lon=lons)
print("SAVED ->", OUT / "raw_do_3d.npz", flush=True)
