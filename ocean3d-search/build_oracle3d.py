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
"""build_oracle3d.py — Build 3D dissolved oxygen experiment field
Target: real western Mediterranean dissolved oxygen low-value cores (mid-depth oxygen minimum zone, DO below global threshold)
Covariates 19 = 3 real geometry (depth/lat/lon normalized) + 4 physics-synthesized proxies (transparently annotated) + 12 noise
Output: data/oracle3d.npz (presence[K,L,J] bool, X[19,N], mask, metadata)
"""
import numpy as np
from pathlib import Path

ROOT = Path(REPO_ROOT / "ocean3d-search")
DATA = ROOT / "data"
z = np.load(DATA / "raw_do_3d.npz")
do3d = z["do3d"]           # (29, 33, 73) umol/l
depth = z["depth"]         # m
lat = z["lat"]             # deg N
lon = z["lon"]             # deg E
K, L, J = do3d.shape
print("do3d", do3d.shape)

# ---- Dissolved oxygen distribution statistics (threshold determination) ----
vals = do3d[np.isfinite(do3d)]
print("DO range %0.1f..%0.1f umol/l" % (vals.min(), vals.max()))
for q in [1, 2, 3, 5, 10, 25, 50, 75, 90]:
    print("  p%02d = %0.1f" % (q, np.percentile(vals, q)))
# Depth profile (mean DO per depth) — confirm mid-depth oxygen minimum zone
prof = np.nanmean(do3d, axis=(1, 2))
print("depth profile DO (top 12):")
for k in range(min(12, K)):
    print("  z=%6.0fm  DO=%0.1f" % (depth[k], prof[k]))

# ---- Target: hypoxic cores (core of mid-depth oxygen minimum zone) ----
# Western Mediterranean mid-deep oxygen minimum zone core: DO <= 161 umol/l (~p1.2 rare target, ~350 cells)
TARGET_TH = 161.0
presence = np.zeros((K, L, J), dtype=bool)
presence[np.isfinite(do3d) & (do3d <= TARGET_TH)] = True
n_tgt = int(presence.sum())
print("target DO<=%0.1f: n=%d (%.2f%% of valid)" % (TARGET_TH, n_tgt, 100 * n_tgt / int(np.isfinite(do3d).sum())))

# ---- Covariates 19 ----
# Real geometry 3 (normalized)
g = np.zeros((3, K, L, J))
g[0] = depth[:, None, None] / 1500.0
g[1] = (lat[None, :, None] - 30.0) / 15.0
g[2] = (lon[None, None, :] + 6.0) / 20.0

# Physics-synthesized proxies 4 (transparently annotated: not measured, based on oceanographic empirical laws)
zz = depth[:, None, None] / 1500.0
s = np.zeros((4, K, L, J))
s[0] = 17.0 - 5.0 * (1.0 - np.exp(-depth[:, None, None] / 200.0))     # Empirical temperature profile °C
s[1] = 14.6 - 0.4 * s[0]                                              # Dissolved oxygen saturation concentration approximation (Garcia-Gordon linear approximation)
s[2] = depth[:, None, None] / (depth[:, None, None] + 300.0)           # Pseudo-density stratification
s[3] = np.exp(-depth[:, None, None] / 100.0)                           # Mixed layer indicator
# Small spatial perturbation (give proxies minor horizontal structure rather than pure depth functions)
y = np.linspace(0, 1, L)[None, :, None]
x = np.linspace(0, 1, J)[None, None, :]
s[0] += 0.5 * np.sin(3 * x) * np.cos(2 * y)
s[1] -= 0.2 * np.cos(2 * x + y)
s[2] += 0.05 * x * y

# Noise 12
rng = np.random.default_rng(42)
nz = rng.random((12, K, L, J))

X = np.concatenate([g, s, nz], axis=0).reshape(19, -1)   # (19, N)
mask = np.isfinite(do3d).reshape(-1)
print("X:", X.shape, "mask:", int(mask.sum()))

np.savez(DATA / "oracle3d.npz",
         presence=presence, do3d=do3d.astype(np.float32),
         depth=depth, lat=lat, lon=lon,
         X=X.astype(np.float32), mask=mask,
         target_th=TARGET_TH)
print("SAVED ->", DATA / "oracle3d.npz")
