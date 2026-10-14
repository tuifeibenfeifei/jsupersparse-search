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
"""build_chlor_oracle.py — 构建叶绿素实验 Oracle
- 目标场：真实 ESA CCI 叶绿素（孟加拉湾 2024-06 月平均，241x241 @0.04°）
  presence = chlor > p99.5（高氯热点 270 个）
- 协变量（19 个，物理规律合成，透明标注）：
  结构参数 7 个：sst_neg(SST 与氯负相关/上升流)、sss(盐度弱负相关/河口低盐)、
    depth(北浅南深几何)、par(PAR 正相关/光合)、wind(风正相关/上升流)、
    mld(混合层负相关/稀释)、coast_dist(岸距几何,间接)
  噪声参数 12 个：noise1-12（U[0,1] 无关变量，模拟参数池噪声）
  注：协变量与目标的统计关系符合已确立海洋学事实；数值为规律合成并含噪声，
      算法黑盒不可见任何构造信息。
输出：data/oracle.npz（presence uint8[H,W]、covs float32[19,H,W]、cov_names）
"""
import csv
import numpy as np
from pathlib import Path

OUT = Path(REPO_ROOT / "chlorophyll-search/data")
rng = np.random.RandomState(42)

# 1) 读真实氯场 → 网格
rows = []
with open(OUT / "raw_chlorophyll.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows.append((float(r["latitude"]), float(r["longitude"]), float(r["chlor_a"])))
lats = np.array([r[0] for r in rows])
lons = np.array([r[1] for r in rows])
chl = np.array([r[2] for r in rows])
uniq_lat = np.sort(np.unique(np.round(lats, 3)))[::-1]   # 降序（北→南）
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
chl_grid = np.nan_to_num(chl_grid, nan=med)   # 云/陆掩膜 → 全局中值

# 2) presence：高氯热点
valid = chl_grid[chl_grid >= 0]
p995 = float(np.percentile(valid, 99.5))
presence = (chl_grid > p995).astype(np.uint8)
print("presence:", int(presence.sum()), "thr=%.3f" % p995, "grid", H, "x", W)

# 3) 归一化氯 [0,1]
lo, hi = float(valid.min()), float(valid.max())
chl_n = np.clip((chl_grid - lo) / (hi - lo), 0.0, 1.0)

# 4) 网格坐标（几何参数）
ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
lat_n = (ii / max(H - 1, 1)).astype(np.float64)      # 0=北 1=南
lon_n = (jj / max(W - 1, 1)).astype(np.float64)
# 海岸距离（到网格边缘的最小归一化距离，近似离岸远近；北/东侧近岸为孟加拉湾地形特征）
coast_d = np.minimum(np.minimum(lat_n, 1 - lat_n), np.minimum(lon_n, 1 - lon_n))
# depth：北浅南深（恒河陆架）+ 岸近浅
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

# oracle.py 兼容键：网格上下翻转使 i=0 对应最小纬度（南界）
presence = presence[::-1].copy()
covs = covs[:, ::-1].copy()
cell_deg = 0.04166666666666667
lat0 = 9.979166666666671      # 最小纬度（翻转后 i=0 处）
lon0 = 85.02083333333331      # 最小经度

np.savez(OUT / "oracle.npz", presence=presence, covs=covs, cov_names=np.array(cov_names),
         chlor_thr=p995, chlor_grid=chl_grid[::-1].astype(np.float32),
         lat0=lat0, lon0=lon0, cell_deg=cell_deg,
         nrows=H, ncols=W, study_poly=np.array([85.0, 95.0, 10.0, 20.0]))
print("ORACLE SAVED ->", OUT / "oracle.npz")
print("cov:", cov_names)
