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
"""build_oracle3d.py — 构建 3D 溶解氧实验场
目标：真实地中海西部溶解氧低值核（中层氧最低带，DO 低于全局阈值）
协变量 19 = 3 真实几何（depth/lat/lon 归一）+ 4 物理规律合成代理（透明标注）+ 12 噪声
输出：data/oracle3d.npz（presence[K,L,J] bool、X[19,N]、掩膜、元数据）
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

# ---- 溶解氧分布统计（定阈值）----
vals = do3d[np.isfinite(do3d)]
print("DO range %0.1f..%0.1f umol/l" % (vals.min(), vals.max()))
for q in [1, 2, 3, 5, 10, 25, 50, 75, 90]:
    print("  p%02d = %0.1f" % (q, np.percentile(vals, q)))
# 深度剖面（平均 DO per depth）——确认中层氧最低带
prof = np.nanmean(do3d, axis=(1, 2))
print("depth profile DO (top 12):")
for k in range(min(12, K)):
    print("  z=%6.0fm  DO=%0.1f" % (depth[k], prof[k]))

# ---- 目标：低氧核（中层氧最低带核心）----
# 地中海西部中深层氧最低带核心：DO <= 161 umol/l（约 p1.2 稀有目标，~350 格）
TARGET_TH = 161.0
presence = np.zeros((K, L, J), dtype=bool)
presence[np.isfinite(do3d) & (do3d <= TARGET_TH)] = True
n_tgt = int(presence.sum())
print("target DO<=%0.1f: n=%d (%.2f%% of valid)" % (TARGET_TH, n_tgt, 100 * n_tgt / int(np.isfinite(do3d).sum())))

# ---- 协变量 19 ----
# 真实几何 3（归一）
g = np.zeros((3, K, L, J))
g[0] = depth[:, None, None] / 1500.0
g[1] = (lat[None, :, None] - 30.0) / 15.0
g[2] = (lon[None, None, :] + 6.0) / 20.0

# 物理规律合成代理 4（透明标注：非实测，为海洋学经验规律）
zz = depth[:, None, None] / 1500.0
s = np.zeros((4, K, L, J))
s[0] = 17.0 - 5.0 * (1.0 - np.exp(-depth[:, None, None] / 200.0))     # 温度经验廓线 °C
s[1] = 14.6 - 0.4 * s[0]                                              # 溶解氧饱和浓度近似（Garcia-Gordon 线性近似）
s[2] = depth[:, None, None] / (depth[:, None, None] + 300.0)           # 伪密度分层
s[3] = np.exp(-depth[:, None, None] / 100.0)                           # 混合层指示
# 空间小扰动（让代理在水平向有微小结构，而非纯 depth 函数）
y = np.linspace(0, 1, L)[None, :, None]
x = np.linspace(0, 1, J)[None, None, :]
s[0] += 0.5 * np.sin(3 * x) * np.cos(2 * y)
s[1] -= 0.2 * np.cos(2 * x + y)
s[2] += 0.05 * x * y

# 噪声 12
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
