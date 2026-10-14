# -*- coding: utf-8 -*-
"""build_drift_oracle.py — 海上搜救逆向漂移 双阶段 oracle 构建（物理仿真）

【重要声明】本 oracle 为**物理仿真**，不使用真实海流/风场/漂移观测数据。
漂移轨迹由二维 leeway 平流方程 dP/dt = C(P) + L_i * W(t) 数值积分构造，
用于在完全黑盒、0/1 探测设定下测试 SpeciesSearcher 框架的"顺序解码双阶段"管线。

场景（顺序解码 A→B，不同时空，不融合地图）：
  阶段 A（当前时空，搜漂移残骸快照）：
    120x120 格，每格 2 km（域 240x240 km）。海难后 10 件残骸从失事点 P0
    随流漂移 T_A=48 h，各自 leeway L_i∈[0.01,0.03]，共享洋流 C 与风 W（缓变）。
    残骸位置在 T_A 时刻**冻结**（快照搜索，查询不推进时间）。探测半径 R_A=2 格
    (=4 km)；presence = 距任一残骸 ≤ R_A。
  阶段 B（过去时空，搜失事点 P0）：
    同一 120x120 网格，oracle 换成 P0 的静态证据（残骸聚集/油膜），
    presence = 距 P0 ≤ R_B=3 格(=6 km)，静态不随时间变。

协变量 19 个（静态逐格，喂 AUC 稀疏选参）：
  物理(7)：0 cur_u / 1 cur_v / 2 wind_u / 3 wind_v /
            4 dist_shore / 5 depth / 6 SAR_prior
  噪声(12)：7..18 平滑随机场
  阶段 B 运行时在末尾追加第 20 个协变量：到"解码质心"的距离（wrapper 生成）。

产物：
  drift-search/data/stageA.npz  阶段 A（残骸快照 presence + 19 cov + 真值）
  drift-search/data/stageB.npz  阶段 B（P0 静态 presence + 19 cov + 真值）
"""
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(parents=True, exist_ok=True)

# ---------------- 物理参数（全部显式记录，标注物理仿真） ----------------
N = 120                  # 网格 120x120
CELL_KM = 2.0            # 每格 2 km
DOM_KM = N * CELL_KM     # 240 km
R_A_CELL = 2             # 阶段 A 探测半径 2 格 (=4 km)
R_B_CELL = 3             # 阶段 B 探测半径 3 格 (=6 km)
T_A_H = 48.0             # 漂移时长 48 h
DT_H = 0.5               # 积分步长 0.5 h
N_DEBRIS = 10            # 残骸件数
P0_CELL = (40, 75)       # 失事点（格 row i, col j）
LEEWAY_LO, LEEWAY_HI = 0.01, 0.03   # 各残骸 leeway 系数范围
SIM_SEED = 20260930

# 失事点 km（格中心）
I0, J0 = P0_CELL
X0 = (J0 + 0.5) * CELL_KM
Y0 = (I0 + 0.5) * CELL_KM


def current_kmh(x, y):
    """洋流场（缓变空间场），返回 (cu, cv) km/h。"""
    cu = 0.15 + 0.020 * np.sin(y / 40.0)
    cv = 0.03 + 0.015 * np.sin(x / 50.0 + 1.0)
    return cu, cv


def wind_mps(t_h, w_rng):
    """风场（缓变时间分量 + 慢噪声），返回 (wu, wv) m/s。
    主风向偏北向（wv 主导），使 leeway 散布沿行(i)方向拉长，
    盲扫光栅扫到该列时可一次命中多行。"""
    wu = 3.0 + 0.8 * np.sin(2 * np.pi * t_h / 60.0) + 0.2 * w_rng.standard_normal()
    wv = 6.0 + 1.2 * np.sin(2 * np.pi * t_h / 80.0 + 0.5) + 0.3 * w_rng.standard_normal()
    return wu, wv


COV_NAMES_19 = [
    "洋流u(m/s)", "洋流v(m/s)", "风u(m/s)", "风v(m/s)", "离岸距离(km)",
    "水深(m)", "SAR先验",
    "浪高(m)", "浊度(NTU)", "渔业强度", "盐度异常", "波向(rad)",
    "阵风(m/s)", "叶绿素(mg/m3)", "海雾概率", "船舶密度", "水温异常",
    "浊流(m2/s)", "白噪声",
]
assert len(COV_NAMES_19) == 19


def smooth_noise(rng, shape, sigma):
    z = rng.standard_normal(shape)
    z = ndimage.gaussian_filter(z, sigma=sigma, mode="wrap")
    z -= z.min()
    z /= (z.max() + 1e-12)
    return z


def main():
    rng = np.random.default_rng(SIM_SEED)
    w_rng = np.random.default_rng(SIM_SEED + 7)

    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")  # ii=row, jj=col
    X = (jj + 0.5) * CELL_KM   # east km
    Y = (ii + 0.5) * CELL_KM   # north km

    # ---- 19 个静态协变量场 ----
    # 0,1 洋流 u/v（m/s）——与漂移积分用同一函数
    cu_grid, cv_grid = current_kmh(X, Y)
    covs = np.zeros((19, N, N), dtype=np.float32)
    covs[0] = (cu_grid / 3.6).astype(np.float32)
    covs[1] = (cv_grid / 3.6).astype(np.float32)
    # 2,3 风气候 u/v（m/s，空间大尺度平滑；与时间缓变风区分）
    covs[2] = (3.0 + 0.3 * np.sin(Y / 90.0)).astype(np.float32)
    covs[3] = (6.0 + 0.4 * np.sin(X / 100.0 + 0.5)).astype(np.float32)
    # 4 离岸距离（km）：到最近边界
    covs[4] = (np.minimum(np.minimum(X, DOM_KM - X),
                          np.minimum(Y, DOM_KM - Y))).astype(np.float32)
    # 5 水深（m）：离岸加深
    depth = 10.0 + covs[4] * 2.0 + rng.normal(0, 0.5, (N, N))
    covs[5] = depth.astype(np.float32)
    # 7..18 噪声池
    for k in range(7, 19):
        sig = rng.uniform(2.0, 7.0)
        covs[k] = smooth_noise(rng, (N, N), sig).astype(np.float32)

    # ---- 漂移积分：10 件残骸从 P0 出发，48h ----
    n_steps = int(T_A_H / DT_H)            # 96
    L_vals = np.linspace(LEEWAY_LO, LEEWAY_HI, N_DEBRIS)   # 0.01..0.03，均值 0.02
    # 风场是共享环境强迫：预计算一条风时间序列，所有残骸共用（仅 leeway 系数不同）
    wind_u_t = np.zeros(n_steps)
    wind_v_t = np.zeros(n_steps)
    for s in range(n_steps):
        t_h = s * DT_H
        wind_u_t[s], wind_v_t[s] = wind_mps(t_h, w_rng)
    wu_mean = float(wind_u_t.mean())
    wv_mean = float(wind_v_t.mean())

    debris_km = np.zeros((N_DEBRIS, 2))     # 末态 (x,y) km
    diff_rng = np.random.default_rng(SIM_SEED + 13)
    DIFF_KM2H = 0.5     # 湍流扩散系数（km^2/h），随机游走扩散
    for k in range(N_DEBRIS):
        pos = np.array([X0, Y0], dtype=float)
        for s in range(n_steps):
            cu, cv = current_kmh(pos[0], pos[1])
            wu, wv = wind_u_t[s], wind_v_t[s]
            # leeway 速度 km/h = L * W(m/s) * 3.6
            lu = L_vals[k] * wu * 3.6
            lv = L_vals[k] * wv * 3.6
            vx = cu + lu
            vy = cv + lv
            # 湍流扩散：随机游走 sqrt(2 D dt)
            sg = (2.0 * DIFF_KM2H * DT_H) ** 0.5
            pos = pos + np.array([vx, vy]) * DT_H + sg * diff_rng.standard_normal(2)
        debris_km[k] = pos

    # 代表性解码漂移（wrapper 反演用）：当前取 P0 处流 + 时间平均风 + 平均 leeway 0.02
    cu_p0, cv_p0 = current_kmh(X0, Y0)
    L_hat = 0.5 * (LEEWAY_LO + LEEWAY_HI)   # 0.02
    disp_dec = np.array([cu_p0 + L_hat * wu_mean * 3.6,
                         cv_p0 + L_hat * wv_mean * 3.6]) * T_A_H   # km
    cloud_center_km = np.array([X0, Y0]) + disp_dec          # 期望残骸云中心

    # SAR 先验：以"P0+漂移估计"为中心、σ 大（=30 km）
    SIGMA_SAR_KM = 30.0
    sar_prior = np.exp(-((X - cloud_center_km[0]) ** 2 +
                         (Y - cloud_center_km[1]) ** 2) /
                       (2.0 * SIGMA_SAR_KM ** 2))
    covs[6] = sar_prior.astype(np.float32)

    # ---- 阶段 A presence：距任一残骸 ≤ R_A_CELL 格 ----
    presenceA = np.zeros((N, N), dtype=bool)
    debris_cell = np.zeros((N_DEBRIS, 2), dtype=int)
    for k in range(N_DEBRIS):
        dc = int(round(debris_km[k][1] / CELL_KM - 0.5))   # row i ~ y
        dr = int(round(debris_km[k][0] / CELL_KM - 0.5))   # col j ~ x
        debris_cell[k] = (dc, dr)
        for di in range(-R_A_CELL, R_A_CELL + 1):
            for dj in range(-R_A_CELL, R_A_CELL + 1):
                if di * di + dj * dj > R_A_CELL * R_A_CELL:
                    continue
                ii_, jj_ = dc + di, dr + dj
                if 0 <= ii_ < N and 0 <= jj_ < N:
                    presenceA[ii_, jj_] = True

    # ---- 阶段 B presence：距 P0 ≤ R_B_CELL 格 ----
    presenceB = np.zeros((N, N), dtype=bool)
    for di in range(-R_B_CELL, R_B_CELL + 1):
        for dj in range(-R_B_CELL, R_B_CELL + 1):
            if di * di + dj * dj > R_B_CELL * R_B_CELL:
                continue
            ii_, jj_ = I0 + di, J0 + dj
            if 0 <= ii_ < N and 0 <= jj_ < N:
                presenceB[ii_, jj_] = True

    # ---- 自检 ----
    disp = np.hypot(debris_km[:, 0] - X0, debris_km[:, 1] - Y0)
    print(f"[build] P0=({X0:.1f},{Y0:.1f}) km  域=[0,{DOM_KM}]km")
    print(f"[build] 残骸末态距 P0 位移: min={disp.min():.1f} max={disp.max():.1f} "
          f"mean={disp.mean():.1f} km")
    print(f"[build] 期望残骸云中心=({cloud_center_km[0]:.1f},{cloud_center_km[1]:.1f}) km")
    print(f"[build] 解码代表位移=({disp_dec[0]:.2f},{disp_dec[1]:.2f}) km "
          f"(cu_p0={cu_p0:.3f} cv_p0={cv_p0:.3f} wu_mean={wu_mean:.2f} wv_mean={wv_mean:.2f})")
    print(f"[build] 阶段A presence 格数={int(presenceA.sum())} "
          f"({presenceA.mean()*100:.2f}%)")
    print(f"[build] 阶段B presence 格数={int(presenceB.sum())} "
          f"({presenceB.mean()*100:.2f}%)")

    metaA = ("PHYSICS_SIMULATION_NOT_REAL_DATA | leeway drift snapshot (frozen at T_A=48h); "
             "dP/dt=C(P)+L_i*W(t); 0/1 detection within R_A=2 cells=4km of any debris; "
             "static snapshot, query does NOT advance time.")
    metaB = ("PHYSICS_SIMULATION_NOT_REAL_DATA | static wreck-point evidence (debris "
             "aggregation/oil slick); presence within R_B=3 cells=6km of P0; static.")

    # ---- 保存 stageA.npz ----
    np.savez(
        DATA / "stageA.npz",
        covs=covs.astype(np.float32),
        presence=presenceA,
        cov_names=np.array(COV_NAMES_19),
        nrows=N, ncols=N, cell_km=CELL_KM,
        # 真值（仅报告/解码用，对黑盒搜索不可见）
        P0_i=I0, P0_j=J0, P0_x=X0, P0_y=Y0,
        debris_cell=debris_cell.astype(np.int32),     # (10,2) row,col
        debris_km=debris_km.astype(np.float64),       # (10,2) x,y
        R_A_cell=R_A_CELL, R_B_cell=R_B_CELL,
        T_A_h=T_A_H, dt_h=DT_H, n_debris=N_DEBRIS,
        leeway_lo=LEEWAY_LO, leeway_hi=LEEWAY_HI, leeway_hat=L_hat,
        # 解码 wrapper 代表量
        cu_p0=cu_p0, cv_p0=cv_p0, wu_mean=wu_mean, wv_mean=wv_mean,
        cloud_center_x=cloud_center_km[0], cloud_center_y=cloud_center_km[1],
        sim_seed=SIM_SEED,
        meta=metaA,
    )
    # ---- 保存 stageB.npz（同 19 cov，presence 换 P0 静态证据）----
    np.savez(
        DATA / "stageB.npz",
        covs=covs.astype(np.float32),
        presence=presenceB,
        cov_names=np.array(COV_NAMES_19),
        nrows=N, ncols=N, cell_km=CELL_KM,
        P0_i=I0, P0_j=J0, P0_x=X0, P0_y=Y0,
        R_B_cell=R_B_CELL,
        sim_seed=SIM_SEED,
        meta=metaB,
    )
    print(f"[build] -> {DATA/'stageA.npz'}  covs={covs.shape}")
    print(f"[build] -> {DATA/'stageB.npz'}")


if __name__ == "__main__":
    main()

