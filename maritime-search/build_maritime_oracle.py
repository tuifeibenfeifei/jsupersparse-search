# -*- coding: utf-8 -*-
"""build_maritime_oracle.py — 海上搜救/UAV 黑盒 oracle 构建（物理仿真）

场景：
  二维海面 150x150 格，每格 2 km，域 300x300 km。
  一名遇险人员/救生筏从失事点 P0 落水后随流漂移；UAV 每次出动探测一格。
  探测：目标在格中心半径 R=3 格(=6km) 内 -> pres=1，否则 0。
  时间：t_q = query_count * Δt，Δt=0.5h；预算 600 queries = 300h 搜索窗口。
  漂移：dP/dt = C(P) + L * W(t)，C=洋流场(静态空间)，W(t)=风(缓变正弦+噪声)，
        L=leeway 系数。

数据策略：无现成公开漂移轨迹可直接做成 oracle，采用 leeway 漂移物理仿真构造，
  本脚本与产物均标注"物理仿真"。协变量 19 个：9 个物理可用场 + 10 个噪声场。
  SAR 先验 = 以 P0 为中心、覆盖漂移不确定度 sigma 的高斯场（模拟"已知失事位置"）。

产物：maritime-search/data/oracle.npz
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(parents=True, exist_ok=True)

# ---------------- 物理参数（全部显式记录，标注物理仿真）----------------
N = 150                 # 网格 150x150
CELL_KM = 2.0           # 每格 2 km
R_CELL = 3              # 探测半径 3 格
R_KM = R_CELL * CELL_KM  # =6 km 搜索条带
DT_H = 0.5              # 每 query 时间步 0.5 h
MAX_Q = 600             # 预算 600 queries
P0_CELL = (75, 75)      # 失事点（格）
L_LEEWAY = 0.02         # leeway 系数（救生筏典型 1.5%-3% 风速）
SIM_SEED = 20240930     # 漂移场/轨迹确定性随机种子

rng = np.random.RandomState(SIM_SEED)

# ---------------- 坐标网格（km）----------------
xj = (np.arange(N) + 0.5) * CELL_KM          # 列方向 x
yi = (np.arange(N) + 0.5) * CELL_KM          # 行方向 y
X, Y = np.meshgrid(xj, yi)                   # shape (N,N)，X 沿列，Y 沿行
x0, y0 = (P0_CELL[1] + 0.5) * CELL_KM, (P0_CELL[0] + 0.5) * CELL_KM


# ---------------- 静态协变量场 ----------------
# 0,1 洋流 u/v（先以 km/h 建模，存 m/s）
cur_u_kmh = 0.10 + 0.06 * np.sin(Y / 55.0)
cur_v_kmh = 0.05 * np.sin(X / 45.0 + 1.0)
# 2,3 风气候 u/v（m/s，空间大尺度平滑）
wind_u = 3.5 + 1.2 * np.sin(Y / 90.0)
wind_v = 1.6 * np.sin(X / 110.0 + 0.7)
# 4 离岸距离（km）：以西北岸线（x=0 或 y=0）为最近岸
dist_shore = np.clip(np.minimum(X, Y), 0.0, None)
# 5 水深（m）：离岸加深
depth = 10.0 + dist_shore * 3.0 + rng.normal(0, 0.6, (N, N))
# 6 SST（℃） 7 SSS（psu）：平滑气候场
sst = 22.0 - 0.02 * (Y - 150.0) + 0.5 * np.sin(X / 60.0)
sss = 33.0 + 0.01 * (X - 150.0)
# 8 SAR 先验：以 P0 为中心高斯，sigma=50km（覆盖漂移不确定度）
SIGMA_KM = 50.0
sar_prior = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2.0 * SIGMA_KM ** 2))

# 9..18 噪声参数池（浪高/浊度/渔业/盐度异常/波向/阵风/叶绿素/海雾/船舶密度/白噪声）
wave_h = np.abs(rng.normal(1.5, 0.8, (N, N)))
turb = np.abs(rng.normal(2.0, 1.0, (N, N)))
fish = rng.uniform(0.0, 1.0, (N, N))
salt_anom = rng.normal(0.0, 0.3, (N, N))
wave_dir = rng.uniform(0.0, 2 * np.pi, (N, N))
gust = np.abs(rng.normal(6.0, 2.0, (N, N)))
chl = np.abs(rng.normal(0.3, 0.2, (N, N)))
fog = rng.uniform(0.0, 1.0, (N, N))
ship = rng.uniform(0.0, 1.0, (N, N))
white = rng.normal(0.0, 1.0, (N, N))

covs = np.stack([
    cur_u_kmh / 3.6, cur_v_kmh / 3.6,   # 0,1  洋流 u/v (m/s)
    wind_u, wind_v,                      # 2,3  风 u/v (m/s)
    dist_shore, depth, sst, sss,         # 4,5,6,7
    sar_prior,                           # 8    SAR 先验得分
    wave_h, turb, fish, salt_anom, wave_dir, gust, chl, fog, ship, white,
]).astype(np.float32)                     # 9..18 噪声池
assert covs.shape[0] == 19, covs.shape

cov_names = [
    "洋流u(m/s)", "洋流v(m/s)", "风u(m/s)", "风v(m/s)", "离岸距离(km)",
    "水深(m)", "SST(℃)", "SSS(psu)", "SAR先验",
    "浪高(m)", "浊度(NTU)", "渔业强度", "盐度异常", "波向(rad)",
    "阵风(m/s)", "叶绿素(mg/m3)", "海雾概率", "船舶密度", "白噪声",
]


# ---------------- 漂移轨迹仿真（确定性，固定 SIM_SEED 噪声）----------------
# 风的时间缓变分量（正弦+小幅噪声）
wind_time = np.random.RandomState(SIM_SEED + 7)
pos = np.array([x0, y0], dtype=float)
traj = np.zeros((MAX_Q + 1, 2), dtype=np.float64)
traj[0] = pos
for k in range(1, MAX_Q + 1):
    t_h = k * DT_H
    # 洋流：在目标位置采样静态场
    cu = 0.10 + 0.06 * np.sin(pos[1] / 55.0)
    cv = 0.05 * np.sin(pos[0] / 45.0 + 1.0)
    # 风：缓变周期分量 + 慢漂移噪声
    wu = 3.5 + 1.2 * np.sin(2 * np.pi * t_h / 72.0) \
        + 0.4 * wind_time.randn()
    wv = 1.6 * np.sin(2 * np.pi * t_h / 96.0 + 0.5) \
        + 0.4 * wind_time.randn()
    # leeway 速度 km/h = L * W(m/s) * 3.6
    lu = L_LEEWAY * wu * 3.6
    lv = L_LEEWAY * wv * 3.6
    vx = cu + lu
    vy = cv + lv
    pos = pos + np.array([vx, vy]) * DT_H
    traj[k] = pos

# ---------------- 自检：轨迹是否留在域内 / 漂移距离 ----------------
disp = np.hypot(traj[:, 0] - x0, traj[:, 1] - y0)
print(f"[build] P0=({x0:.1f},{y0:.1f}) km  域=[0,{N*CELL_KM}]km")
print(f"[build] 轨迹末点=({traj[-1,0]:.1f},{traj[-1,1]:.1f}) km")
print(f"[build] 距 P0 位移: min={disp.min():.1f} max={disp.max():.1f} "
      f"末={disp[-1]:.1f} km")
inbox = ((traj[:, 0] >= 0) & (traj[:, 0] <= N * CELL_KM)
         & (traj[:, 1] >= 0) & (traj[:, 1] <= N * CELL_KM)).mean()
print(f"[build] 轨迹在域内比例={inbox*100:.1f}%  "
      f"SAR先验σ={SIGMA_KM}km  R={R_KM}km")

# ---------------- 元数据（明确标注物理仿真）----------------
meta = {
    "data_source": "物理仿真 (leeway drift simulation), NOT real observational data",
    "nrows": N, "ncols": N, "cell_km": CELL_KM,
    "R_cell": R_CELL, "R_km": R_KM, "dt_h": DT_H, "max_queries": MAX_Q,
    "P0_cell": P0_CELL, "P0_km": [x0, y0],
    "leeway_coeff": L_LEEWAY, "sim_seed": SIM_SEED,
    "sar_prior_sigma_km": SIGMA_KM,
    "current_mean_kmh": 0.10, "wind_mean_mps": 3.5,
    "cov_names": cov_names,
    "note": "0/1 探测黑盒；presence 随 query_count(时间) 动态变化，由 traj 查表决定",
}
presence_meta = np.array([json.dumps(meta, ensure_ascii=False)], dtype=object)

out = DATA / "oracle.npz"
np.savez(
    out,
    nrows=N, ncols=N,
    cell_km=CELL_KM, R_km=R_KM, dt_h=DT_H, max_queries=MAX_Q,
    covs=covs,                                   # (19,150,150)
    cov_names=np.array(cov_names),
    traj=traj,                                   # (MAX_Q+1, 2) km 坐标 [x,y]
    P0_cell=np.array(P0_CELL),
    presence=sar_prior.astype(np.float32),       # 占位（真实 presence 动态计算）
    presence_meta=presence_meta,
    meta=presence_meta,
)
print(f"[build] saved -> {out}  covs={covs.shape} traj={traj.shape}")


# =====================================================================
# oracle 类（接口对齐 SpeciesOracle）
# =====================================================================
class MaritimeOracle:
    """海上搜救黑盒 oracle。query_cell(i,j) -> (pres, covs[:,i,j])。

    presence 是动态的：每次 query_count 推进 Δt，查 traj[k] 得目标当前位置，
    若格中心距目标 <= R_km 则 pres=1。covs 为静态逐格场。
    """

    def __init__(self, npz_path: Path = out):
        d = np.load(npz_path, allow_pickle=True)
        self.nrows = int(d["nrows"])
        self.ncols = int(d["ncols"])
        self.covs = d["covs"]                      # (K, nrows, ncols)
        self.cov_names = [str(x) for x in d["cov_names"]]
        self.traj = d["traj"]                      # (MAX_Q+1, 2) km
        self.cell_km = float(d["cell_km"])
        self.R_km = float(d["R_km"])
        self.P0_cell = tuple(int(v) for v in d["P0_cell"])
        self.query_count = 0

    def in_bounds(self, i, j):
        return 0 <= i < self.nrows and 0 <= j < self.ncols

    def reset_counter(self):
        self.query_count = 0

    @property
    def total_cells(self):
        return self.nrows * self.ncols

    def query_cell(self, i, j):
        self.query_count += 1
        if not self.in_bounds(i, j):
            return None, None
        k = min(self.query_count, self.traj.shape[0] - 1)
        tx, ty = self.traj[k]
        cx = (j + 0.5) * self.cell_km
        cy = (i + 0.5) * self.cell_km
        pres = float(np.hypot(cx - tx, cy - ty)) <= self.R_km
        return bool(pres), self.covs[:, i, j].copy()


if __name__ == "__main__":
    # 自检：实例化并模拟几次查询
    o = MaritimeOracle(out)
    print(f"[self-test] oracle {o.nrows}x{o.ncols} covs={o.covs.shape} R={o.R_km}km")
    hits = 0
    for k in range(1, MAX_Q + 1):
        i = int(o.traj[k][1] / o.cell_km - 0.5)
        j = int(o.traj[k][0] / o.cell_km - 0.5)
        if not o.in_bounds(i, j):
            continue
        p, _ = o.query_cell(i, j)
        hits += int(p is True)
    print(f"[self-test] 沿轨迹逐格探测命中率={hits}/{MAX_Q} "
          f"(应接近 {MAX_Q}，因格中心距目标<=R)")
    o.reset_counter()
    print("[self-test] reset_counter ok, query_count=", o.query_count)
