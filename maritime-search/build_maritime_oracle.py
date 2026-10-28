# -*- coding: utf-8 -*-
"""build_maritime_oracle.py - maritime search-and-rescue/UAV black-box oracle construction (physical simulation)

Scenario:
  2D sea surface, 150x150 cells, 2 km per cell, domain 300x300 km.
  A distressed person / life raft drifts with the current after entering the water at the incident point P0; each UAV sortie probes one cell.
  Detection: if the target is within radius R=3 cells (=6 km) of the cell center -> pres=1, else 0.
  Time: t_q = query_count * Δt, Δt=0.5h; budget 600 queries = 300h search window.
  Drift: dP/dt = C(P) + L * W(t), C = ocean-current field (static spatial), W(t) = wind (slowly varying sinusoid + noise),
        L = leeway coefficient.

Data strategy: no ready-made public drift trajectory can be turned directly into an oracle, so a leeway-drift physical simulation is used;
  this script and its outputs are both marked "physical simulation". 19 covariates: 9 usable physical fields + 10 noise fields.
  SAR prior = a Gaussian field centered on P0 with sigma covering the drift uncertainty (simulating a "known incident location").

Output: maritime-search/data/oracle.npz
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(parents=True, exist_ok=True)

# ---------------- physical parameters (all explicitly recorded; marked as physical simulation) ----------------
N = 150                 # grid 150x150
CELL_KM = 2.0           # 2 km per cell
R_CELL = 3              # detection radius 3 cells
R_KM = R_CELL * CELL_KM  # =6 km search strip
DT_H = 0.5              # time step per query 0.5 h
MAX_Q = 600             # budget 600 queries
P0_CELL = (75, 75)      # incident point (cell)
L_LEEWAY = 0.02         # leeway coefficient (typical life raft 1.5%-3% of wind speed)
SIM_SEED = 20240930     # deterministic random seed for drift field / trajectory

rng = np.random.RandomState(SIM_SEED)

# ---------------- coordinate grid (km) ----------------
xj = (np.arange(N) + 0.5) * CELL_KM          # column direction x
yi = (np.arange(N) + 0.5) * CELL_KM          # row direction y
X, Y = np.meshgrid(xj, yi)                   # shape (N,N), X along columns, Y along rows
x0, y0 = (P0_CELL[1] + 0.5) * CELL_KM, (P0_CELL[0] + 0.5) * CELL_KM


# ---------------- static covariate fields ----------------
# 0,1 ocean current u/v (first modeled in km/h, stored in m/s)
cur_u_kmh = 0.10 + 0.06 * np.sin(Y / 55.0)
cur_v_kmh = 0.05 * np.sin(X / 45.0 + 1.0)
# 2,3 wind climate u/v (m/s, large-scale spatial smoothing)
wind_u = 3.5 + 1.2 * np.sin(Y / 90.0)
wind_v = 1.6 * np.sin(X / 110.0 + 0.7)
# 4 distance to shore (km): the northwest shoreline (x=0 or y=0) is the nearest shore
dist_shore = np.clip(np.minimum(X, Y), 0.0, None)
# 5 water depth (m): deepens offshore
depth = 10.0 + dist_shore * 3.0 + rng.normal(0, 0.6, (N, N))
# 6 SST (C) 7 SSS (psu): smoothed climate fields
sst = 22.0 - 0.02 * (Y - 150.0) + 0.5 * np.sin(X / 60.0)
sss = 33.0 + 0.01 * (X - 150.0)
# 8 SAR prior: Gaussian centered on P0, sigma=50 km (covering drift uncertainty)
SIGMA_KM = 50.0
sar_prior = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2.0 * SIGMA_KM ** 2))

# 9..18 noise parameter pool (wave height / turbidity / fishing / salinity anomaly / wave direction / gust / chlorophyll / sea fog / ship density / white noise)
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
    cur_u_kmh / 3.6, cur_v_kmh / 3.6,   # 0,1  ocean current u/v (m/s)
    wind_u, wind_v,                      # 2,3  wind u/v (m/s)
    dist_shore, depth, sst, sss,         # 4,5,6,7
    sar_prior,                           # 8    SAR prior score
    wave_h, turb, fish, salt_anom, wave_dir, gust, chl, fog, ship, white,
]).astype(np.float32)                     # 9..18 noise pool
assert covs.shape[0] == 19, covs.shape

cov_names = [
    "current u (m/s)", "current v (m/s)", "wind u (m/s)", "wind v (m/s)", "distance to shore (km)",
    "water depth (m)", "SST (C)", "SSS (psu)", "SAR prior",
    "wave height (m)", "turbidity (NTU)", "fishing intensity", "salinity anomaly", "wave direction (rad)",
    "gust (m/s)", "chlorophyll (mg/m3)", "sea-fog probability", "ship density", "white noise",
]


# ---------------- drift trajectory simulation (deterministic, fixed SIM_SEED noise) ----------------
# slowly varying temporal wind component (sinusoid + small noise)
wind_time = np.random.RandomState(SIM_SEED + 7)
pos = np.array([x0, y0], dtype=float)
traj = np.zeros((MAX_Q + 1, 2), dtype=np.float64)
traj[0] = pos
for k in range(1, MAX_Q + 1):
    t_h = k * DT_H
    # current: sample the static field at the target position
    cu = 0.10 + 0.06 * np.sin(pos[1] / 55.0)
    cv = 0.05 * np.sin(pos[0] / 45.0 + 1.0)
    # wind: slowly varying periodic component + slow drift noise
    wu = 3.5 + 1.2 * np.sin(2 * np.pi * t_h / 72.0) \
        + 0.4 * wind_time.randn()
    wv = 1.6 * np.sin(2 * np.pi * t_h / 96.0 + 0.5) \
        + 0.4 * wind_time.randn()
    # leeway speed km/h = L * W(m/s) * 3.6
    lu = L_LEEWAY * wu * 3.6
    lv = L_LEEWAY * wv * 3.6
    vx = cu + lu
    vy = cv + lv
    pos = pos + np.array([vx, vy]) * DT_H
    traj[k] = pos

# ---------------- self-check: whether the trajectory stays in-domain / drift distance ----------------
disp = np.hypot(traj[:, 0] - x0, traj[:, 1] - y0)
print(f"[build] P0=({x0:.1f},{y0:.1f}) km  domain=[0,{N*CELL_KM}]km")
print(f"[build] trajectory end point=({traj[-1,0]:.1f},{traj[-1,1]:.1f}) km")
print(f"[build] displacement from P0: min={disp.min():.1f} max={disp.max():.1f} "
      f"end={disp[-1]:.1f} km")
inbox = ((traj[:, 0] >= 0) & (traj[:, 0] <= N * CELL_KM)
         & (traj[:, 1] >= 0) & (traj[:, 1] <= N * CELL_KM)).mean()
print(f"[build] in-domain trajectory fraction={inbox*100:.1f}%  "
      f"SAR prior sigma={SIGMA_KM}km  R={R_KM}km")

# ---------------- metadata (explicitly marked as physical simulation) ----------------
meta = {
    "data_source": "physical simulation (leeway drift simulation), NOT real observational data",
    "nrows": N, "ncols": N, "cell_km": CELL_KM,
    "R_cell": R_CELL, "R_km": R_KM, "dt_h": DT_H, "max_queries": MAX_Q,
    "P0_cell": P0_CELL, "P0_km": [x0, y0],
    "leeway_coeff": L_LEEWAY, "sim_seed": SIM_SEED,
    "sar_prior_sigma_km": SIGMA_KM,
    "current_mean_kmh": 0.10, "wind_mean_mps": 3.5,
    "cov_names": cov_names,
    "note": "0/1 detection black box; presence changes dynamically with query_count (time), determined by looking up traj",
}
presence_meta = np.array([json.dumps(meta, ensure_ascii=False)], dtype=object)

out = DATA / "oracle.npz"
np.savez(
    out,
    nrows=N, ncols=N,
    cell_km=CELL_KM, R_km=R_KM, dt_h=DT_H, max_queries=MAX_Q,
    covs=covs,                                   # (19,150,150)
    cov_names=np.array(cov_names),
    traj=traj,                                   # (MAX_Q+1, 2) km coordinates [x,y]
    P0_cell=np.array(P0_CELL),
    presence=sar_prior.astype(np.float32),       # placeholder (the real presence is computed dynamically)
    presence_meta=presence_meta,
    meta=presence_meta,
)
print(f"[build] saved -> {out}  covs={covs.shape} traj={traj.shape}")


# =====================================================================
# oracle class (interface aligned with SpeciesOracle)
# =====================================================================
class MaritimeOracle:
    """Maritime search-and-rescue black-box oracle. query_cell(i,j) -> (pres, covs[:,i,j]).

    presence is dynamic: each query_count advances Δt, look up traj[k] to get the target's current position,
    if the cell center is within R_km of the target then pres=1. covs are static per-cell fields.
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
    # self-check: instantiate and simulate a few queries
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
    print(f"[self-test] per-cell probe hit rate along the trajectory={hits}/{MAX_Q} "
          f"(should be close to {MAX_Q}, since the cell center is within R of the target)")
    o.reset_counter()
    print("[self-test] reset_counter ok, query_count=", o.query_count)
