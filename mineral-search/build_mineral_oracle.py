# -*- coding: utf-8 -*-
"""build_mineral_oracle.py — Sequential-decoding two-stage oracle for mineralized outcrop to subsurface ore body
================================================================================
[SYNTHETIC GEOLOGICAL MODEL / PHYSICS SIMULATION] This oracle does not use real deposit data; everything is a synthetic geological model:
  - Stage A (surface mineralized outcrop): 120x120 cells, 1 km per cell. Three mineralized structures (strike 30/75/120 deg,
    each with +/-20 deg strike jitter); the surface outcrop is a +/-2 cell band along each structure line.
  - Stage B (three-dimensional subsurface ore body): 80x80x40 cells (z = depth 0-2 km, 50 m per layer). The ore body is a
    sheet along the true structural plane (true strike 75 deg / true dip 70 deg) with +/-1 cell thickness, plus ~0.3% random
    weak anomalies (false positives).
USGS MRDS: not integrated as an oracle within the time limit (public deposit points carry no co-located three-dimensional structural-plane geometry, so they cannot directly
serve as a black-box query oracle); following the task fallback, a synthetic geological model is used and explicitly labelled as such.

Coordinate convention (world coordinates, km):
  x = east, y = north, z = vertically downward (depth).
  Strike theta: clockwise from true north (0 deg = north, 90 deg = east, 180 deg = south).
  Dip azimuth = theta + 90 deg; dip angle phi (angle to the horizontal, 70 deg is steep).
  Plane normal n = (cos(theta) sin(phi), -sin(theta) sin(phi), -cos(phi)) (unit vector).

Grid mapping:
  Stage A cell (i,j) centre world coordinates: x = j+0.5, y = (119-i)+0.5 = 119.5-i (i=0 is the northern edge).
  Stage B cell (k,i,j) centre world coordinates: x = j+0.5, y = (79-i)+0.5 = 79.5-i, z = (k+0.5)*0.05.
  The two stages cover different space-time blocks and their maps are never fused; block B is shifted 20 km west/south of block A
  (A centre (60,60) corresponds to B centre (40,40)); decoding applies this shift to map the centroid of stage-A discoveries back into B.
"""
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(parents=True, exist_ok=True)

# ---------------- Global geological parameters (written into npz attrs) ----------------
STRIKES_DEG = [30.0, 75.0, 120.0]      # Strike of the three surface mineralized structures
MAIN_IDX = 1                            # The second structure (75 deg) is the main mineralizing structure (stage-B ore body)
DIP_DEG = 70.0                          # True dip of the subsurface structural plane (steep)
DIP_JITTER_DEG = 0.0                    # True plane dip is fixed (synthetic, controllable)
# Surface trace pass-through point (cx, cy) of each structure in world coordinates
# S2 (main mineralizing) sits at A world (60,60) so that, after the decode mapping into B, it lands exactly on the B true ore-plane origin (40,40)
# (block B is shifted CX_SHIFT=20 to the south-west of A: B = A - 20)
TRACE_POINTS = [(30.0, 30.0), (60.0, 60.0), (95.0, 95.0)]
OUTCROP_HALF_WIDTH_KM = 2.0             # Half-width of the surface outcrop band (cells)
# Surface traces of the three structures are finite segments (half-length in km); the main structure S2 is the longest so that RANSAC decoding favours the main strike
TRACE_HALF_LEN_KM = [12.0, 45.0, 12.0]
ORE_HALF_THICK_KM = 0.6                 # Half-thickness of the ore sheet (km), roughly +/-1 cell
NOISE_ANOMALY_RATE = 0.003              # Rate of random weak anomalies (false positives) in stage B

NA = 120                                # Stage A side length in cells
CA = 1.0                                # Stage A cell size in km
K_B, L_B, J_B = 40, 80, 80              # Stage B three-dimensional dimensions
DZ_KM = 0.05                            # B layer thickness in km (40*0.05 = 2 km)
CX_SHIFT = 20.0                         # South-west shift of block B relative to block A (km)

META = dict(
    model="synthetic_geological_model",
    note="synthetic geological model / physics simulation; NOT real deposit data; USGS MRDS not integrated as an oracle within the time limit",
    strikes_deg=STRIKES_DEG, main_idx=MAIN_IDX, dip_deg=DIP_DEG,
    trace_points=TRACE_POINTS,
    outcrop_half_width_km=OUTCROP_HALF_WIDTH_KM,
    trace_half_len_km=TRACE_HALF_LEN_KM,
    ore_half_thick_km=ORE_HALF_THICK_KM,
    noise_anomaly_rate=NOISE_ANOMALY_RATE,
    NA=NA, CA=CA, K_B=K_B, L_B=L_B, J_B=J_B, DZ_KM=DZ_KM, CX_SHIFT=CX_SHIFT,
)


# ---------------- Geometry helpers ----------------
def plane_normal(strike_deg: float, dip_deg: float):
    θ = np.deg2rad(strike_deg)
    φ = np.deg2rad(dip_deg)
    # n̂ = (cosθ sinφ, -sinθ sinφ, -cosφ)
    return np.array([np.cos(θ) * np.sin(φ),
                     -np.sin(θ) * np.sin(φ),
                     -np.cos(φ)], dtype=np.float64)


def dist_to_trace_line(x, y, cx, cy, strike_deg):
    """Perpendicular distance (km) from a point (x,y) to the surface trace through (cx,cy) with the given strike."""
    θ = np.deg2rad(strike_deg)
    sx, sy = np.sin(θ), np.cos(θ)          # Horizontal unit vector along strike (east, north)
    # Cross product = (x-cx)*sy - (y-cy)*sx
    return np.abs((x - cx) * sy - (y - cy) * sx)


def signed_dist_to_plane_3d(x, y, z, cx, cy, strike_deg, dip_deg):
    """Signed distance (km) from a point to the plane through (cx,cy,z=0) with the given strike and dip."""
    n = plane_normal(strike_deg, dip_deg)
    return n[0] * (x - cx) + n[1] * (y - cy) + n[2] * z


# ---------------- Stage A: surface outcrop oracle ----------------
def build_stageA(seed=20240930):
    rng = np.random.default_rng(seed)
    # Grid-centre world coordinates
    jg, ig = np.meshgrid(np.arange(NA) + 0.5, np.arange(NA) + 0.5)  # (NA,NA)
    x = jg * CA                       # east, km
    y = (NA - 1 - ig + 0.5) * CA      # north, km (i=0 is the northern edge)

    # ---- presence: outcrop bands of the three structures (finite segments) ----
    presence = np.zeros((NA, NA), dtype=bool)
    true_struct_id = -np.ones((NA, NA), dtype=np.int8)   # 0/1/2 marks which structure a cell belongs to
    for sid, (strike, (cx, cy)) in enumerate(zip(STRIKES_DEG, TRACE_POINTS)):
        θ = np.deg2rad(strike)
        sx, sy = np.sin(θ), np.cos(θ)          # Horizontal unit vector along strike (east, north)
        d = np.abs((x - cx) * sy - (y - cy) * sx)   # perpendicular distance to the trace
        t = (x - cx) * sx + (y - cy) * sy           # coordinate projected along the trace
        half = TRACE_HALF_LEN_KM[sid]
        band = (d <= OUTCROP_HALF_WIDTH_KM) & (np.abs(t) <= half)
        presence |= band
        true_struct_id[band] = sid

    # ---- 19 covariates ----
    covs = np.zeros((19, NA, NA), dtype=np.float32)
    # 0: elevation (smooth relief)
    elev = (300 + 200 * np.sin(x / 18.0) * np.cos(y / 22.0)
            + 80 * np.sin((x + y) / 9.0))
    covs[0] = elev.astype(np.float32)
    # 1: slope (elevation-gradient magnitude)
    gy, gx = np.gradient(elev, CA)
    slope = np.hypot(gx, gy)
    covs[1] = slope.astype(np.float32)
    # 2,3,4: distance to the three faults. Fault 1 follows the main structure (S2, 75 deg), fault 2 follows S1 (30 deg), fault 3 has a random strike
    fault_geoms = [(TRACE_POINTS[1][0], TRACE_POINTS[1][1], STRIKES_DEG[1]),
                   (TRACE_POINTS[0][0], TRACE_POINTS[0][1], STRIKES_DEG[0]),
                   (60.0, 60.0, 105.0)]
    for fi, (cx, cy, st) in enumerate(fault_geoms):
        covs[2 + fi] = dist_to_trace_line(x, y, cx, cy, st).astype(np.float32)
    # 5: lithology index (smooth random field)
    lith = gaussian_filter(rng.normal(size=(NA, NA)), sigma=6.0)
    covs[5] = lith.astype(np.float32)
    # 6..18: noise covariates (13 Gaussian noise fields)
    for k in range(6, 19):
        covs[k] = rng.normal(size=(NA, NA)).astype(np.float32)

    cov_names = ["elevation", "slope", "fault1_dist", "fault2_dist", "fault3_dist",
                 "lithology"] + [f"noise{i}" for i in range(13)]

    pres_rate = presence.mean()
    print(f"[stageA] presence cells={int(presence.sum())}/{NA*NA} = {pres_rate*100:.2f}%")
    np.savez(DATA / "stageA_surface.npz",
             presence=presence, covs=covs,
             true_struct_id=true_struct_id,
             x=x.astype(np.float32), y=y.astype(np.float32),
             cov_names=np.array(cov_names),
             nrows=NA, ncols=NA, cell_km=CA,
             **{f"META_{k}": (v if not isinstance(v, (list, tuple))
                              else np.array(v)) for k, v in META.items()})
    return presence, covs


# ---------------- Stage B: 3-D subsurface ore-body oracle (base version; channel 1 is injected by the decoder) ----------------
def build_stageB_base(seed=777):
    rng = np.random.default_rng(seed)
    # Grid-centre world coordinates
    jj, ii = np.meshgrid(np.arange(J_B) + 0.5, np.arange(L_B) + 0.5)  # (L_B,J_B)
    x2d = jj.copy()                    # east, km
    y2d = (L_B - 1 - ii + 0.5)         # north, km
    z3 = (np.arange(K_B) + 0.5) * DZ_KM   # depth, km (K_B,)

    cx_true, cy_true = 40.0, 40.0     # the true mineralizing plane at the origin of the B-block world coordinates
    # (corresponds to stage-A S2 trace point (60,60); B = A - CX_SHIFT = 20 gives (40,40))
    strike_true = STRIKES_DEG[MAIN_IDX]

    # True ore sheet: |signed distance| <= half-thickness
    XX = np.broadcast_to(x2d[None, :, :], (K_B, L_B, J_B))
    YY = np.broadcast_to(y2d[None, :, :], (K_B, L_B, J_B))
    ZZ = np.broadcast_to(z3[:, None, None], (K_B, L_B, J_B))
    sd = signed_dist_to_plane_3d(XX, YY, ZZ, cx_true, cy_true,
                                 strike_true, DIP_DEG)
    ore = np.abs(sd) <= ORE_HALF_THICK_KM

    # Random weak anomalies (false positives): not on the ore body
    noise_anom = rng.random((K_B, L_B, J_B)) < NOISE_ANOMALY_RATE
    noise_anom &= ~ore
    presence = ore | noise_anom

    # ---- 19 covariates (channel 1 is a zero placeholder; the run script injects the decoded strike) ----
    covs = np.zeros((19, K_B, L_B, J_B), dtype=np.float32)
    # 0: depth
    covs[0] = np.broadcast_to(z3[:, None, None], (K_B, L_B, J_B)).astype(np.float32)
    # 1: distance to the predicted structural plane (0 placeholder, injected by run_mineral; the true-plane distance is stored here for comparison/evaluation)
    covs[1] = sd.astype(np.float32)   # placeholder: signed distance to the true plane (used for evaluation; overwritten during the real run)
    # 2: host lithology index (smooth random field; lithology is favourable near the structural belt hosting the ore body)
    lith2 = gaussian_filter(rng.normal(size=(K_B, L_B, J_B)), sigma=5.0)
    covs[2] = lith2.astype(np.float32)
    # 3: fault proximity (a known fault inside block B that is parallel to the mineralizing plane but offset 5 km along the normal; weak signal;
    #    it does not coincide with the true ore plane, so it cannot act as a cheating oracle signal)
    fault_d = np.abs(np.abs(sd) - 5.0)
    covs[3] = fault_d.astype(np.float32)
    # 4..18: noise covariates
    for k in range(4, 19):
        covs[k] = rng.normal(size=(K_B, L_B, J_B)).astype(np.float32)

    cov_names = ["depth_km", "dist_pred_plane", "host_lithology", "fault_prox"] + \
                [f"noise{i}" for i in range(15)]

    ore_rate = ore.mean()
    pres_rate = presence.mean()
    print(f"[stageB] ore cells={int(ore.sum())} ({ore_rate*100:.2f}%), "
          f"presence(ore+noise)={int(presence.sum())} ({pres_rate*100:.2f}%), "
          f"vol={K_B*L_B*J_B}")

    np.savez(DATA / "stageB_subsurface.npz",
             presence=presence, ore_mask=ore, noise_anom=noise_anom,
             covs=covs, sd_true=sd.astype(np.float32),
             cov_names=np.array(cov_names),
             K=K_B, L=L_B, J=J_B, dz_km=DZ_KM, cell_km=1.0,
             strike_true_deg=strike_true, dip_deg=DIP_DEG,
             cx_true=cx_true, cy_true=cy_true,
             ore_half_thick_km=ORE_HALF_THICK_KM,
             **{f"META_{k}": (v if not isinstance(v, (list, tuple))
                              else np.array(v)) for k, v in META.items()})
    return presence, ore, covs


# ---------------- Oracle classes (matching the SpeciesOracle / Oracle3D interfaces) ----------------
class MineralOracle2D:
    """Matches the interface required by species_agent.SpeciesOracle:
    covs(K,nrows,ncols), presence(nrows,ncols), nrows,ncols,
    in_bounds(i,j), query_cell(i,j)->(pres,cov), query_count, reset_counter().
    Constructed from stageA_surface.npz."""

    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.nrows = int(d["nrows"])
        self.ncols = int(d["ncols"])
        self.presence = d["presence"]
        self.covs = d["covs"]                 # (19, nrows, ncols)
        self.cov_names = [str(x) for x in d["cov_names"]]
        self.true_struct_id = d["true_struct_id"]
        self.query_count = 0

    def in_bounds(self, i, j):
        return 0 <= i < self.nrows and 0 <= j < self.ncols

    def query_cell(self, i, j):
        self.query_count += 1
        if not self.in_bounds(i, j):
            return None, None
        return bool(self.presence[i, j]), self.covs[:, i, j].copy()

    @property
    def total_cells(self):
        return self.nrows * self.ncols

    def presence_cells(self):
        return [tuple(c) for c in np.argwhere(self.presence)]

    def reset_counter(self):
        self.query_count = 0


class MineralOracle3D:
    """Matches the interface required by species_agent_3d.Oracle3D:
    covs(Kcov,K,L,J), presence(K,L,J), K,L,J,
    in_bounds(k,i,j), query_cell(k,i,j)->(pres,cov), query_count,
    total_cells, reset_counter().
    Allows run_mineral to inject the decoded dist_pred_plane channel (overwriting covs[1])."""

    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.presence = d["presence"]           # (K,L,J)
        self.ore_mask = d["ore_mask"]
        self.noise_anom = d["noise_anom"]
        self.covs = d["covs"].copy()            # (19,K,L,J)
        self.cov_names = [str(x) for x in d["cov_names"]]
        self.K = int(d["K"])
        self.L = int(d["L"])
        self.J = int(d["J"])
        self.dz = float(d["dz_km"])
        self.strike_true_deg = float(d["strike_true_deg"])
        self.dip_deg = float(d["dip_deg"])
        self.cx_true = float(d["cx_true"])
        self.cy_true = float(d["cy_true"])
        self.ore_half_thick = float(d["ore_half_thick_km"])
        self.query_count = 0

    def inject_predicted_plane(self, dist_map):
        """Inject the stage-A-decoded 'distance to the predicted structural plane' covariate (channel 1)."""
        assert dist_map.shape == (self.K, self.L, self.J), dist_map.shape
        self.covs[1] = dist_map.astype(np.float32)

    def in_bounds(self, k, i, j):
        return 0 <= k < self.K and 0 <= i < self.L and 0 <= j < self.J

    def query_cell(self, k, i, j):
        self.query_count += 1
        if not self.in_bounds(k, i, j):
            return None, None
        return bool(self.presence[k, i, j]), self.covs[:, k, i, j].copy()

    @property
    def total_cells(self):
        return self.K * self.L * self.J

    def presence_cells(self):
        return [tuple(c) for c in np.argwhere(self.presence)]

    def reset_counter(self):
        self.query_count = 0


if __name__ == "__main__":
    print("=== Building the synthetic geological model oracle ===")
    build_stageA()
    build_stageB_base()
    print("=== Done: data/stageA_surface.npz, data/stageB_subsurface.npz ===")
