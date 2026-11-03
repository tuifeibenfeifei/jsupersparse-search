# -*- coding: utf-8 -*-
"""build_pollution_oracle.py - oracle construction for abrupt pollution source tracing (physics simulation)

[IMPORTANT] This oracle is a **physics simulation**; it uses no real monitoring or tracer-release data.
The concentration field is computed analytically from a steady-state Gaussian plume model of two-dimensional advection-diffusion,
and is used to test the SpeciesSearcher framework under a fully black-box, 0/1 above-threshold detection setting.

Scene:
  - Two-dimensional water body, 120x120 cells, 500 m per cell (a 60x60 km domain).
  - Unknown abrupt point source: position (sx,sy), source strength Q, release start time t0 and all hydrological parameters are unknown.
  - The concentration field at observation time T equals background advection plus turbulent diffusion plus first-order temporal decay.
  - Detection: a cell is 1 when its concentration is >= the threshold (calibrated so that above-threshold cells are about 7% of the domain), and 0 otherwise.
    The above-threshold region is one continuous plume body with the source at its upstream tip, elongated along the flow direction.

19 covariates (static per cell, fed to sparse AUC-based covariate selection):
  Real (8): 0 dist_to_river / 1 u_east / 2 v_north / 3 D_proxy (substrate) /
           4 turb_proxy (turbulence) / 5 wind / 6 depth / 7 river_density
  Noise (11): 8 turbidity / 9 pH / 10 chlorophyll / 11 suspended_solids /
           12 bottom_temp / 13 salinity / 14 redox / 15 algae_biovolume /
           16 coliform / 17 oil_index / 18 noise_extra
  Noise covariates are smooth random fields unrelated to the plume; AUC should reject them automatically (about 0.5).

Output: pollution-search/data/oracle.npz
"""
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

# ---------------- Grid and physics parameters (dimensionless; the cell is the length unit) ----------------
N = 120                      # 120x120
CELL_M = 500.0               # 500 m per cell
# Source position (cells; row i = north-south, col j = east-west)
I_S, J_S = 60, 20
# Background flow direction (unit vector in the i,j plane): about 17.5 deg north of east
DI, DJ = 0.30, 0.95
U_MAG = 1.0                  # Flow speed (cells per unit time)
D_DIFF = 0.20                # Turbulent diffusion coefficient (cells^2 per unit time)
DECAY = 0.012                # First-order decay rate (per cell downstream)
S0 = 1.5                     # Source regularisation distance (avoids the s=0 singularity)
CHAN_HALF = 5.5              # Channel half-width (cells)
PLUME_LEN = 95.0             # Effective downstream plume length (cells)
TARGET_FRAC = 0.07           # Target fraction of above-threshold cells (middle of the 5-10% range)

COV_NAMES = [
    "dist_to_river", "u_east", "v_north", "D_proxy_substrate",
    "turbulence_proxy", "wind", "water_depth", "river_network_density",
    "turbidity", "pH", "chlorophyll", "suspended_solids",
    "bottom_temp", "salinity", "redox", "algae_biovolume",
    "coliform", "oil_index", "noise_extra",
]
N_REAL = 8   # the first 8 are physically meaningful


def smooth_noise(rng, shape, sigma):
    """Generate a smooth random field (Gaussian-smoothed white noise), normalised to [0,1]."""
    z = rng.standard_normal(shape)
    z = ndimage.gaussian_filter(z, sigma=sigma, mode="wrap")
    z -= z.min()
    z /= (z.max() + 1e-12)
    return z


def main():
    rng = np.random.default_rng(20260930)
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")  # ii=row, jj=col

    # ---- streamwise s / cross-stream r coordinates ----
    di, dj = DI, DJ
    s = (ii - I_S) * di + (jj - J_S) * dj          # projection along the downstream direction
    r = -(ii - I_S) * dj + (jj - J_S) * di         # cross-stream (perpendicular to the flow)

    # ---- steady-state Gaussian plume concentration field (advection dominated, longitudinal diffusion neglected) ----
    # s>=0 includes the source itself (highest concentration there); upstream s<0 is clean water
    s_eff = s + S0
    C = np.zeros((N, N), dtype=np.float64)
    downstream = s >= 0
    sigma = np.full((N, N), np.nan)
    sigma[downstream] = np.sqrt(2.0 * D_DIFF * s_eff[downstream] / U_MAG)
    # centre-line concentration ~ 1/sigma (dilution), Gaussian cross-stream decay, exponential downstream decay
    C[downstream] = (
        (1.0 / (sigma[downstream] + 1e-6))
        * np.exp(-r[downstream] ** 2 / (2.0 * sigma[downstream] ** 2))
        * np.exp(-DECAY * s[downstream] / U_MAG)
    )
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    # truncated beyond the plume length (far field taken as 0)
    C[s > PLUME_LEN] = 0.0

    # ---- threshold calibration so that above-threshold cells ~ TARGET_FRAC ----
    thr = np.quantile(C, 1.0 - TARGET_FRAC)
    presence = C >= thr
    # Ensure at least the source itself is above threshold (it has the highest concentration and should be included; if the threshold cuts it out, lower the threshold)
    if not presence[I_S, J_S]:
        thr = min(thr, C[I_S, J_S] * 0.9)
        presence = C >= thr
    frac = presence.mean()
    print(f"[oracle] threshold={thr:.4e}  above-threshold fraction={frac*100:.2f}%  "
          f"above-threshold cells={int(presence.sum())}")

    # ---- channel mask: one band downstream of the source (along the flow, half-width CHAN_HALF) ----
    chan = (s >= -3) & (s <= PLUME_LEN + 5) & (np.abs(r) <= CHAN_HALF)
    # Cross-channel velocity profile (Gaussian; strongest at the channel centre, decaying towards the banks)
    chan_profile = np.exp(-(r / CHAN_HALF) ** 2)
    in_sflow = s > -2   # flow exists from near the source

    # ---- 19 covariates ----
    covs = np.zeros((19, N, N), dtype=np.float32)

    # 0: distance to the nearest channel (pixels). 0 inside the channel, increasing outwards
    dist_to_chan = ndimage.distance_transform_edt(~chan)
    covs[0] = dist_to_chan.astype(np.float32)

    # 1/2: velocity component fields u (east) / v (north) - flow along the channel, near zero on the floodplain
    flow = chan_profile * in_sflow.astype(float)
    covs[1] = (U_MAG * dj * flow).astype(np.float32)   # east = +j component
    covs[2] = (U_MAG * di * flow).astype(np.float32)   # north = +i component

    # 3: diffusion-coefficient proxy (substrate type): gravel in the channel gives high diffusion, silt on the floodplain low
    covs[3] = (0.25 + 0.75 * flow).astype(np.float32)
    # 4: turbulence-intensity proxy: high inside the channel
    covs[4] = (0.15 + 0.85 * flow).astype(np.float32)
    # 5: wind field: large-scale smooth gradient (weak signal, not plume-specific)
    wind = 2.0 + 0.02 * ii + 0.01 * jj + 0.5 * smooth_noise(rng, (N, N), 6.0)
    covs[5] = wind.astype(np.float32)
    # 6: water depth: deep in the channel (~8 m), shallow on the floodplain (~1 m)
    covs[6] = (1.0 + 7.0 * flow).astype(np.float32)
    # 7: river-network density: local density of channel pixels
    dens = ndimage.gaussian_filter(chan.astype(float), sigma=4.0)
    dens /= (dens.max() + 1e-12)
    covs[7] = dens.astype(np.float32)

    # 8..18: pure-noise smooth random fields (unrelated to the channel or the plume)
    for k in range(8, 19):
        sig = rng.uniform(2.0, 7.0)
        covs[k] = smooth_noise(rng, (N, N), sig).astype(np.float32)

    # ---- save the npz (metadata written as separate arrays, labelled "physics simulation") ----
    meta = (
        "PHYSICS_SIMULATION_NOT_REAL_DATA | "
        "advection-diffusion steady gaussian plume; 2D depth-averaged; "
        "background advection+turbulent diffusion+1st-order decay; "
        "detection=0/1 threshold exceedance."
    )
    np.savez(
        DATA / "oracle.npz",
        covs=covs.astype(np.float32),
        presence=presence.astype(bool),
        cov_names=np.array(COV_NAMES),
        # grid
        nrows=N, ncols=N, cell_m=CELL_M,
        # source parameters (ground truth, used only to evaluate tracing error; invisible to the black-box search)
        source_i=I_S, source_j=J_S,
        flow_di=DI, flow_dj=DJ, u_mag=U_MAG,
        D_diff=D_DIFF, decay=DECAY, s0=S0, plume_len=PLUME_LEN,
        chan_half=CHAN_HALF,
        # threshold and labels
        threshold=float(thr), plume_frac=float(frac),
        target_frac=TARGET_FRAC,
        n_real_covs=N_REAL,
        meta=meta,
    )
    print(f"[oracle] covs shape={covs.shape} dtype={covs.dtype}")
    print(f"[oracle] source cell ({I_S},{J_S}) above threshold = {bool(presence[I_S,J_S])}")
    # most upstream above-threshold cell of the plume (smallest s along the flow)
    pres_idx = np.argwhere(presence)
    s_pres = (pres_idx[:, 0] - I_S) * DI + (pres_idx[:, 1] - J_S) * DJ
    up = pres_idx[np.argmin(s_pres)]
    print(f"[oracle] most upstream above-threshold cell = (i={up[0]},j={up[1]}) s={s_pres.min():.2f}")
    print(f"[oracle] -> {DATA/'oracle.npz'}")


if __name__ == "__main__":
    main()
