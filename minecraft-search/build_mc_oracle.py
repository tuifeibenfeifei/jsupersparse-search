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
"""build_mc_oracle.py — Materialize Minecraft 1.17 world grid as SpeciesOracle-compatible npz.

For each world_seed (1..20), scan a 100x100 grid (each cell = 1 chunk = 16x16 blocks; world range ±800 blocks),
Materialize:
  presence: (100,100) bool — whether deep ocean biome (deep_ocean / deep_lukewarm / deep_cold / deep_frozen)
  covs:     (K=5, 100, 100) float32 — covariate vectors:
      cov0  temp_true     : mc.get_temperature(x, z)         —— true ocean temperature field (correlated with presence)
      cov1  temp_coarse   : mc.get_temperature(x//4*4, z//4*4) —— coarse-scale temperature (correlated with cov0 but smoother)
      cov2  temp_offset   : mc.get_temperature(x+128, z+128) —— offset temperature (weakly correlated)
      cov3  noise_humidity : pseudo-noise (hash salt="humidity")   —— pure noise, independent of presence
      cov4  noise_continent: pseudo-noise (hash salt="continent") —— pure noise, independent of presence

Design intent: let the framework's Presence-AUC online sparsification automatically select the true signal (cov0) from 5 covariates,
while sparsifying out the pure noise (cov3/cov4). This is a direct demonstration of paper innovation point 2.

Output: minecraft-search/oracle_npz/mc_world_seed{N}.npz
"""
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(REPO_ROOT)
sys.path.insert(0, str(ROOT))

from src.cubiomes_wrapper import CubiomesWrapper  # noqa: E402

OUT_DIR = ROOT / "minecraft-search" / "oracle_npz"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N = 100
WS_LIST = list(range(1, 21))
CELL = 16  # 16 blocks per cell
HALF = N // 2


def _cell_to_world(i: int, j: int):
    return (j - HALF) * CELL, (i - HALF) * CELL


def _pseudo_noise(salt: str, world_seed: int, x: int, z: int) -> float:
    """Hash pseudo-noise: [-1, 1], completely independent of presence."""
    h = hashlib.md5(f"{salt}_{world_seed}_{x}_{z}".encode()).hexdigest()
    return (int(h[:8], 16) / 0xFFFFFFFF) * 2.0 - 1.0


def build_one(world_seed: int):
    mc = CubiomesWrapper(seed=world_seed)
    presence = np.zeros((N, N), dtype=bool)
    covs = np.zeros((5, N, N), dtype=np.float32)

    for i in range(N):
        for j in range(N):
            x, z = _cell_to_world(i, j)
            presence[i, j] = mc.is_deep_ocean(x, z)
            covs[0, i, j] = mc.get_temperature(x, z)
            covs[1, i, j] = mc.get_temperature((x // 4) * 4, (z // 4) * 4)
            covs[2, i, j] = mc.get_temperature(x + 128, z + 128)
            covs[3, i, j] = _pseudo_noise("humidity", world_seed, x, z)
            covs[4, i, j] = _pseudo_noise("continent", world_seed, x, z)

    # Metadata aligned with the SpeciesOracle interface
    np.savez(
        OUT_DIR / f"mc_world_seed{world_seed}.npz",
        presence=presence,
        covs=covs,
        cov_names=np.array([
            "temp_true", "temp_coarse", "temp_offset",
            "noise_humidity", "noise_continent",
        ]),
        nrows=N,
        ncols=N,
        lon0=-HALF * CELL,   # Pseudo-coordinates: using block coordinates as proxy
        lat0=-HALF * CELL,
        cell_deg=float(CELL),
        study_poly=np.array([-HALF * CELL, HALF * CELL, -HALF * CELL, HALF * CELL]),
        world_seed=world_seed,
    )
    return int(presence.sum())


def main():
    t0 = time.time()
    print(f"[build] Materializing {len(WS_LIST)} world seeds, each {N}x{N} grid...")
    for ws in WS_LIST:
        ts = time.time()
        n_hit = build_one(ws)
        print(f"  world_seed={ws:3d}  deep-ocean cells={n_hit:4d}/10000  "
              f"({(time.time()-ts):.1f}s)")
    print(f"[build] Done, total time {(time.time()-t0)/60:.1f} min -> {OUT_DIR}")


if __name__ == "__main__":
    main()
