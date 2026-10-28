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
"""mc_oracle.py — Minecraft 1.17 grid black-box oracle (for baselines).

Game scenario semantics (for paper):
  - Player agent explores an unknown Minecraft world, goal is to locate the "Ocean Monument"
    spawn location. Ocean Monuments only spawn in deep ocean biomes (deep_ocean / deep_lukewarm /
    deep_cold / deep_frozen_ocean), so presence=1 means "this cell is deep ocean and meets monument spawn conditions".
  - Covariate = the cell's ocean temperature noise field (cubiomes getTemperatureNoise), which can precisely distinguish
    five ocean variants (Spearman=1.0), but does not directly tell the agent "whether this is deep ocean".
  - Black box: the agent does not know the biome map corresponding to the world seed; it can only query presence cell by cell + read temperature.

Grid specification:
  - nrows=ncols=100; cell (i,j) corresponds to MC world block coordinates x=(i-50)*16, z=(j-50)*16
    (each cell = 1 chunk = 16x16 blocks; world range ≈ ±800 blocks).
  - query_count only counts presence (is_deep_ocean) calls, aligned with the framework budget metric;
    covariate (temperature) queries are not billed, because baselines do not use temperature guidance.

Apples-to-apples with the framework:
  - The framework DeepSeaExplorer internally calls the same cubiomes_wrapper (same seed, same world).
  - This oracle simply wraps the same cubiomes black box into the common baselines query_cell(i,j) interface.
"""
import sys
from pathlib import Path

ROOT = Path(REPO_ROOT)
sys.path.insert(0, str(ROOT))

from src.cubiomes_wrapper import CubiomesWrapper  # noqa: E402


class MCGridOracle:
    def __init__(self, world_seed: int = 1, n: int = 100):
        self.nrows = n
        self.ncols = n
        self._w = 16  # 16 blocks per cell (1 chunk)
        self._c = n // 2  # Center offset
        self._mc = CubiomesWrapper(seed=world_seed)
        self._world_seed = world_seed
        self.query_count = 0

    def set_seed(self, world_seed: int):
        self._world_seed = world_seed
        self._mc.set_seed(world_seed)
        self.query_count = 0

    def reset_counter(self):
        self.query_count = 0

    def _cell_to_world(self, i: int, j: int):
        return (j - self._c) * self._w, (i - self._c) * self._w

    def query_cell(self, i: int, j: int):
        """Returns (presence: bool, covariate: float). Out of bounds returns (None, None)."""
        if not (0 <= i < self.nrows and 0 <= j < self.ncols):
            return None, None
        x, z = self._cell_to_world(i, j)
        presence = bool(self._mc.is_deep_ocean(x, z))
        self.query_count += 1
        cov = self._mc.get_temperature(x, z)  # covariate, not billed
        return presence, cov

    @property
    def world_seed(self):
        return self._world_seed


if __name__ == "__main__":
    # Quick self-check: scan a 5x5 grid to see presence distribution
    o = MCGridOracle(world_seed=42, n=100)
    hits = 0
    for i in range(0, 100, 10):
        for j in range(0, 100, 10):
            p, t = o.query_cell(i, j)
            if p:
                hits += 1
    print(f"seed=42, deep ocean cells in 10x10 sampling={hits}, query_count={o.query_count}")
