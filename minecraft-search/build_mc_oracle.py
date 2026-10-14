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
"""build_mc_oracle.py — 物化 Minecraft 1.17 世界格网为 SpeciesOracle 兼容的 npz。

对每个 world_seed（1..20），扫 100x100 格网（每格=1 区块=16x16 方块，世界范围 ±800 方块），
物化：
  presence: (100,100) bool — 是否深海群系（deep_ocean / deep_lukewarm / deep_cold / deep_frozen）
  covs:     (K=5, 100, 100) float32 — 协变量向量：
      cov0  temp_true     : mc.get_temperature(x, z)         —— 真海洋温度场（与 presence 相关）
      cov1  temp_coarse   : mc.get_temperature(x//4*4, z//4*4) —— 粗尺度温度（与 cov0 相关但更平滑）
      cov2  temp_offset   : mc.get_temperature(x+128, z+128) —— 偏移温度（弱相关）
      cov3  noise_humidity : 伪噪声（hash salt="humidity"）   —— 纯噪声，与 presence 无关
      cov4  noise_continent: 伪噪声（hash salt="continent"） —— 纯噪声，与 presence 无关

设计意图：让框架的 presence-AUC 在线稀疏化自动从 5 个 covariate 里选出真信号（cov0），
稀疏掉纯噪声（cov3/cov4）。这正是论文创新点 2 的直接演示。

输出：minecraft-search/oracle_npz/mc_world_seed{N}.npz
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
CELL = 16  # 每格 16 方块
HALF = N // 2


def _cell_to_world(i: int, j: int):
    return (j - HALF) * CELL, (i - HALF) * CELL


def _pseudo_noise(salt: str, world_seed: int, x: int, z: int) -> float:
    """hash 伪噪声：[-1, 1]，与 presence 完全无关。"""
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

    # 与 SpeciesOracle 接口对齐的元数据
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
        lon0=-HALF * CELL,   # 伪坐标：以方块坐标代
        lat0=-HALF * CELL,
        cell_deg=float(CELL),
        study_poly=np.array([-HALF * CELL, HALF * CELL, -HALF * CELL, HALF * CELL]),
        world_seed=world_seed,
    )
    return int(presence.sum())


def main():
    t0 = time.time()
    print(f"[build] 物化 {len(WS_LIST)} 个世界种子，每种子 {N}x{N} 格网...")
    for ws in WS_LIST:
        ts = time.time()
        n_hit = build_one(ws)
        print(f"  world_seed={ws:3d}  深海格数={n_hit:4d}/10000  "
              f"({(time.time()-ts):.1f}s)")
    print(f"[build] 完成，总耗时 {(time.time()-t0)/60:.1f} min -> {OUT_DIR}")


if __name__ == "__main__":
    main()
