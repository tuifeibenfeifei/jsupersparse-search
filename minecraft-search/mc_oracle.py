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
"""mc_oracle.py — Minecraft 1.17 格网黑盒 oracle（供 baselines 用）。

游戏场景语义（论文用）：
  - 玩家智能体在未知 Minecraft 世界里探索，目标是定位"海底神殿"（Ocean Monument）
    的生成位置。海底神殿只生成在深海群系（deep_ocean / deep_lukewarm /
    deep_cold / deep_frozen_ocean），因此 presence=1 即"该格为深海、具备神殿生成条件"。
  - 协变量 = 该格的海洋温度噪声场（cubiomes getTemperatureNoise），它能精确区分
    五种海洋变体（Spearman=1.0），但不直接告诉智能体"这里是不是深海"。
  - 黑盒：智能体不知道世界种子对应的群系地图，只能逐格查询 presence + 读温度。

格网口径：
  - nrows=ncols=100，单元格 (i,j) 对应 MC 世界方块坐标 x=(i-50)*16, z=(j-50)*16
    （每格 = 1 个区块 = 16x16 方块，世界范围约 ±800 方块）。
  - query_count 只计 presence（is_deep_ocean）调用，与框架预算口径对齐；
    covariate（温度）查询不计费，因为 baseline 不使用温度引导。

与框架 apples-to-apples：
  - 框架 DeepSeaExplorer 内部直接调用同一个 cubiomes_wrapper（同 seed、同世界）。
  - 本 oracle 只是把同一 cubiomes 黑盒包装成 baselines 通用的 query_cell(i,j) 接口。
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
        self._w = 16  # 每格 16 方块（1 区块）
        self._c = n // 2  # 中心偏移
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
        """返回 (presence: bool, covariate: float)。越界返回 (None, None)。"""
        if not (0 <= i < self.nrows and 0 <= j < self.ncols):
            return None, None
        x, z = self._cell_to_world(i, j)
        presence = bool(self._mc.is_deep_ocean(x, z))
        self.query_count += 1
        cov = self._mc.get_temperature(x, z)  # 协变量，不计费
        return presence, cov

    @property
    def world_seed(self):
        return self._world_seed


if __name__ == "__main__":
    # 快速自检：扫 5x5 格看 presence 分布
    o = MCGridOracle(world_seed=42, n=100)
    hits = 0
    for i in range(0, 100, 10):
        for j in range(0, 100, 10):
            p, t = o.query_cell(i, j)
            if p:
                hits += 1
    print(f"seed=42, 10x10 采样中深海格数={hits}, query_count={o.query_count}")
