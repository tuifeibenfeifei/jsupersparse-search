
"""
spiral_scan.py
[宏观] 螺旋扫描：从起点向外逐层扩展，顺时针螺旋

【修复依据】(≤100字)
仅补 reset()：螺旋的当前坐标、方向、leg 计数均为实例状态，跨 episode 不重置会让
第2次起的扫描从上次终点继续外扩，违背需求第一节第3条「无状态重置」。
螺旋推进逻辑保持原设计不变。
"""

from typing import Tuple
from .base_finder import BaseFinder


class SpiralScanFinder(BaseFinder):
    """
    螺旋扫描：从起始区块出发，按顺时针螺旋方向逐层向外扩展
    适合均匀覆盖、无先验信息的探索场景
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self._x, self._z = 0, 0
        self._dx, self._dz = 1, 0
        self._steps_taken = 0
        self._steps_in_leg = 1
        self._leg = 0

    @property
    def name(self) -> str:
        return "spiral_scan"

    def reset(self) -> None:
        """重置螺旋状态，确保每个 episode 都从原点 (0,0) 重新螺旋扫描"""
        self._x, self._z = 0, 0
        self._dx, self._dz = 1, 0
        self._steps_taken = 0
        self._steps_in_leg = 1
        self._leg = 0

    def get_next_chunk(self, current_x: int, current_z: int) -> Tuple[int, int]:
        result = (self._x, self._z)
        # 移动到下一个位置
        self._x += self._dx
        self._z += self._dz
        self._steps_taken += 1
        # 当前 leg 走完后转向（右旋90度）
        if self._steps_taken == self._steps_in_leg:
            self._steps_taken = 0
            self._dx, self._dz = -self._dz, self._dx
            self._leg += 1
            if self._leg % 2 == 0:
                self._steps_in_leg += 1
        return result
