
"""
ray_scan.py
[宏观] 射线扫描：从中心向8个方向直线跳跃
对应需求.txt第九节：ray_scan（8方向射线扫描，与zhongxin.py核心逻辑一致）

【修复依据】(≤100字)
仅补 reset()：原实现的方向索引与步长为实例状态，跨 episode 不重置会让第2次起的
扫描轨迹从上次结束处继续外扩，违背需求第一节第3条「无状态重置」。扩散逻辑、
方向表、步长推进方式均保持原设计不变。
"""

from typing import Tuple, List
from .base_finder import BaseFinder


class RayScanFinder(BaseFinder):
    """
    射线扫描：从起始区块出发，向8个方向轮流发射射线
    每完成一轮完整的8方向扫描后，步长增加 4 个区块，形成辐射状向外覆盖

    步长增量 4 区块的依据（需求第七节）：
        短步长 1–2 区块，长步长 4–8 区块；初始跳跃 4 区块 = 海底神殿结构半径。
    """

    # 8个方向：东、东南、南、西南、西、西北、北、东北
    _DIRECTIONS: List[Tuple[int, int]] = [
        (1, 0), (1, -1), (0, -1), (-1, -1),
        (-1, 0), (-1, 1), (0, 1), (1, 1),
    ]

    # 每轮完整8方向扫描后的步长增量（区块）
    _STEP_INCREMENT = 4

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self._dir_idx = 0
        self._step = 0

    @property
    def name(self) -> str:
        return "ray_scan"

    def reset(self) -> None:
        """重置射线状态，确保每个 episode 都从原点 (0,0) 重新辐射扫描"""
        self._dir_idx = 0
        self._step = 0

    def get_next_chunk(self, current_x: int, current_z: int) -> Tuple[int, int]:
        if self._step == 0 and self._dir_idx == 0:
            # 第一次调用：返回起始点 (0, 0)
            self._step = 1
            return (0, 0)

        dx, dz = self._DIRECTIONS[self._dir_idx]
        nx = dx * self._step
        nz = dz * self._step
        self._dir_idx = (self._dir_idx + 1) % 8
        if self._dir_idx == 0:
            # 每轮完整8方向后步长增加4个区块（需求第七节：长步长4-8区块）
            self._step += self._STEP_INCREMENT
        return (nx, nz)
