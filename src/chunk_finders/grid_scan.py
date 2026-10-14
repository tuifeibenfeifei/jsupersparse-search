
"""
grid_scan.py
[宏观] 间距网格扫描：按「深海点间距阈值」对齐的网格由内向外逐环扫描

设计依据（Stage1 实测数据驱动，非人工先验）：
    Stage1 九种微观策略的排名与「每区块查询数」完全单调反相关
    （center_point 1查询/区块 → 32.84点；grid_3x3 9查询/区块 → 29.30点），
    证明瓶颈是「覆盖广度」而非「区块内采样密度」。

    进一步核算 ray_scan 的浪费来源：
        9138 次查询 × 深海命中率约27% ≈ 2400 次命中，但仅 32.84 个点通过间距校验，
        即约 98.6% 的命中因落在已有点的排斥区（阈值 416 格）内被丢弃。
        根因是 ray_scan 自原点辐射、环间距 64 格远小于阈值 416 格，
        连续多环的命中点全部挤在同一排斥区内。

    本策略令网格间距 = 间距阈值（416 格 = 26 区块），使相邻格点欧氏距离恰为阈值。
    依据 database._is_valid_point 的判据（dx²+dz² < threshold² 才拒绝），
    距离恰好等于阈值判定为有效，故每次命中必然是合法新点，排斥浪费趋近于零。

    覆盖顺序为「由内向外逐环」（切比雪夫环），保证近处优先、符合需求第二节的
    渐进扩散语义；环内格点间距恒定，不存在重复覆盖或空洞。
"""

from typing import Tuple, List
from .base_finder import BaseFinder


class GridScanFinder(BaseFinder):
    """
    间距网格扫描：以 (0,0) 为中心，按 grid_spacing_chunks 的整数倍构成网格，
    由内向外逐环（切比雪夫距离）返回格点。

    配置项：
        grid_spacing_chunks (int): 网格间距（区块）。缺省时由
            deep_sea_distance_threshold // 16 推导，即阈值对应的区块数。
        deep_sea_distance_threshold (float): 深海点最小间距（格），用于推导缺省间距。
    """

    _BLOCKS_PER_CHUNK = 16

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        threshold = self.config.get("deep_sea_distance_threshold", 416)
        derived = int(threshold) // self._BLOCKS_PER_CHUNK
        spacing = self.config.get("grid_spacing_chunks", derived)
        self._spacing = max(1, int(spacing))
        self._ring = 0
        self._ring_points: List[Tuple[int, int]] = [(0, 0)]
        self._ring_idx = 0

    @property
    def name(self) -> str:
        return "grid_scan"

    @property
    def spacing_chunks(self) -> int:
        """当前网格间距（区块），供调度脚本记录与核对"""
        return self._spacing

    def reset(self) -> None:
        """重置到原点环，确保每个 episode 都从 (0,0) 重新由内向外扫描"""
        self._ring = 0
        self._ring_points = [(0, 0)]
        self._ring_idx = 0

    @staticmethod
    def _build_ring(k: int) -> List[Tuple[int, int]]:
        """
        构建切比雪夫距离恰为 k 的环上全部整数格点，共 8k 个（k>=1）。
        上/下两条边各含 2k+1 点，左/右两条边各含 2k-1 点，合计 8k，无重复。
        """
        if k <= 0:
            return [(0, 0)]
        pts: List[Tuple[int, int]] = []
        for i in range(-k, k + 1):
            pts.append((i, -k))
            pts.append((i, k))
        for j in range(-k + 1, k):
            pts.append((-k, j))
            pts.append((k, j))
        return pts

    def get_next_chunk(self, current_x: int, current_z: int) -> Tuple[int, int]:
        if self._ring_idx >= len(self._ring_points):
            self._ring += 1
            self._ring_points = self._build_ring(self._ring)
            self._ring_idx = 0
        gi, gj = self._ring_points[self._ring_idx]
        self._ring_idx += 1
        return (gi * self._spacing, gj * self._spacing)
