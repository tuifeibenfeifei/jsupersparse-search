# src/traversal_strategies/diagonal_two_points.py
"""[微观] 策略9：只查对角线上的2个点"""
from typing import List, Tuple
from .base_strategy import BaseStrategy


class DiagonalTwoPointsStrategy(BaseStrategy):
    """对角线采样：仅查询区块主对角线上的2个端点"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "diagonal_two_points"

    def get_points(self, chunk_x: int, chunk_z: int, temperature: float = None) -> List[Tuple[int, int]]:
        x0, z0 = chunk_x * 16, chunk_z * 16
        x1, z1 = (chunk_x + 1) * 16 - 1, (chunk_z + 1) * 16 - 1
        return [(x0, z0), (x1, z1)]