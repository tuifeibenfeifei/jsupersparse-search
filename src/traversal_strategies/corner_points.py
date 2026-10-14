# src/traversal_strategies/corner_points.py
"""[微观] 策略2：只查区块四个角共4个点"""
from typing import List, Tuple
from .base_strategy import BaseStrategy
from ._utils import generate_corner_points


class CornerPointsStrategy(BaseStrategy):
    """区块四角采样：查询区块四个角落的坐标点"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "corner_points"

    def get_points(self, chunk_x: int, chunk_z: int, temperature: float = None) -> List[Tuple[int, int]]:
        return generate_corner_points(chunk_x, chunk_z)