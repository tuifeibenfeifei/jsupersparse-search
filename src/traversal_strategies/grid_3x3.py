# src/traversal_strategies/grid_3x3.py
"""[微观] 策略4：把区块划成3x3网格查9个点"""
from typing import List, Tuple
from .base_strategy import BaseStrategy
from ._utils import generate_grid_points


class Grid3x3Strategy(BaseStrategy):
    """3x3网格采样：将区块划分为3x3均匀网格，查询9个网格交点"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "grid_3x3"

    def get_points(self, chunk_x: int, chunk_z: int, temperature: float = None) -> List[Tuple[int, int]]:
        return generate_grid_points(chunk_x, chunk_z, rows=3, cols=3)