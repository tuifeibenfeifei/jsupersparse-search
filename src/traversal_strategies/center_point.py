# src/traversal_strategies/center_point.py
"""[微观] 策略1：只查区块正中心1个点（默认基准）"""
from typing import List, Tuple
from .base_strategy import BaseStrategy


class CenterPointStrategy(BaseStrategy):
    """区块正中心采样：仅查询区块中心1个坐标点"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "center_point"

    def get_points(self, chunk_x: int, chunk_z: int, temperature: float = None) -> List[Tuple[int, int]]:
        return [(chunk_x * 16 + 8, chunk_z * 16 + 8)]