# src/traversal_strategies/corners_and_center.py
"""[微观] 策略5：四角+中心共5个点（Stage1基准策略）"""
from typing import List, Tuple
from .base_strategy import BaseStrategy
from ._utils import generate_corner_points


class CornersAndCenterStrategy(BaseStrategy):
    """四角+中心采样：查询区块四个角 + 正中心，共5个坐标点"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "corners_and_center"

    def get_points(self, chunk_x: int, chunk_z: int, temperature: float = None) -> List[Tuple[int, int]]:
        cx = chunk_x * 16 + 8
        cz = chunk_z * 16 + 8
        return [(cx, cz)] + generate_corner_points(chunk_x, chunk_z)