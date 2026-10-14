# src/traversal_strategies/t_shape.py
"""[微观] 策略7：按T字形布局查点"""
from typing import List, Tuple
from .base_strategy import BaseStrategy


class TShapeStrategy(BaseStrategy):
    """T字形采样：顶部3点 + 中间1点 + 底部1点，形成T字布局"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "t_shape"

    def get_points(self, chunk_x: int, chunk_z: int, temperature: float = None) -> List[Tuple[int, int]]:
        x0, z0 = chunk_x * 16, chunk_z * 16
        x1, z1 = (chunk_x + 1) * 16 - 1, (chunk_z + 1) * 16 - 1
        cx = x0 + 8
        return [
            (cx, z0),      # 顶部中（最高优先级）
            (x0, z0),      # 顶部左角
            (x1, z0),      # 顶部右角
            (cx, z0 + 8),  # 中心偏上
            (cx, z1),      # 底部中
        ]