# src/traversal_strategies/t_shape.py
"""[micro] strategy 7: query points along a T-shaped layout"""
from typing import List, Tuple
from .base_strategy import BaseStrategy


class TShapeStrategy(BaseStrategy):
    """T-shaped sampling: 3 top points + 1 middle point + 1 bottom point, forming a T layout"""

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
            (cx, z0),      # top center (highest priority)
            (x0, z0),      # top-left corner
            (x1, z0),      # top-right corner
            (cx, z0 + 8),  # slightly above center
            (cx, z1),      # bottom center
        ]