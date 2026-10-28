# src/traversal_strategies/corner_points.py
"""Micro strategy 2: query the four chunk corners."""
from typing import List, Tuple
from .base_strategy import BaseStrategy
from ._utils import generate_corner_points


class CornerPointsStrategy(BaseStrategy):
    """Corner sampling: query the four corners of the chunk."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "corner_points"

    def get_points(self, chunk_x: int, chunk_z: int, temperature: float = None) -> List[Tuple[int, int]]:
        return generate_corner_points(chunk_x, chunk_z)