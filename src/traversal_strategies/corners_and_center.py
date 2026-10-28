# src/traversal_strategies/corners_and_center.py
"""Micro strategy 5: four corners plus centre (Stage-1 baseline)."""
from typing import List, Tuple
from .base_strategy import BaseStrategy
from ._utils import generate_corner_points


class CornersAndCenterStrategy(BaseStrategy):
    """Four corners plus centre: five coordinates per chunk."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "corners_and_center"

    def get_points(self, chunk_x: int, chunk_z: int, temperature: float = None) -> List[Tuple[int, int]]:
        cx = chunk_x * 16 + 8
        cz = chunk_z * 16 + 8
        return [(cx, cz)] + generate_corner_points(chunk_x, chunk_z)