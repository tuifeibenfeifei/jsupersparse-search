# src/traversal_strategies/center_point.py
"""Micro strategy 1: query only the chunk centre (default baseline)."""
from typing import List, Tuple
from .base_strategy import BaseStrategy


class CenterPointStrategy(BaseStrategy):
    """Centre sampling: query the single centre coordinate of the chunk."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "center_point"

    def get_points(self, chunk_x: int, chunk_z: int, temperature: float = None) -> List[Tuple[int, int]]:
        return [(chunk_x * 16 + 8, chunk_z * 16 + 8)]