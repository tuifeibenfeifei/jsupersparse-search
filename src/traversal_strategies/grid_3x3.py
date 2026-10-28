# src/traversal_strategies/grid_3x3.py
"""Micro strategy 4: 3x3 grid over the chunk, nine points."""
from typing import List, Tuple
from .base_strategy import BaseStrategy
from ._utils import generate_grid_points


class Grid3x3Strategy(BaseStrategy):
    """3x3 grid sampling: split the chunk into a uniform 3x3 grid and query the nine intersections."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "grid_3x3"

    def get_points(self, chunk_x: int, chunk_z: int, temperature: float = None) -> List[Tuple[int, int]]:
        return generate_grid_points(chunk_x, chunk_z, rows=3, cols=3)