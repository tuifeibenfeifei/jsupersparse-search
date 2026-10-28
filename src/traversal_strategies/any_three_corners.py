# src/traversal_strategies/any_three_corners.py
"""Micro strategy 6: query three randomly chosen corners."""
import random
from typing import List, Tuple
from .base_strategy import BaseStrategy
from ._utils import generate_corner_points


class AnyThreeCornersStrategy(BaseStrategy):
    """Random three-corner sampling: pick 3 of the 4 chunk corners at random."""

    def __init__(self, seed: int = None, **kwargs):
        super().__init__(**kwargs)
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "any_three_corners"

    def get_points(self, chunk_x: int, chunk_z: int, temperature: float = None) -> List[Tuple[int, int]]:
        return self._rng.sample(generate_corner_points(chunk_x, chunk_z), 3)