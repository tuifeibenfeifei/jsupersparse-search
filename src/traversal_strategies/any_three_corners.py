# src/traversal_strategies/any_three_corners.py
"""[微观] 策略6：随机挑3个角查"""
import random
from typing import List, Tuple
from .base_strategy import BaseStrategy
from ._utils import generate_corner_points


class AnyThreeCornersStrategy(BaseStrategy):
    """随机三角采样：从区块四个角中随机选取3个进行查询"""

    def __init__(self, seed: int = None, **kwargs):
        super().__init__(**kwargs)
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "any_three_corners"

    def get_points(self, chunk_x: int, chunk_z: int, temperature: float = None) -> List[Tuple[int, int]]:
        return self._rng.sample(generate_corner_points(chunk_x, chunk_z), 3)