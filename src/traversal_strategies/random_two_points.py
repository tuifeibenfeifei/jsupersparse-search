# src/traversal_strategies/random_two_points.py
"""[微观] 策略3：随机抽2个点查"""
import random
from typing import List, Tuple
from .base_strategy import BaseStrategy


class RandomTwoPointsStrategy(BaseStrategy):
    """区块内随机2点采样：每次查询随机选取2个不同坐标点"""

    def __init__(self, seed: int = None, **kwargs):
        super().__init__(**kwargs)
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "random_two_points"

    def get_points(self, chunk_x: int, chunk_z: int, temperature: float = None) -> List[Tuple[int, int]]:
        x0, z0 = chunk_x * 16, chunk_z * 16
        x1, z1 = (chunk_x + 1) * 16 - 1, (chunk_z + 1) * 16 - 1
        all_coords = [(x, z) for x in range(x0, x1 + 1) for z in range(z0, z1 + 1)]
        return self._rng.sample(all_coords, 2)