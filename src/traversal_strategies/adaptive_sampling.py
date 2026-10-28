# src/traversal_strategies/adaptive_sampling.py
"""[Micro] Strategy 8: dynamically adjust query-point positions based on temperature feedback"""
from typing import List, Tuple, Optional
from .base_strategy import BaseStrategy
from ._utils import generate_grid_points, generate_corner_points


class AdaptiveSamplingStrategy(BaseStrategy):
    """
    Adaptive sampling: dynamically adjust the in-chunk sampling strategy based on accumulated temperature data

    When temperature is enabled:
        if the current chunk temperature is close to the mean temperature of discovered deep sea -> use dense 3x3 grid sampling
        otherwise -> query only the four corners + center (lightweight sampling)

    When temperature is disabled:
        degrade to 5-point sampling: four corners + center
    """

    def __init__(self, reference_temps: Optional[List[float]] = None, **kwargs):
        super().__init__(**kwargs)
        self._reference_temps = reference_temps or []

    @property
    def name(self) -> str:
        return "adaptive_sampling"

    def set_reference_temps(self, temps: List[float]) -> None:
        """inject reference temperatures externally (called by main_algorithm.py when a deep-sea point is found)"""
        self._reference_temps = list(temps)

    def get_points(self, chunk_x: int, chunk_z: int, temperature: float = None) -> List[Tuple[int, int]]:
        """
        Decide the sampling strategy based on temperature feedback
        Args:
            chunk_x/chunk_z: chunk coordinates
            temperature: current chunk temperature (injected by main_algorithm.py)
        """
        # no reference temperature or no current temperature -> use the default strategy
        if not self._reference_temps or temperature is None:
            return self._default_points(chunk_x, chunk_z)

        import math
        ref_mean = sum(self._reference_temps) / len(self._reference_temps)

        # compute the standard deviation of reference temperatures
        if len(self._reference_temps) >= 3:
            ref_std = (sum((t - ref_mean) ** 2 for t in self._reference_temps) / len(self._reference_temps)) ** 0.5
            threshold = ref_std * 0.5 if ref_std > 0 else 0.3
        else:
            threshold = 0.3

        # temperature close to the reference mean -> dense sampling
        if abs(temperature - ref_mean) < threshold:
            return generate_grid_points(chunk_x, chunk_z, rows=3, cols=3)

        return self._default_points(chunk_x, chunk_z)

    def _default_points(self, chunk_x: int, chunk_z: int) -> List[Tuple[int, int]]:
        """5-point sampling: four corners + center"""
        cx = chunk_x * 16 + 8
        cz = chunk_z * 16 + 8
        return [(cx, cz)] + generate_corner_points(chunk_x, chunk_z)