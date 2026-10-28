"""
database.py
[Data] In-memory deep-sea point database: stores confirmed deep-sea point coordinates and temperature
Core responsibility: Euclidean distance deduplication check, valid point set management
Strictly prohibited from containing any search logic, reward computation, or state encoding
"""

import math
from typing import List, Tuple


class DeepSeaDatabase:
    """
    Deep-sea point record manager
    ⚠️ Not thread-safe; for serial use within a single episode only
    Must call clear() before each experiment to reset and ensure no historical experience remains
    """

    def __init__(self, distance_threshold: float = 416.0):
        """
        Args:
            distance_threshold: Minimum Euclidean distance between valid deep-sea points (default 416 cells)
        """
        if distance_threshold <= 0:
            raise ValueError(f"Distance threshold must be positive, received: {distance_threshold}")
        self._threshold = float(distance_threshold)
        self._threshold_sq = self._threshold * self._threshold  # Precompute squared
        self._points: List[Tuple[int, int, float]] = []

    def add_point(self, x: int, z: int, temp: float) -> bool:
        """
        Attempt to add a new deep-sea point
        Returns:
            True=successfully added (valid new point)
            False=filtered by deduplication (too close)
        """
        if not isinstance(x, int) or not isinstance(z, int):
            raise TypeError(f"Coordinates must be int type, received x={type(x).__name__}, z={type(z).__name__}")
        if temp is not None and (not isinstance(temp, (int, float)) or math.isnan(temp)):
            raise TypeError(f"Temperature must be a valid number, received temp={temp}")

        if self._is_valid_point(x, z):
            self._points.append((x, z, float(temp) if temp is not None else 0.0))
            return True
        return False

    def _is_valid_point(self, x: int, z: int) -> bool:
        """Check whether the Euclidean distance from a new point to all existing points ≥ threshold"""
        for px, pz, _ in self._points:
            dx = x - px
            dz = z - pz
            if dx * dx + dz * dz < self._threshold_sq:
                return False
        return True

    def get_nearest_distance(self, x: int, z: int) -> float:
        """
        Return the minimum Euclidean distance from candidate point (x, z) to all existing deep-sea points
        Returns 0.0 when the database is empty (indicating no reference points yet)
        Used by main_algorithm.py to pass to the reward module's distance_to_nearest parameter
        """
        if not self._points:
            return 0.0
        min_dist_sq = min((x - px) ** 2 + (z - pz) ** 2 for px, pz, _ in self._points)
        return math.sqrt(min_dist_sq)

    def get_all_coordinates(self) -> List[Tuple[int, int]]:
        """Return a list of (x, z) coordinates of all valid deep-sea points"""
        return [(p[0], p[1]) for p in self._points]

    def get_all_points(self) -> List[Tuple[int, int, float]]:
        """Return a copy of the complete records (x, z, temp) of all valid deep-sea points"""
        return list(self._points)

    def clear(self) -> None:
        """Clear the database and reset to empty state"""
        self._points.clear()

    def __len__(self) -> int:
        return len(self._points)

    @property
    def distance_threshold(self) -> float:
        return self._threshold
