
"""
grid_scan.py
[Macro] Spaced grid scan: grid aligned to the "deep-sea point spacing threshold", scanned ring by ring from inside out

Design rationale (Stage1 empirical data-driven, not manual prior):
    The ranking of Stage1's nine micro strategies is perfectly monotonically inversely correlated with "queries per chunk"
    (center_point 1 query/chunk -> 32.84 points; grid_3x3 9 queries/chunk -> 29.30 points),
    proving the bottleneck is "coverage breadth" rather than "in-chunk sampling density".

    Further accounting for ray_scan's waste source:
        9138 queries x deep-sea hit rate ~27% ~= 2400 hits, but only 32.84 points pass the spacing check,
        i.e. ~98.6% of hits are discarded because they fall within existing points' exclusion zone (threshold 416 cells).
        The root cause is ray_scan radiating from the origin with ring spacing 64 cells far below the threshold 416 cells,
        so hits across consecutive rings all crowd into the same exclusion zone.

    This strategy sets grid spacing = spacing threshold (416 cells = 26 chunks), so the Euclidean distance between adjacent grid points equals the threshold.
    Per database._is_valid_point's criterion (reject only when dx^2+dz^2 < threshold^2),
    a distance exactly equal to the threshold is valid; thus every hit is necessarily a legal new point, and exclusion waste approaches zero.

    Coverage order is "ring by ring from inside out" (Chebyshev rings), ensuring near-priority and matching Req. Sec. 2's
    progressive diffusion semantics; grid point spacing within a ring is constant, with no overlap or gaps.
"""

from typing import Tuple, List
from .base_finder import BaseFinder


class GridScanFinder(BaseFinder):
    """
    Spaced grid scan: centered at (0,0), grid formed by integer multiples of grid_spacing_chunks,
    returning grid points ring by ring from inside out (Chebyshev distance).

    Config:
        grid_spacing_chunks (int): grid spacing (chunks). When omitted, derived from
            deep_sea_distance_threshold // 16, i.e. the number of chunks corresponding to the threshold.
        deep_sea_distance_threshold (float): minimum deep-sea point spacing (cells), used to derive the default spacing.
    """

    _BLOCKS_PER_CHUNK = 16

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        threshold = self.config.get("deep_sea_distance_threshold", 416)
        derived = int(threshold) // self._BLOCKS_PER_CHUNK
        spacing = self.config.get("grid_spacing_chunks", derived)
        self._spacing = max(1, int(spacing))
        self._ring = 0
        self._ring_points: List[Tuple[int, int]] = [(0, 0)]
        self._ring_idx = 0

    @property
    def name(self) -> str:
        return "grid_scan"

    @property
    def spacing_chunks(self) -> int:
        """Current grid spacing (chunks), for the scheduler script to record and verify"""
        return self._spacing

    def reset(self) -> None:
        """Reset to the origin ring, ensuring each episode rescans from (0,0) inside out"""
        self._ring = 0
        self._ring_points = [(0, 0)]
        self._ring_idx = 0

    @staticmethod
    def _build_ring(k: int) -> List[Tuple[int, int]]:
        """
        Build all integer grid points on the ring at Chebyshev distance exactly k, total 8k (k>=1).
        Top/bottom edges each have 2k+1 points, left/right edges each have 2k-1 points, total 8k with no duplicates.
        """
        if k <= 0:
            return [(0, 0)]
        pts: List[Tuple[int, int]] = []
        for i in range(-k, k + 1):
            pts.append((i, -k))
            pts.append((i, k))
        for j in range(-k + 1, k):
            pts.append((-k, j))
            pts.append((k, j))
        return pts

    def get_next_chunk(self, current_x: int, current_z: int) -> Tuple[int, int]:
        if self._ring_idx >= len(self._ring_points):
            self._ring += 1
            self._ring_points = self._build_ring(self._ring)
            self._ring_idx = 0
        gi, gj = self._ring_points[self._ring_idx]
        self._ring_idx += 1
        return (gi * self._spacing, gj * self._spacing)
