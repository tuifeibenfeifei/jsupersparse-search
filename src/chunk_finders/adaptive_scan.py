
"""
adaptive_scan.py
[Overview] Adaptive scan: automatically switch search granularity based on deep-sea point density
Dense directions use small step size for fine search; sparse directions use large step size for fast sweep

[Fix Rationale] (<=100 chars)
1. Complete density adaptation: the original implementation injected small_step/_deep_sea_chunks but never used them,
   get_next_chunk always returned large_step, contradicting the design intent declared in this file,
   and left Req. Sec. 1.8 "adaptive search granularity" unfulfilled. Now deep-sea point density is counted across 8 directions,
   using small step size when the current direction hits the dense set, otherwise large step size.
2. Add reset(): current coordinates/direction/step count are instance state and must be zeroed across episodes.
3. Fix infinite loop: the original implementation returned the cursor exactly to the ring start after one ring (32 chunks); when density no longer changes, the next ring
   repeats the same visited chunks -> the main loop dedup branch loops with zero queries. Changed to advance by a per-ring base radius
   (adjacent rings do not overlap) and apply boundary wraparound, ensuring the path never periodically repeats and never goes out of bounds.
"""

from typing import Tuple, List, Set, Dict
from collections import defaultdict
from .base_finder import BaseFinder


class AdaptiveScanFinder(BaseFinder):
    """
    Adaptive scan: dynamically adjust search strategy based on deep-sea point density
    - Directions with discovered deep-sea points -> small step size for fine search
    - Directions without discovered deep-sea points -> large step size for fast sweep

    Density data source: called by main_algorithm.py after each new deep-sea point is confirmed,
    update_deep_sea_density() to inject the chunk coordinate list.
    Before any data is injected, large step size fast sweep is used by default.
    """

    # 8 directions: E, SE, S, SW, W, NW, N, NE
    _DIRECTIONS: List[Tuple[int, int]] = [
        (1, 0), (1, -1), (0, -1), (-1, -1),
        (-1, 0), (-1, 1), (0, 1), (1, 1),
    ]

    def __init__(
        self,
        config=None,
        small_step: int = 2,
        large_step: int = 8,
        density_threshold_ratio: float = 0.5,
        **kwargs
    ):
        """
        Args:
            config: configuration dict (passed by main_algorithm.py)
            small_step: fine search step size (chunks) used for dense directions
            large_step: fast sweep step size (chunks) used for sparse directions
            density_threshold_ratio: ratio threshold to determine "dense direction",
                a direction whose deep-sea point count >= max direction count * this ratio is considered dense
        """
        super().__init__(config, **kwargs)
        self._small_step = int(small_step)
        self._large_step = int(large_step)
        self._density_ratio = float(density_threshold_ratio)
        # Chunk-level search radius = search_radius (cells) / 16; used for boundary wraparound (fix rationale item 3)
        self._chunk_radius = max(1, self.config.get("search_radius", 10000) // 16)

        self._deep_sea_chunks: Set[Tuple[int, int]] = set()
        self._direction_density: Dict[int, int] = defaultdict(int)
        self._dense_directions: Set[int] = set()

        self.reset()

    @property
    def name(self) -> str:
        return "adaptive_scan"

    def reset(self) -> None:
        """Reset traversal cursor; density stats are managed independently by update_deep_sea_density()"""
        self._x, self._z = 0, 0
        self._direction = 0
        self._step = 0
        self._ring = 0
        self._first = True

    def update_deep_sea_density(self, chunks: List[Tuple[int, int]]) -> None:
        """
        Externally inject chunk coordinates of discovered deep-sea points (called by main_algorithm.py)
        Recompute deep-sea point density for each of the 8 directions and mark dense directions.
        Args:
            chunks: list of chunk coordinates with discovered deep-sea points [(cx1, cz1), ...]
        """
        self._deep_sea_chunks = set(chunks)

        density: Dict[int, int] = defaultdict(int)
        for cx, cz in self._deep_sea_chunks:
            if cx == 0 and cz == 0:
                continue  # origin has no direction, excluded from directional density stats
            # Assign to the unique direction with the largest dot product, so a chunk is not double-counted across directions
            best_idx = max(
                range(len(self._DIRECTIONS)),
                key=lambda i: cx * self._DIRECTIONS[i][0] + cz * self._DIRECTIONS[i][1],
            )
            density[best_idx] += 1

        self._direction_density = density
        self._dense_directions = self._compute_dense_directions(density)

    def _compute_dense_directions(self, density: Dict[int, int]) -> Set[int]:
        """Determine which directions are dense according to density_threshold_ratio"""
        if not density:
            return set()
        max_count = max(density.values())
        if max_count <= 0:
            return set()
        cutoff = max_count * self._density_ratio
        return {idx for idx, count in density.items() if count >= cutoff}

    def get_next_chunk(self, current_x: int, current_z: int) -> Tuple[int, int]:
        # First call returns the origin (0,0) (consistent with original design, ensures the origin chunk is scanned)
        if self._first:
            self._first = False
            self._advance_cursor()
            return (0, 0)

        # Req. Sec. 1.8: dense directions use small step size for fine search; sparse directions use large step size for fast sweep
        if self._direction in self._dense_directions:
            step_size = self._small_step
        else:
            step_size = self._large_step

        dx, dz = self._DIRECTIONS[self._direction]

        # Per-ring base radius: each ring expands outward by 4*large_step along 8 directions, so adjacent rings never overlap.
        # Fix rationale item 3: the original implementation returned the cursor exactly to the ring start after one ring (32 chunks); when density no longer changes,
        # the next ring repeats the same visited chunks -> the main loop dedup branch loops with zero queries.
        base = self._ring * 4 * self._large_step
        radius = base + (self._step + 1) * step_size
        nx, nz = dx * radius, dz * radius

        # Boundary wraparound: wrap to the opposite side when chunk coordinates exceed +/-chunk_radius, ensuring out-of-bounds chunks are never returned
        # (the main loop out-of-bounds branch consumes no queries and would likewise cause an infinite loop)
        lim = self._chunk_radius
        if abs(nx) > lim or abs(nz) > lim:
            span = 2 * lim
            nx = (nx + lim) % span - lim
            nz = (nz + lim) % span - lim

        self._advance_cursor()
        return (nx, nz)

    def _advance_cursor(self) -> None:
        """Advance to the next chunk: turn after 4 steps per direction; enter next ring after all 8 directions"""
        self._step += 1
        if self._step >= 4:
            self._step = 0
            self._direction = (self._direction + 1) % 8
            if self._direction == 0:
                self._ring += 1
