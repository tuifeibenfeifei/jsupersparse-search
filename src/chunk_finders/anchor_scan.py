"""
anchor_scan.py
[Overview] Anchor relocation scan: after discovering a deep-sea point, use it as a new anchor and expand outward via BFS.

Corresponding requirements:
  - Req. Sec. 1.4: after discovering a deep-sea point, the area within threshold around it is considered the same generation grid region,
                 subsequent search should continue outward from that point.
  - Req. Sec. 2: if all chunks queried per the point-selection strategy are non-deep-sea points, jump some distance away and continue searching.
Corresponding paper novelty 3: master-slave coordination - BFS ensures coverage completeness + RL guides direction.
  This finder handles BFS-style systematic coverage for Stage1/2 (advancing along the edge of the known deep-sea region),
  Stage3 is taken over by Q-learning + temperature-directed bias for direction decisions; the two form master-slave coordination.

Why it works (empirical basis):
  Deep-sea field spatial autocorrelation at lag=416 cells (threshold) still reaches 0.23, strong clustering;
  hence expanding outward from known deep-sea anchors yields a much higher hit rate than blindly radiating from the origin.
  This breaks ray_scan's budget bottleneck - the latter still radiates from the origin after finding deep-sea points,
  wasting many queries on already-probed areas; even exhausting the budget fails to reach 40 points.
"""

from collections import deque
from typing import Tuple, List, Set

from .base_finder import BaseFinder


class AnchorScanFinder(BaseFinder):
    """
    Anchor relocation scan (BFS coverage + ray frontier expansion)

    Working mechanism:
        1. BFS frontier queue: whenever a new deep-sea anchor is found, enqueue exploration starts at 8 directions, distance ~ spacing threshold,
           so search always advances systematically along the edge of the known deep-sea region.
        2. Fallback ray scan: when frontier is empty, use layered rays (same as ray_scan) to expand frontier,
           to find new deep-sea anchors.
        3. Anchor dedup: update_deep_sea_density receives the full deep-sea point set each time and derives frontier only for new anchors.
    """

    # 8 directions, order consistent with main_algorithm._DIRECTION_DELTAS / RayScanFinder._DIRECTIONS
    _DIRECTIONS: List[Tuple[int, int]] = [
        (1, 0), (1, -1), (0, -1), (-1, -1),
        (-1, 0), (-1, 1), (0, 1), (1, 1),
    ]

    # Default step sizes for ray fallback, only used as fallback when config lacks action_steps
    _DEFAULT_STEPS: List[int] = [1, 2, 4, 8]

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        # Spacing threshold (cells) -> chunks; derived jump distance is slightly larger than threshold to ensure jumping out of the exclusion zone (Req. Sec. 1.4)
        threshold_blocks = self.config.get("deep_sea_distance_threshold", 416)
        self._jump_chunks = max(int(threshold_blocks) // 16 + 1, 1)

        # Ray fallback step sizes are injected via config; hardcoding is forbidden (Req. Sec. 10)
        steps = self.config.get("action_steps") or self._DEFAULT_STEPS
        self._ray_steps: List[int] = [int(s) for s in steps]

        # Frontier capacity limit: prevent unbounded queue growth in extreme cases (deep-sea point cap 40 x 8 directions = 320)
        self._max_frontier = int(self.config.get("anchor_max_frontier", 4096))

        self.reset()

    @property
    def name(self) -> str:
        return "anchor_scan"

    def reset(self) -> None:
        """Reset all internal state, ensuring each episode restarts from the origin (Req. Sec. 1.3)"""
        self._frontier: deque = deque()
        self._anchors: Set[Tuple[int, int]] = set()
        # fallback ray scan cursor
        self._ray_dir = 0
        self._ray_step_idx = 0
        self._ray_layer = 1
        self._emitted_origin = False

    def update_deep_sea_density(self, chunks: List[Tuple[int, int]]) -> None:
        """
        Externally inject chunk coordinates of discovered deep-sea points (called by main_algorithm._sync_finder_density).
        For each [new] anchor, derive BFS exploration starts at 8 directions, distance ~ threshold.
        Args:
            chunks: full list of deep-sea point chunk coordinates [(cx, cz), ...]
        """
        for anchor in chunks:
            ax, az = anchor
            if (ax, az) in self._anchors:
                continue
            self._anchors.add((ax, az))
            for dx, dz in self._DIRECTIONS:
                if len(self._frontier) >= self._max_frontier:
                    break
                self._frontier.append(
                    (ax + dx * self._jump_chunks, az + dz * self._jump_chunks)
                )

    def _next_ray_chunk(self) -> Tuple[int, int]:
        """Fallback when frontier is empty: layered ray scan to expand frontier (same logic as ray_scan)"""
        if not self._emitted_origin:
            self._emitted_origin = True
            return (0, 0)

        dx, dz = self._DIRECTIONS[self._ray_dir]
        step = self._ray_steps[self._ray_step_idx % len(self._ray_steps)] * self._ray_layer
        nx, nz = dx * step, dz * step

        # Advance: step size (inner) -> direction (middle) -> layer (outer)
        self._ray_step_idx += 1
        if self._ray_step_idx % len(self._ray_steps) == 0:
            self._ray_dir += 1
            if self._ray_dir >= len(self._DIRECTIONS):
                self._ray_dir = 0
                self._ray_layer += 1
        return (nx, nz)

    def get_next_chunk(self, current_x: int, current_z: int) -> Tuple[int, int]:
        """
        Return the next chunk to query: prefer BFS frontier (systematically advancing along the known deep-sea region edge),
        fall back to ray scan frontier expansion when frontier is empty.
        Duplicate/visited chunks are deduped by main_algorithm's _visited_chunks; not checked here.
        """
        if self._frontier:
            return self._frontier.popleft()
        return self._next_ray_chunk()

    @property
    def anchor_count(self) -> int:
        """Current number of registered deep-sea anchors (for debugging and convergence analysis)"""
        return len(self._anchors)

    @property
    def frontier_size(self) -> int:
        """Current BFS pending exploration queue length"""
        return len(self._frontier)
