
"""
ray_scan.py
[Overview] ray scan: straight-line jumps from the center toward 8 directions
Corresponds to Req. Sec. 9: ray_scan (8-direction ray scan; core logic identical to zhongxin.py)

[Rationale for the fix] (<=100 chars)
Only adds reset(): in the original implementation the direction index and step size are instance state; without resetting across episodes, from the 2nd run onward the
scan trajectory keeps expanding outward from where it ended last time, violating Req. Sec. 1, item 3 "stateless reset". The expansion logic,
direction table, and step-size progression all remain unchanged from the original design.
"""

from typing import Tuple, List
from .base_finder import BaseFinder


class RayScanFinder(BaseFinder):
    """
    Ray scan: starting from the initial chunk, rays are emitted in rotation toward the 8 directions
    after each complete round of the 8-direction scan, the step size increases by 4 chunks, forming radial outward coverage

    Rationale for the 4-chunk step increment (Req. Sec. 7):
        short step 1-2 chunks, long step 4-8 chunks; the initial jump of 4 chunks equals the structural radius of an ocean monument.
    """

    # the 8 directions: east, southeast, south, southwest, west, northwest, north, northeast
    _DIRECTIONS: List[Tuple[int, int]] = [
        (1, 0), (1, -1), (0, -1), (-1, -1),
        (-1, 0), (-1, 1), (0, 1), (1, 1),
    ]

    # step increment (in chunks) after each complete 8-direction round
    _STEP_INCREMENT = 4

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self._dir_idx = 0
        self._step = 0

    @property
    def name(self) -> str:
        return "ray_scan"

    def reset(self) -> None:
        """reset the ray state so every episode re-radiates from the origin (0,0)"""
        self._dir_idx = 0
        self._step = 0

    def get_next_chunk(self, current_x: int, current_z: int) -> Tuple[int, int]:
        if self._step == 0 and self._dir_idx == 0:
            # first call: return the start point (0, 0)
            self._step = 1
            return (0, 0)

        dx, dz = self._DIRECTIONS[self._dir_idx]
        nx = dx * self._step
        nz = dz * self._step
        self._dir_idx = (self._dir_idx + 1) % 8
        if self._dir_idx == 0:
            # after each complete 8-direction round, increase step size by 4 chunks (Req. Sec. 7: long step 4-8 chunks)
            self._step += self._STEP_INCREMENT
        return (nx, nz)
