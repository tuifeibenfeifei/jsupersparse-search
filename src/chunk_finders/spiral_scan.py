
"""
spiral_scan.py
[Overview] spiral scan: expand outward layer by layer from the start point, clockwise spiral

[Rationale for the fix] (<=100 chars)
Only adds reset(): the spiral's current coordinates, direction, and leg count are all instance state; without resetting across episodes,
from the 2nd run onward the scan keeps expanding from the previous endpoint, violating Req. Sec. 1, item 3 "stateless reset".
The spiral-progression logic remains unchanged from the original design.
"""

from typing import Tuple
from .base_finder import BaseFinder


class SpiralScanFinder(BaseFinder):
    """
    Spiral scan: starting from the initial chunk, expand outward layer by layer in a clockwise spiral
    suited to exploration scenarios requiring uniform coverage with no prior information
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self._x, self._z = 0, 0
        self._dx, self._dz = 1, 0
        self._steps_taken = 0
        self._steps_in_leg = 1
        self._leg = 0

    @property
    def name(self) -> str:
        return "spiral_scan"

    def reset(self) -> None:
        """reset the spiral state so every episode re-spirals from the origin (0,0)"""
        self._x, self._z = 0, 0
        self._dx, self._dz = 1, 0
        self._steps_taken = 0
        self._steps_in_leg = 1
        self._leg = 0

    def get_next_chunk(self, current_x: int, current_z: int) -> Tuple[int, int]:
        result = (self._x, self._z)
        # move to the next position
        self._x += self._dx
        self._z += self._dz
        self._steps_taken += 1
        # turn after finishing the current leg (right turn by 90 degrees)
        if self._steps_taken == self._steps_in_leg:
            self._steps_taken = 0
            self._dx, self._dz = -self._dz, self._dx
            self._leg += 1
            if self._leg % 2 == 0:
                self._steps_in_leg += 1
        return result
