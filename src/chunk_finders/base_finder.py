
"""
base_finder.py
[Macro] Abstract base class for chunk traversal strategies
Core responsibility: define a unified interface for "getting the next chunk to query"
All macro strategies must inherit this class, ensuring main_algorithm.py can dynamically instantiate them by string name
No micro point-selection logic or reward computation is allowed in this file

[Fix Rationale] (<=100 chars)
Added reset() hook (default no-op): macro strategies hold internal cursors (direction index, step size, current coordinates),
if not reset before each episode, the cursor accumulates across episodes when the same Explorer runs multiple times,
and from the 2nd run onward it no longer scans from the origin, causing cross-episode contamination. Subclasses override as needed.
"""

from abc import ABC, abstractmethod
from typing import Tuple


class BaseFinder(ABC):
    """
    Macro chunk traversal strategy base class
    WARNING: do not instantiate directly; must be subclassed

    Subclass implementation constraints:
        1. Must implement get_next_chunk(current_x, current_z)
        2. Must explicitly declare name via @property, strictly matching the YAML config
        3. __init__ must accept a config parameter to be compatible with main_algorithm.py's instantiation
        4. If holding internal cursor state, must override reset() to support stateless reset (Req. Sec. 1.3)
    """

    def __init__(self, config=None, **kwargs):
        """
        Strategy constructor (all subclasses must be compatible with this signature)
        Args:
            config: configuration dict (passed by main_algorithm.py's self.config)
            **kwargs: strategy-specific hyperparameters
        """
        self.config = config or {}

    @abstractmethod
    def get_next_chunk(self, current_x: int, current_z: int) -> Tuple[int, int]:
        """
        Given the current chunk coordinates, return the next chunk coordinate to query
        Args:
            current_x: current chunk X coordinate
            current_z: current chunk Z coordinate
        Returns:
            (chunk_x, chunk_z) tuple, chunk-level coordinates (not world coordinates)
        """
        pass

    def reset(self) -> None:
        """
        Reset internal traversal state so the next get_next_chunk() starts from the beginning.
        Default no-op: stateless strategies do not need to override.
        Called by main_algorithm.py at the start of each episode.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name identifier, must strictly match the chunk_finder field in YAML config"""
        pass
