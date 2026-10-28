"""
base_strategy.py
[Micro] Abstract base class for within-chunk point-selection strategies
Core responsibility: define a unified interface for "getting the list of coordinates to query"
All 9 micro strategies must inherit this class, ensuring main_algorithm.py can dynamically instantiate them by string name
No search logic or reward computation is allowed in this file

[Fix Rationale] (<=100 chars)
Added optional temperature parameter to the abstract method signature: all 9 subclass implementations already have it; the original base class lacked it, causing
signature mismatch, and main_algorithm could not legally inject black-box temperature feedback into adaptive_sampling,
making Req. Sec. 9 "dynamically adjust query point position based on temperature feedback" permanently ineffective. Only adds the optional parameter; semantics unchanged.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional


class BaseStrategy(ABC):
    """
    Micro point-selection strategy base class
    WARNING: do not instantiate directly; must be subclassed

    Subclass implementation constraints:
        1. Must implement get_points(), returning a coordinate list sorted by exploration priority descending
        2. Must explicitly declare name via @property, strictly matching the string in YAML config
        3. __init__ must accept **kwargs for unified registry instantiation
        4. get_points() must accept an optional temperature parameter:
           Pure geometric strategies may ignore it; adaptive_sampling depends on it for dynamic adjustment
    """

    def __init__(self, **kwargs):
        """
        Strategy constructor (all subclasses must be compatible with this signature)
        Args:
            **kwargs: strategy-specific hyperparameters (e.g., random_two_points seed, etc.)
        """
        pass

    @abstractmethod
    def get_points(self, chunk_x: int, chunk_z: int,
                   temperature: Optional[float] = None) -> List[Tuple[int, int]]:
        """
        Given the target chunk coordinates, return the list of world coordinates within the chunk to query
        Args:
            chunk_x: X coordinate of the target chunk (chunk-level, not world coordinate)
            chunk_z: Z coordinate of the target chunk (chunk-level, not world coordinate)
            temperature: temperature feedback from the most recent black-box query (may be None)
                         Used only by adaptive_sampling; other pure geometric strategies should ignore it.
                         WARNING: do not design prior rules like "a certain temperature is better" based on this (Req. Sec. 1)
        Returns:
            List of world coordinates to query [(x1, z1), (x2, z2), ...]
            WARNING: must be sorted by exploration priority descending (first coordinate is the best candidate)
            Coordinates must be within the target chunk range (chunk_x*16 ~ chunk_x*16+15)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Strategy name identifier, must strictly match the micro_strategy field in YAML config
        Example: Grid3x3Strategy's name should return "grid_3x3"
        """
        pass
