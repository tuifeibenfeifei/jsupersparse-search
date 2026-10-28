
"""
penalty_reward.py
[Reward] Penalty reward module
Core responsibility: apply negative reward (penalty) for out-of-bounds actions and repeated visits
Controlled on/off via boolean switch in config; penalty values are injected from YAML
Strictly prohibited from hardcoding any penalty values in this file
"""

from typing import Optional


class PenaltyReward:
    """Penalty reward calculator

    Applies two types of penalties:
        1. Out-of-bounds penalty: triggered when the AI's next-step coordinates exceed the search area (square region of radius 10000 cells)
        2. Repeated visit penalty: triggered when the AI repeatedly queries already-visited chunk coordinates

    Penalty signals drive the AI to avoid invalid moves and stay within the valid search area.

    Ablation experiment control:
        config["penalty_enabled"] = False  → disable all penalties (Stage3 reward_no_penalty scenario)
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Args:
            config: Configuration dict, supporting the following keys:
                - enabled (bool): whether to enable penalties, default True
                - out_of_bounds (float): out-of-bounds penalty value (negative), default -10.0
                - repeated_visit (float): repeated visit penalty value (negative), default -5.0
        """
        config = config or {}
        self._enabled = config.get("enabled", True)
        self._out_of_bounds = float(config.get("out_of_bounds", -10.0))
        self._repeated_visit = float(config.get("repeated_visit", -5.0))

    def calculate(self, is_out_of_bounds: bool = False, is_repeated_visit: bool = False) -> float:
        """
        Calculate penalty value (stackable)

        Args:
            is_out_of_bounds: whether the search area is exceeded (square region of radius 10000 cells)
            is_repeated_visit: whether already-queried chunk coordinates are revisited

        Returns:
            Total penalty value (float). Returns 0.0 when disabled
            If both penalties trigger simultaneously, total penalty = out_of_bounds + repeated_visit
        """
        if not self._enabled:
            return 0.0
        total = 0.0
        if is_out_of_bounds:
            total += self._out_of_bounds
        if is_repeated_visit:
            total += self._repeated_visit
        return total

    @property
    def name(self) -> str:
        return "penalty_reward"

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def out_of_bounds(self) -> float:
        return self._out_of_bounds

    @property
    def repeated_visit(self) -> float:
        return self._repeated_visit