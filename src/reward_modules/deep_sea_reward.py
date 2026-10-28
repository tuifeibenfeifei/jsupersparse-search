
"""
deep_sea_reward.py
[Reward] Deep-sea discovery reward module
Core responsibility: give positive reward when a new deep-sea point is found (distance to all existing deep-sea points ≥ distance threshold)
Controlled on/off via boolean switch in config; all numerical parameters are injected from YAML
Strictly prohibited from hardcoding any reward values in this file
"""

from typing import Optional


class DeepSeaReward:
    """Deep-sea discovery reward calculator

    When the AI queries a new deep-sea point and its Euclidean distance to all existing deep-sea points is ≥ the distance threshold,
    a positive reward is given. This reward signal drives the AI to actively seek deep-sea points in "new regions" rather than repeatedly searching in already-discovered areas.

    Ablation experiment control:
        config["deep_sea_enabled"] = False  → disable this reward (Stage3 reward_no_deep scenario)
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Args:
            config: Configuration dict, supporting the following keys:
                - enabled (bool): whether to enable deep-sea discovery reward, default True
                - reward (float): reward value upon discovering a valid deep-sea point, default 10.0
                - distance_threshold (float): minimum distance threshold between valid deep-sea points (cells), default 416.0
        """
        config = config or {}
        self._enabled = config.get("enabled", True)
        self._reward = float(config.get("reward", 10.0))
        self._distance_threshold = float(config.get("distance_threshold", 416.0))

    def calculate(self, is_new_deep_sea: bool, distance_to_nearest: Optional[float] = None) -> float:
        """
        Calculate deep-sea discovery reward

        Args:
            is_new_deep_sea: whether this is a valid new deep-sea point (passed the Euclidean distance check in database._is_valid_point)
            distance_to_nearest: Euclidean distance to the nearest existing deep-sea point (for logging/debugging only, does not affect computation)

        Returns:
            Reward value (float). Returns 0.0 when disabled or when not a new deep-sea point
        """
        if not self._enabled:
            return 0.0
        if is_new_deep_sea:
            return self._reward
        return 0.0

    @property
    def name(self) -> str:
        return "deep_sea_reward"

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def reward(self) -> float:
        return self._reward

    @property
    def distance_threshold(self) -> float:
        return self._distance_threshold