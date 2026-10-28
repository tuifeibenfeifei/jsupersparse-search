
"""
curiosity_reward.py
[Reward] Curiosity reward module
Core responsibility: give positive reward when the AI reaches a never-visited state (chunk coordinates + temperature bin)
Controlled on/off via boolean switch in config; weight is adjustable via YAML
Strictly prohibited from hardcoding any reward values in this file
"""

from typing import Optional


class CuriosityReward:
    """Curiosity reward calculator

    State is defined by state_manager as a tuple of (chunk_x, chunk_z, temp_bin).
    When the AI first reaches a discrete state, a positive reward is given to incentivize the AI to explore unknown regions,
    avoiding loops in already-explored areas.

    Ablation experiment control:
        config["curiosity_enabled"] = False  → disable curiosity reward (Stage3 reward_no_curiosity scenario)
        config["curiosity_weight"] adjusts the weight (Stage4 hyperparameter sweep scenario)
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Args:
            config: Configuration dict, supporting the following keys:
                - enabled (bool): whether to enable curiosity reward, default True
                - reward (float): base reward value upon reaching a new state, default 3.0
                - weight (float): curiosity reward weight (adjustable), default 1.0
                    total curiosity reward = reward × weight (when is_new_state is True)
        """
        config = config or {}
        self._enabled = config.get("enabled", True)
        self._reward = float(config.get("reward", 3.0))
        self._weight = float(config.get("weight", 1.0))

    def calculate(self, is_new_state: bool) -> float:
        """
        Calculate curiosity reward

        Args:
            is_new_state: whether the current state is a first-time visit
                         Determined by the return value of state_manager.is_new_state()

        Returns:
            Curiosity reward value (float). Returns 0.0 when disabled or when not a new state
        """
        if not self._enabled:
            return 0.0
        if is_new_state:
            return self._reward * self._weight
        return 0.0

    @property
    def name(self) -> str:
        return "curiosity_reward"

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def reward(self) -> float:
        return self._reward

    @property
    def weight(self) -> float:
        return self._weight