
"""
full_reward_baseline.py
[Reward] Full-reward aggregator (control group)
Core responsibility: aggregate the three rewards (deep-sea discovery, curiosity, penalty) and weight-sum them according to their on/off states
Called by main_algorithm.py in the Stage3 reward-driven phase; returns the scalar reward used for the Q-learning update

Design notes:
    This file implements no reward-computation logic; it only aggregates the three submodules.
    Each submodule is independently toggled on/off by a boolean switch in config,
    supporting the Stage3 ablation scenario of "turning off one item in isolation".

[Fix rationale] (<=100 words)
The config keys read all match the actual YAML keys under configs/ (Req. Sec. 10: config-file driven,
no hardcoding). The original implementation read self-invented keys deep_sea_enabled / deep_sea_reward / penalty_repeated_visit
that did not exist in the YAML, so all 7 switches and values silently fell back to defaults -- the Stage3
ablation config had no effect. It now reads deep_sea_reward_enabled / deep_sea_reward_value /
repeat_visit_penalty, etc., verbatim-identical to temp_on.yaml. The reward remains the sum of 3 terms; the structure is unchanged.
"""

from typing import Optional
from .deep_sea_reward import DeepSeaReward
from .curiosity_reward import CuriosityReward
from .penalty_reward import PenaltyReward


class FullRewardBaseline:
    """
    Full-reward aggregator

    Reads the boolean switches in config to decide whether to enable the three rewards (deep-sea discovery, curiosity, penalty),
    and returns the weight-summed total reward.

    Req. Sec. 5: the total reward is the sum of the following items; temperature is only a state feature and does not directly enter the reward.

    Example ablation configs (Stage3):
        reward_all_on:      all-on control group (all three enabled are True)
        reward_no_deep:     deep-sea discovery off (deep_sea_reward_enabled=False)
        reward_no_curiosity: curiosity off (curiosity_reward_enabled=False)
        reward_no_penalty:  penalty off (penalty_reward_enabled=False)
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Args:
            config: config dict; keys strictly match configs/ablation/*.yaml:
                # deep-sea discovery reward (Req. Sec. 5: default +10)
                - deep_sea_reward_enabled (bool): whether to enable the deep-sea discovery reward, default True
                - deep_sea_reward_value (float): deep-sea discovery reward value, default 10.0
                - deep_sea_distance_threshold (float): deep-sea spacing threshold, default 416.0

                # curiosity reward (Req. Sec. 5: default +3, weight adjustable)
                - curiosity_reward_enabled (bool): whether to enable the curiosity reward, default True
                - curiosity_reward_value (float): curiosity reward value, default 3.0
                - curiosity_weight (float): curiosity weight, default 1.0

                # penalty (Req. Sec. 5: out-of-bounds -10 / repeated visit -5)
                - penalty_reward_enabled (bool): whether to enable the penalty, default True
                - out_of_bounds_penalty (float): out-of-bounds penalty value, default -10.0
                - repeat_visit_penalty (float): repeated-visit penalty value, default -5.0
        """
        config = config or {}

        # deep-sea discovery reward module
        deep_sea_config = {
            "enabled": config.get("deep_sea_reward_enabled", True),
            "reward": config.get("deep_sea_reward_value", 10.0),
            "distance_threshold": config.get("deep_sea_distance_threshold", 416.0),
        }
        self._deep_sea = DeepSeaReward(deep_sea_config)

        # curiosity reward module
        curiosity_config = {
            "enabled": config.get("curiosity_reward_enabled", True),
            "reward": config.get("curiosity_reward_value", 3.0),
            "weight": config.get("curiosity_weight", 1.0),
        }
        self._curiosity = CuriosityReward(curiosity_config)

        # penalty module
        penalty_config = {
            "enabled": config.get("penalty_reward_enabled", True),
            "out_of_bounds": config.get("out_of_bounds_penalty", -10.0),
            "repeated_visit": config.get("repeat_visit_penalty", -5.0),
        }
        self._penalty = PenaltyReward(penalty_config)

    def calculate(
        self,
        is_new_deep_sea: bool = False,
        distance_to_nearest: Optional[float] = None,
        is_new_state: bool = False,
        is_out_of_bounds: bool = False,
        is_repeated_visit: bool = False,
    ) -> float:
        """
        Compute the total reward (sum of the three rewards)

        Call order is strictly: deep-sea discovery -> curiosity -> penalty,
        the modules are fully decoupled and do not affect each other.

        Args:
            is_new_deep_sea: whether this is a valid new deep-sea point (passed the spacing-threshold check)
            distance_to_nearest: Euclidean distance to the nearest existing deep-sea point (for logging only)
            is_new_state: whether this is the first visit to this discrete state (determined by state_manager)
            is_out_of_bounds: whether it goes outside the search range (radius configured by search_radius)
            is_repeated_visit: whether it repeatedly visits an already-queried chunk

        Returns:
            the total reward (float), used for the Q-learning TD-error computation
        """
        total = 0.0
        total += self._deep_sea.calculate(is_new_deep_sea, distance_to_nearest)
        total += self._curiosity.calculate(is_new_state)
        total += self._penalty.calculate(is_out_of_bounds, is_repeated_visit)
        return total

    @property
    def deep_sea(self) -> DeepSeaReward:
        """deep-sea discovery reward module instance"""
        return self._deep_sea

    @property
    def curiosity(self) -> CuriosityReward:
        """curiosity reward module instance"""
        return self._curiosity

    @property
    def penalty(self) -> PenaltyReward:
        """penalty reward module instance"""
        return self._penalty

    @property
    def config_summary(self) -> str:
        """
        Return a summary of the current reward module's switches and parameters

        used by main_algorithm.py to dynamically generate the output header at startup,
        meeting the Req. Sec. 10 "dynamic output header" requirement.
        """
        deep_status = f"deep-sea discovery={'on' if self._deep_sea.enabled else 'off'}(+{self._deep_sea.reward})"
        curi_status = f"curiosity={'on' if self._curiosity.enabled else 'off'}(+{self._curiosity.reward}x{self._curiosity.weight})"
        pen_status = f"penalty={'on' if self._penalty.enabled else 'off'}(out-of-bounds {self._penalty.out_of_bounds}/repeated {self._penalty.repeated_visit})"
        return f"{deep_status} | {curi_status} | {pen_status}"
