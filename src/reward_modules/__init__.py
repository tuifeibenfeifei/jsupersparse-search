"""reward_modules/__init__.py - 奖励模块插件包"""
from .curiosity_reward import CuriosityReward
from .deep_sea_reward import DeepSeaReward
from .full_reward_baseline import FullRewardBaseline
from .penalty_reward import PenaltyReward

__all__ = ["CuriosityReward", "DeepSeaReward", "FullRewardBaseline", "PenaltyReward"]