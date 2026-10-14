
"""
deep_sea_reward.py
[奖励] 深海发现奖励模块
核心职责：当发现新深海点（距所有已有深海点 ≥ 间距阈值）时给予正向奖励
通过 config 中的布尔开关控制是否生效，所有数值参数由 YAML 注入
严禁在此文件中硬编码任何奖励数值
"""

from typing import Optional


class DeepSeaReward:
    """深海发现奖励计算器

    当 AI 查询到一个新的深海点，且该点与所有已有深海点的欧氏距离均 ≥ 间距阈值时，
    给予正向奖励。该奖励信号驱动 AI 主动寻找"新区域"的深海点，而非在已发现区域反复搜索。

    消融实验控制：
        config["deep_sea_enabled"] = False  → 关闭此项奖励（Stage3 reward_no_deep 场景）
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Args:
            config: 配置字典，支持以下键：
                - enabled (bool): 是否启用深海发现奖励，默认 True
                - reward (float): 发现有效深海点时的奖励值，默认 10.0
                - distance_threshold (float): 有效深海点最小间距阈值（格），默认 416.0
        """
        config = config or {}
        self._enabled = config.get("enabled", True)
        self._reward = float(config.get("reward", 10.0))
        self._distance_threshold = float(config.get("distance_threshold", 416.0))

    def calculate(self, is_new_deep_sea: bool, distance_to_nearest: Optional[float] = None) -> float:
        """
        计算深海发现奖励

        Args:
            is_new_deep_sea: 是否为有效新深海点（已通过 database._is_valid_point 的欧氏距离校验）
            distance_to_nearest: 到最近已有深海点的欧氏距离（仅用于日志/调试，不影响计算）

        Returns:
            奖励值（float）。未启用或非新深海点时返回 0.0
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