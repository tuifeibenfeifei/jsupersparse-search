
"""
penalty_reward.py
[奖励] 惩罚奖励模块
核心职责：对越界行为和重复访问施加负奖励（惩罚）
通过 config 中的布尔开关控制是否生效，惩罚数值由 YAML 注入
严禁在此文件中硬编码任何惩罚数值
"""

from typing import Optional


class PenaltyReward:
    """惩罚奖励计算器

    施加两类惩罚：
        1. 越界惩罚：当 AI 的下一步坐标超出搜索范围（半径 10000 格的正方形区域）时触发
        2. 重复访问惩罚：当 AI 重复查询已访问过的区块坐标时触发

    惩罚信号驱动 AI 避免无效移动，保持在有效搜索区域内。

    消融实验控制：
        config["penalty_enabled"] = False  → 关闭所有惩罚（Stage3 reward_no_penalty 场景）
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Args:
            config: 配置字典，支持以下键：
                - enabled (bool): 是否启用惩罚，默认 True
                - out_of_bounds (float): 越界惩罚值（负数），默认 -10.0
                - repeated_visit (float): 重复访问惩罚值（负数），默认 -5.0
        """
        config = config or {}
        self._enabled = config.get("enabled", True)
        self._out_of_bounds = float(config.get("out_of_bounds", -10.0))
        self._repeated_visit = float(config.get("repeated_visit", -5.0))

    def calculate(self, is_out_of_bounds: bool = False, is_repeated_visit: bool = False) -> float:
        """
        计算惩罚值（可叠加）

        Args:
            is_out_of_bounds: 是否超出搜索范围（半径 10000 格的正方形区域）
            is_repeated_visit: 是否重复访问已查询过的区块坐标

        Returns:
            惩罚总值（float）。未启用时返回 0.0
            若两项惩罚同时触发，则总惩罚 = out_of_bounds + repeated_visit
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