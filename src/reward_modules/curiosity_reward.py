
"""
curiosity_reward.py
[奖励] 好奇心奖励模块
核心职责：当 AI 到达从未访问过的状态（区块坐标 + 温度分桶）时给予正向奖励
通过 config 中的布尔开关控制是否生效，权重可通过 YAML 调节
严禁在此文件中硬编码任何奖励数值
"""

from typing import Optional


class CuriosityReward:
    """好奇心奖励计算器

    状态由 state_manager 定义为 (chunk_x, chunk_z, temp_bin) 的元组。
    当 AI 首次到达某个离散状态时，给予正向奖励，激励 AI 探索未知区域，
    避免在已探索区域循环。

    消融实验控制：
        config["curiosity_enabled"] = False  → 关闭好奇心奖励（Stage3 reward_no_curiosity 场景）
        config["curiosity_weight"] 可调节权重（Stage4 超参数扫描场景）
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Args:
            config: 配置字典，支持以下键：
                - enabled (bool): 是否启用好奇心奖励，默认 True
                - reward (float): 到达新状态时的基础奖励值，默认 3.0
                - weight (float): 好奇心奖励权重（可调），默认 1.0
                    总好奇心奖励 = reward × weight（当 is_new_state 为 True 时）
        """
        config = config or {}
        self._enabled = config.get("enabled", True)
        self._reward = float(config.get("reward", 3.0))
        self._weight = float(config.get("weight", 1.0))

    def calculate(self, is_new_state: bool) -> float:
        """
        计算好奇心奖励

        Args:
            is_new_state: 当前状态是否为首次访问
                         由 state_manager.is_new_state() 的返回值决定

        Returns:
            好奇心奖励值（float）。未启用或非新状态时返回 0.0
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