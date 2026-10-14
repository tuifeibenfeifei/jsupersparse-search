
"""
full_reward_baseline.py
[奖励] 全奖励聚合器（对照组）
核心职责：聚合深海发现、好奇心、惩罚三项奖励，根据开关状态加权求和
被 main_algorithm.py 在 Stage3 奖励驱动阶段调用，返回用于 Q-learning 更新的标量奖励值

设计说明：
    本文件不实现任何奖励计算逻辑，仅作为三个子模块的聚合器。
    每个子模块独立通过 config 中的布尔开关控制启停，
    支持 Stage3 消融实验中"单独关闭某一项"的场景。

【修复依据】(≤100字)
读取的配置键名全部对齐 configs/ 下 YAML 的实际写法（需求第十节：配置文件驱动，
严禁硬编码）。原实现读的是 deep_sea_enabled / deep_sea_reward / penalty_repeated_visit
等自造键名，YAML 里并不存在，导致 7 个开关与数值全部静默走默认值——Stage3 的
消融配置写了也不生效。现改为读 deep_sea_reward_enabled / deep_sea_reward_value /
repeat_visit_penalty 等，与 temp_on.yaml 逐字一致。奖励仍为 3 项之和，结构不变。
"""

from typing import Optional
from .deep_sea_reward import DeepSeaReward
from .curiosity_reward import CuriosityReward
from .penalty_reward import PenaltyReward


class FullRewardBaseline:
    """
    全奖励聚合器

    读取配置中的布尔开关，决定是否启用深海发现、好奇心、惩罚三项奖励，
    并返回加权求和后的总奖励值。

    需求第五节：总奖励为以下各项之和，温度仅作状态特征，不直接参与奖励计算。

    消融实验配置示例（Stage3）：
        reward_all_on:      全开对照组（三项 enabled 均为 True）
        reward_no_deep:     关闭深海发现（deep_sea_reward_enabled=False）
        reward_no_curiosity: 关闭好奇心（curiosity_reward_enabled=False）
        reward_no_penalty:  关闭惩罚（penalty_reward_enabled=False）
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Args:
            config: 配置字典，键名与 configs/ablation/*.yaml 严格一致：
                # 深海发现奖励（需求第五节：默认 +10）
                - deep_sea_reward_enabled (bool): 是否启用深海发现奖励，默认 True
                - deep_sea_reward_value (float): 深海发现奖励值，默认 10.0
                - deep_sea_distance_threshold (float): 深海间距阈值，默认 416.0

                # 好奇心奖励（需求第五节：默认 +3，权重可调）
                - curiosity_reward_enabled (bool): 是否启用好奇心奖励，默认 True
                - curiosity_reward_value (float): 好奇心奖励值，默认 3.0
                - curiosity_weight (float): 好奇心权重，默认 1.0

                # 惩罚（需求第五节：越界 -10 / 重复访问 -5）
                - penalty_reward_enabled (bool): 是否启用惩罚，默认 True
                - out_of_bounds_penalty (float): 越界惩罚值，默认 -10.0
                - repeat_visit_penalty (float): 重复访问惩罚值，默认 -5.0
        """
        config = config or {}

        # 深海发现奖励模块
        deep_sea_config = {
            "enabled": config.get("deep_sea_reward_enabled", True),
            "reward": config.get("deep_sea_reward_value", 10.0),
            "distance_threshold": config.get("deep_sea_distance_threshold", 416.0),
        }
        self._deep_sea = DeepSeaReward(deep_sea_config)

        # 好奇心奖励模块
        curiosity_config = {
            "enabled": config.get("curiosity_reward_enabled", True),
            "reward": config.get("curiosity_reward_value", 3.0),
            "weight": config.get("curiosity_weight", 1.0),
        }
        self._curiosity = CuriosityReward(curiosity_config)

        # 惩罚模块
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
        计算总奖励值（三项奖励之和）

        调用顺序严格为：深海发现 → 好奇心 → 惩罚，
        各模块之间完全解耦，互不影响。

        Args:
            is_new_deep_sea: 是否为有效新深海点（已通过间距阈值校验）
            distance_to_nearest: 到最近已有深海点的欧氏距离（仅用于日志）
            is_new_state: 是否首次访问该离散状态（由 state_manager 判定）
            is_out_of_bounds: 是否超出搜索范围（半径由 search_radius 配置）
            is_repeated_visit: 是否重复访问已查询过的区块

        Returns:
            总奖励值（float），用于 Q-learning 的 TD 误差计算
        """
        total = 0.0
        total += self._deep_sea.calculate(is_new_deep_sea, distance_to_nearest)
        total += self._curiosity.calculate(is_new_state)
        total += self._penalty.calculate(is_out_of_bounds, is_repeated_visit)
        return total

    @property
    def deep_sea(self) -> DeepSeaReward:
        """深海发现奖励模块实例"""
        return self._deep_sea

    @property
    def curiosity(self) -> CuriosityReward:
        """好奇心奖励模块实例"""
        return self._curiosity

    @property
    def penalty(self) -> PenaltyReward:
        """惩罚奖励模块实例"""
        return self._penalty

    @property
    def config_summary(self) -> str:
        """
        返回当前奖励模块的开关与参数摘要

        供 main_algorithm.py 在程序启动时动态生成输出说明头，
        满足需求第十条"动态输出说明头"的规范要求。
        """
        deep_status = f"深海发现={'开' if self._deep_sea.enabled else '关'}(+{self._deep_sea.reward})"
        curi_status = f"好奇心={'开' if self._curiosity.enabled else '关'}(+{self._curiosity.reward}x{self._curiosity.weight})"
        pen_status = f"惩罚={'开' if self._penalty.enabled else '关'}(越界{self._penalty.out_of_bounds}/重复{self._penalty.repeated_visit})"
        return f"{deep_status} | {curi_status} | {pen_status}"
