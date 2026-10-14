"""
base_strategy.py
[微观] 区块内选点策略抽象基类
核心职责：定义统一的"获取待查坐标列表"接口
所有9种微观策略必须继承此类，确保main_algorithm.py可通过字符串名动态实例化
严禁在此文件中包含任何搜索逻辑或奖励计算

【修复依据】(≤100字)
抽象方法签名补 temperature 可选参数：9 个子类实现均已带该参数，原基类缺失导致
签名不一致，且 main_algorithm 无法把黑盒温度反馈合法注入 adaptive_sampling，
使需求第九节"根据温度反馈动态调整查点位置"永久失效。仅补可选参数，不改变语义。
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional


class BaseStrategy(ABC):
    """
    微观选点策略基类
    ⚠️ 不可直接实例化，必须被子类继承实现

    子类实现约束：
        1. 必须实现 get_points() 方法，返回按探索优先级降序排列的坐标列表
        2. 必须通过 @property 显式声明 name，与 YAML 配置中的字符串严格匹配
        3. __init__ 必须接受 **kwargs 以兼容注册表统一实例化
        4. get_points() 必须接受 temperature 可选参数：
           纯几何策略可忽略它；adaptive_sampling 依赖它做动态调整
    """

    def __init__(self, **kwargs):
        """
        策略构造器（所有子类必须兼容此签名）
        Args:
            **kwargs: 策略专属超参数（如 random_two_points 的 seed 等）
        """
        pass

    @abstractmethod
    def get_points(self, chunk_x: int, chunk_z: int,
                   temperature: Optional[float] = None) -> List[Tuple[int, int]]:
        """
        根据目标区块坐标，返回区块内需要查询的具体世界坐标列表
        Args:
            chunk_x: 目标区块的X坐标（区块级，非世界坐标）
            chunk_z: 目标区块的Z坐标（区块级，非世界坐标）
            temperature: 最近一次黑盒查询得到的温度反馈（可能为 None）
                         仅 adaptive_sampling 使用，其余纯几何策略应忽略。
                         ⚠️ 不得据此设计"某温度更好"的先验规则（需求第一节）
        Returns:
            待查询的世界坐标列表 [(x1, z1), (x2, z2), ...]
            ⚠️ 必须按探索优先级降序排列（首个坐标为最优候选点）
            坐标必须在目标区块范围内（chunk_x*16 ~ chunk_x*16+15）
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        策略名称标识符，必须与 YAML 配置中的 micro_strategy 字段严格匹配
        示例：Grid3x3Strategy 的 name 应返回 "grid_3x3"
        """
        pass
