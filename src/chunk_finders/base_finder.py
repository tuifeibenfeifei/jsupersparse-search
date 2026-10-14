
"""
base_finder.py
[宏观] 区块遍历策略抽象基类
核心职责：定义统一的"获取下一个待查区块"接口
所有宏观策略必须继承此类，确保main_algorithm.py可通过字符串名动态实例化
严禁在此文件中包含任何微观选点逻辑或奖励计算

【修复依据】(≤100字)
新增 reset() 钩子（默认空实现）：宏观策略持有内部游标（方向索引、步长、当前坐标），
若不在每个 episode 前重置，同一 Explorer 连续跑多次时游标会跨 episode 累加，
第2次起不再从原点扫描，导致 episode 之间互相污染。子类按需覆写。
"""

from abc import ABC, abstractmethod
from typing import Tuple


class BaseFinder(ABC):
    """
    宏观区块遍历策略基类
    ⚠️ 不可直接实例化，必须被子类继承实现

    子类实现约束：
        1. 必须实现 get_next_chunk(current_x, current_z) 方法
        2. 必须通过 @property 显式声明 name，与 YAML 配置严格匹配
        3. __init__ 必须接受 config 参数以兼容 main_algorithm.py 的实例化方式
        4. 若持有内部游标状态，必须覆写 reset() 以支持无状态重置（需求第一节第3条）
    """

    def __init__(self, config=None, **kwargs):
        """
        策略构造器（所有子类必须兼容此签名）
        Args:
            config: 配置字典（由 main_algorithm.py 的 self.config 传入）
            **kwargs: 策略专属超参数
        """
        self.config = config or {}

    @abstractmethod
    def get_next_chunk(self, current_x: int, current_z: int) -> Tuple[int, int]:
        """
        根据当前区块坐标，返回下一个待查询的区块坐标
        Args:
            current_x: 当前区块X坐标
            current_z: 当前区块Z坐标
        Returns:
            (chunk_x, chunk_z) 元组，区块级坐标（非世界坐标）
        """
        pass

    def reset(self) -> None:
        """
        重置内部遍历状态，使下一次 get_next_chunk() 重新从起点开始。
        默认空实现：无状态策略无需覆写。
        由 main_algorithm.py 在每个 episode 开始时调用。
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称标识符，必须与YAML配置中的chunk_finder字段严格匹配"""
        pass
