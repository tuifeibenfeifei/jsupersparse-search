"""
state_manager.py
[状态] 状态离散化编码器：将(区块区域坐标+温度)映射为Q-table可用的hashable key
核心职责：状态编码、新状态判定（供好奇心奖励使用）
严禁包含任何奖励计算、搜索逻辑或数据库操作

【设计约束一】需求.txt 第一节「严禁先验知识」：
    温度仅作为状态特征输入模型，不得引入任何人工设定的温度范围/阈值。
    因此本文件不使用 temp_min / temp_max 分桶，改为「固定小数位量化」：
        round(temperature, precision)
    precision 是分辨率（保留几位小数），不是温度值域上下界，
    不对"何种温度更接近深海"做任何假设，完全由 AI 自主学习。

【设计约束二】空间粒度必须粗于温度分辨率（本文件最易踩的坑）：
    温度层分辨率为 16 区块（256方块），同一区块内温度恒定。
    若状态格细于或等于 16 区块，温度可由区块坐标整除推出，
    成为确定性函数，对状态空间零贡献，temperature on/off 消融必然同结果。
    故引入 spatial_granularity（默认 32），强制一个状态格跨越多个温度格。
"""

import math
from typing import Tuple, Optional


# 温度缺失（查询失败/NaN）时的独立标记。
# 不使用 -1：-1 落在真实温度值域内，会与某个真实量化桶冲突。
TEMP_MISSING = None

# 温度层的空间分辨率（区块数）= 256方块 / 16方块每区块 = 16 区块。
# 源码依据 cubiomes：generator.h:57 (L_OCEAN_TEMP_256) 与 layers.c:3130 (x/8.0 采样)。
# 实测校验：温度沿 x 方向变化间隔均值恰为 16 区块；同一区块内 2000 次采样
# 温度完全恒定（最大温差 0.000000）。
TEMP_RESOLUTION_CHUNKS = 16

# 状态空间粒度默认值：取温度分辨率的 2 倍。
# 必须「粗于」温度分辨率，温度才携带增量信息；若细于或等于它，
# 温度即成为区块坐标的确定性函数，对状态空间零贡献。
DEFAULT_SPATIAL_GRANULARITY = 2 * TEMP_RESOLUTION_CHUNKS


class StateManager:
    """
    状态管理器
    ⚠️ 非线程安全，仅供单 episode 内串行使用
    温度关闭时状态仅由区块区域坐标决定；开启时温度量化后参与编码

    好奇心奖励时序约束：
        reward_module 必须先调用 is_new_state() 判定，再调用 mark_visited() 记录
        顺序颠倒将导致好奇心奖励永远为0

    【空间粒度与温度分辨率的关系 —— 本类最关键的设计约束】
        温度层分辨率固定为 TEMP_RESOLUTION_CHUNKS(16) 个区块。
        - spatial_granularity <= 16 时：一个状态格内温度恒定，温度是区块坐标的
          确定性函数，(region_x, region_z, temp) 与 (region_x, region_z) 一一对应，
          温度对状态空间零贡献，temperature on/off 消融必然得到完全相同的结果。
        - spatial_granularity > 16 时：一个状态格跨越多个温度格，温度用于区分
          同一区域内的不同温度带，才真正携带增量信息。
        故默认取 32（= 2 倍温度分辨率），每个状态格约覆盖 2x2 个温度格。
    """

    def __init__(
        self,
        temperature_enabled: bool = True,
        temperature_precision: int = 2,
        spatial_granularity: int = DEFAULT_SPATIAL_GRANULARITY,
    ):
        """
        Args:
            temperature_enabled: 是否将温度纳入状态编码
                True=黑盒触觉感知模式（创新基准）
                False=纯盲搜对照模式（消融实验用）
            temperature_precision: 温度量化精度，即保留小数位数（默认2位=0.01分辨率）
                仅作分辨率使用，不设定温度范围，不引入任何先验知识。
                Stage4 可消融测试不同粒度（1位=粗, 2位=中, 3位=细）
            spatial_granularity: 状态格的空间粒度，单位=区块（默认32）
                一个状态格覆盖 spatial_granularity x spatial_granularity 个区块。
                ⚠️ 必须 > TEMP_RESOLUTION_CHUNKS(16) 温度才有增量信息，详见类文档。
                这是状态空间的分辨率超参数，与"何种温度更接近深海"无关，
                不构成需求第一节所禁止的先验知识；Stage4 可对其做消融扫描。
        """
        self._validate_config(temperature_precision, spatial_granularity)
        self._temp_enabled = bool(temperature_enabled)
        self._precision = int(temperature_precision)
        self._granularity = int(spatial_granularity)
        self._visited_states: set = set()

    @staticmethod
    def _validate_config(temperature_precision, spatial_granularity) -> None:
        """集中校验量化精度与空间粒度配置的合法性"""
        if (not isinstance(temperature_precision, int)
                or isinstance(temperature_precision, bool)
                or temperature_precision < 0):
            raise ValueError(
                f"temperature_precision必须为非负整数，收到: {temperature_precision} "
                f"(type={type(temperature_precision).__name__})"
            )
        if (not isinstance(spatial_granularity, int)
                or isinstance(spatial_granularity, bool)
                or spatial_granularity <= 0):
            raise ValueError(
                f"spatial_granularity必须为正整数，收到: {spatial_granularity} "
                f"(type={type(spatial_granularity).__name__})"
            )

    def reconfigure(
        self,
        temperature_enabled: bool = None,
        temperature_precision: int = None,
        spatial_granularity: int = None,
    ) -> None:
        """
        动态重置状态管理器配置（供Stage4消融实验使用）
        ⚠️ 调用后会自动清空已访问记录，确保新旧配置不混用
        Args:
            任一参数为None时保持当前值不变
        """
        new_enabled = temperature_enabled if temperature_enabled is not None else self._temp_enabled
        new_precision = temperature_precision if temperature_precision is not None else self._precision
        new_gran = spatial_granularity if spatial_granularity is not None else self._granularity

        self._validate_config(new_precision, new_gran)

        self._temp_enabled = bool(new_enabled)
        self._precision = int(new_precision)
        self._granularity = int(new_gran)
        self._visited_states.clear()  # 配置变更后必须清空历史访问记录

    def get_state(self, chunk_x: int, chunk_z: int, temperature: Optional[float]) -> Tuple:
        """
        将连续输入编码为离散状态key
        Returns:
            hashable tuple，可直接作为Q-table字典键
            温度缺失编码为独立标记 None，不与任何真实温度桶混淆，不伪造触觉反馈
        """
        if not isinstance(chunk_x, int) or isinstance(chunk_x, bool):
            raise TypeError(f"chunk_x必须为int，收到 {type(chunk_x).__name__}")
        if not isinstance(chunk_z, int) or isinstance(chunk_z, bool):
            raise TypeError(f"chunk_z必须为int，收到 {type(chunk_z).__name__}")

        # 空间粒度粗化：Python 的 // 为向下取整，负坐标同样成立，不产生环绕错位
        region_x = chunk_x // self._granularity
        region_z = chunk_z // self._granularity

        if self._temp_enabled:
            return (region_x, region_z, self._quantize_temp(temperature))
        return (region_x, region_z)

    def is_new_state(self, state: Tuple) -> bool:
        """
        判定状态是否从未访问过（供好奇心奖励模块调用）
        ⚠️ 仅查询不记录，必须在 mark_visited() 之前调用
        """
        return state not in self._visited_states

    def mark_visited(self, state: Tuple) -> None:
        """
        标记状态为已访问
        ⚠️ 必须在 is_new_state() 之后、agent完成状态转移时调用
        """
        self._visited_states.add(state)

    def reset(self) -> None:
        """
        清空访问记录，保持当前配置不变
        ⚠️ 每次新episode开始前必须调用，确保无历史经验残留（需求第一节第3条）
        """
        self._visited_states.clear()

    @property
    def temperature_enabled(self) -> bool:
        return self._temp_enabled

    @property
    def temperature_precision(self) -> int:
        """当前温度量化精度（保留小数位数）"""
        return self._precision

    @property
    def spatial_granularity(self) -> int:
        """当前状态格空间粒度（区块数）。须 > TEMP_RESOLUTION_CHUNKS 温度才有增量信息"""
        return self._granularity

    @property
    def temperature_resolution_chunks(self) -> int:
        """温度层分辨率（区块数），常量，等于 16"""
        return TEMP_RESOLUTION_CHUNKS

    @property
    def granularity_is_coarse_enough(self) -> bool:
        """空间粒度是否粗于温度分辨率（True 时温度才对状态空间有贡献）"""
        return self._granularity > TEMP_RESOLUTION_CHUNKS

    @property
    def visited_count(self) -> int:
        """当前episode已访问的唯一状态数"""
        return len(self._visited_states)

    def _quantize_temp(self, temperature: Optional[float]) -> Optional[float]:
        """
        将连续温度值量化为离散key
        None / NaN / 非数值 → TEMP_MISSING（独立标记，不伪造触觉反馈）

        说明：量化只做四舍五入到固定小数位，不截断、不平移、不设定任何范围，
        因此不会把"某些温度更可能是深海"的先验知识编码进状态空间。
        """
        if temperature is None:
            return TEMP_MISSING
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            return TEMP_MISSING
        if math.isnan(temperature):
            return TEMP_MISSING

        return round(float(temperature), self._precision)
