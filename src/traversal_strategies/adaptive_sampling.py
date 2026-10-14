# src/traversal_strategies/adaptive_sampling.py
"""[微观] 策略8：根据温度反馈动态调整查点位置"""
from typing import List, Tuple, Optional
from .base_strategy import BaseStrategy
from ._utils import generate_grid_points, generate_corner_points


class AdaptiveSamplingStrategy(BaseStrategy):
    """
    自适应采样：根据已积累的温度数据动态调整区块内采样策略

    当温度开启时：
        若当前区块温度接近已发现深海的平均温度 → 采用3x3网格密集采样
        否则 → 仅查四角+中心（轻量采样）

    当温度关闭时：
        退化为四角+中心5点采样
    """

    def __init__(self, reference_temps: Optional[List[float]] = None, **kwargs):
        super().__init__(**kwargs)
        self._reference_temps = reference_temps or []

    @property
    def name(self) -> str:
        return "adaptive_sampling"

    def set_reference_temps(self, temps: List[float]) -> None:
        """外部注入参考温度（由main_algorithm.py在发现深海点时调用）"""
        self._reference_temps = list(temps)

    def get_points(self, chunk_x: int, chunk_z: int, temperature: float = None) -> List[Tuple[int, int]]:
        """
        根据温度反馈决定采样策略
        Args:
            chunk_x/chunk_z: 区块坐标
            temperature: 当前区块温度（由main_algorithm.py注入）
        """
        # 无参考温度或无当前温度 → 使用默认策略
        if not self._reference_temps or temperature is None:
            return self._default_points(chunk_x, chunk_z)

        import math
        ref_mean = sum(self._reference_temps) / len(self._reference_temps)

        # 计算参考温度标准差
        if len(self._reference_temps) >= 3:
            ref_std = (sum((t - ref_mean) ** 2 for t in self._reference_temps) / len(self._reference_temps)) ** 0.5
            threshold = ref_std * 0.5 if ref_std > 0 else 0.3
        else:
            threshold = 0.3

        # 温度接近参考均值 → 密集采样
        if abs(temperature - ref_mean) < threshold:
            return generate_grid_points(chunk_x, chunk_z, rows=3, cols=3)

        return self._default_points(chunk_x, chunk_z)

    def _default_points(self, chunk_x: int, chunk_z: int) -> List[Tuple[int, int]]:
        """四角+中心5点采样"""
        cx = chunk_x * 16 + 8
        cz = chunk_z * 16 + 8
        return [(cx, cz)] + generate_corner_points(chunk_x, chunk_z)