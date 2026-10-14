"""
database.py
[数据] 内存级深海点数据库：存储已确认的深海点坐标与温度
核心职责：欧氏距离去重校验、有效点集合管理
严禁包含任何搜索逻辑、奖励计算或状态编码
"""

import math
from typing import List, Tuple


class DeepSeaDatabase:
    """
    深海点记录管理器
    ⚠️ 非线程安全，仅供单 episode 内串行使用
    每次实验开始前必须调用 clear() 重置，确保无历史经验残留
    """

    def __init__(self, distance_threshold: float = 416.0):
        """
        Args:
            distance_threshold: 有效深海点最小欧氏距离（默认416格）
        """
        if distance_threshold <= 0:
            raise ValueError(f"间距阈值必须为正数，收到: {distance_threshold}")
        self._threshold = float(distance_threshold)
        self._threshold_sq = self._threshold * self._threshold  # 预计算平方
        self._points: List[Tuple[int, int, float]] = []

    def add_point(self, x: int, z: int, temp: float) -> bool:
        """
        尝试添加一个新深海点
        Returns:
            True=成功添加（有效新点）
            False=被去重过滤（距离过近）
        """
        if not isinstance(x, int) or not isinstance(z, int):
            raise TypeError(f"坐标必须为int类型，收到 x={type(x).__name__}, z={type(z).__name__}")
        if temp is not None and (not isinstance(temp, (int, float)) or math.isnan(temp)):
            raise TypeError(f"温度必须为有效数值，收到 temp={temp}")

        if self._is_valid_point(x, z):
            self._points.append((x, z, float(temp) if temp is not None else 0.0))
            return True
        return False

    def _is_valid_point(self, x: int, z: int) -> bool:
        """校验新点与所有已有点的欧氏距离是否 ≥ 阈值"""
        for px, pz, _ in self._points:
            dx = x - px
            dz = z - pz
            if dx * dx + dz * dz < self._threshold_sq:
                return False
        return True

    def get_nearest_distance(self, x: int, z: int) -> float:
        """
        返回候选点(x, z)到所有已有深海点的最小欧氏距离
        数据库为空时返回 0.0（表示尚无参照点）
        供 main_algorithm.py 传给奖励模块的 distance_to_nearest 参数使用
        """
        if not self._points:
            return 0.0
        min_dist_sq = min((x - px) ** 2 + (z - pz) ** 2 for px, pz, _ in self._points)
        return math.sqrt(min_dist_sq)

    def get_all_coordinates(self) -> List[Tuple[int, int]]:
        """返回所有有效深海点的(x, z)坐标列表"""
        return [(p[0], p[1]) for p in self._points]

    def get_all_points(self) -> List[Tuple[int, int, float]]:
        """返回所有有效深海点的完整记录(x, z, temp)的副本"""
        return list(self._points)

    def clear(self) -> None:
        """清空数据库，重置为空状态"""
        self._points.clear()

    def __len__(self) -> int:
        return len(self._points)

    @property
    def distance_threshold(self) -> float:
        return self._threshold
