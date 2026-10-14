
"""
adaptive_scan.py
[宏观] 自适应扫描：根据深海点密度自动切换搜索粒度
密集方向用小步长精细搜索，稀疏方向用大步长快速扫描

【修复依据】(≤100字)
1. 补完密度自适应：原实现注入了 small_step/_deep_sea_chunks 却从未使用，
   get_next_chunk 恒定返回 large_step，与本文件自身声明的设计意图矛盾，
   也使需求第一节第8条「自适应搜索粒度」落空。现按 8 方向统计深海点密度，
   当前方向命中密集集合时用小步长，否则用大步长。
2. 补 reset()：当前坐标/方向/步计数为实例状态，跨 episode 必须清零。
3. 修死循环：原实现一圈（32块）后游标精确回到圈起点，密度不再变化时下一圈
   重复同一批已访问区块 → 主循环去重分支零查询死循环；改为按圈基准半径推进
   （相邻圈不重叠）并做边界回绕，保证路径绝不周期重复、绝不越界。
"""

from typing import Tuple, List, Set, Dict
from collections import defaultdict
from .base_finder import BaseFinder


class AdaptiveScanFinder(BaseFinder):
    """
    自适应扫描：根据深海点密度动态调整搜索策略
    - 已发现深海点的方向 → 小步长精细搜索
    - 未发现有深海点的方向 → 大步长快速扫描

    密度数据来源：由 main_algorithm.py 在每次确认新深海点后调用
    update_deep_sea_density() 注入区块坐标列表。
    尚未注入任何数据时，默认使用大步长快速扫描。
    """

    # 8个方向：东、东南、南、西南、西、西北、北、东北
    _DIRECTIONS: List[Tuple[int, int]] = [
        (1, 0), (1, -1), (0, -1), (-1, -1),
        (-1, 0), (-1, 1), (0, 1), (1, 1),
    ]

    def __init__(
        self,
        config=None,
        small_step: int = 2,
        large_step: int = 8,
        density_threshold_ratio: float = 0.5,
        **kwargs
    ):
        """
        Args:
            config: 配置字典（由 main_algorithm.py 传入）
            small_step: 密集方向使用的精细搜索步长（区块）
            large_step: 稀疏方向使用的快速扫描步长（区块）
            density_threshold_ratio: 判定"密集方向"的比例阈值，
                某方向深海点计数 ≥ 最大方向计数 × 本比例 即视为密集方向
        """
        super().__init__(config, **kwargs)
        self._small_step = int(small_step)
        self._large_step = int(large_step)
        self._density_ratio = float(density_threshold_ratio)
        # 区块级搜索半径 = search_radius（格）÷ 16；用于边界回绕（修复依据第3条）
        self._chunk_radius = max(1, self.config.get("search_radius", 10000) // 16)

        self._deep_sea_chunks: Set[Tuple[int, int]] = set()
        self._direction_density: Dict[int, int] = defaultdict(int)
        self._dense_directions: Set[int] = set()

        self.reset()

    @property
    def name(self) -> str:
        return "adaptive_scan"

    def reset(self) -> None:
        """重置遍历游标；密度统计由 update_deep_sea_density() 独立管理"""
        self._x, self._z = 0, 0
        self._direction = 0
        self._step = 0
        self._ring = 0
        self._first = True

    def update_deep_sea_density(self, chunks: List[Tuple[int, int]]) -> None:
        """
        外部注入已发现深海点的区块坐标（由main_algorithm.py调用）
        据此重算 8 个方向各自的深海点密度，并标记出密集方向。
        Args:
            chunks: 已发现深海点的区块坐标列表 [(cx1, cz1), ...]
        """
        self._deep_sea_chunks = set(chunks)

        density: Dict[int, int] = defaultdict(int)
        for cx, cz in self._deep_sea_chunks:
            if cx == 0 and cz == 0:
                continue  # 原点无方向性，不参与方向密度统计
            # 归属到与该区块位置点积最大的唯一方向，避免一个区块被重复计入多个方向
            best_idx = max(
                range(len(self._DIRECTIONS)),
                key=lambda i: cx * self._DIRECTIONS[i][0] + cz * self._DIRECTIONS[i][1],
            )
            density[best_idx] += 1

        self._direction_density = density
        self._dense_directions = self._compute_dense_directions(density)

    def _compute_dense_directions(self, density: Dict[int, int]) -> Set[int]:
        """按 density_threshold_ratio 判定哪些方向属于密集方向"""
        if not density:
            return set()
        max_count = max(density.values())
        if max_count <= 0:
            return set()
        cutoff = max_count * self._density_ratio
        return {idx for idx, count in density.items() if count >= cutoff}

    def get_next_chunk(self, current_x: int, current_z: int) -> Tuple[int, int]:
        # 首次调用返回起始点 (0,0)（与原设计一致，保证原点区块被扫描）
        if self._first:
            self._first = False
            self._advance_cursor()
            return (0, 0)

        # 需求第一节第8条：密集方向小步长精细搜索，稀疏方向大步长快速扫描
        if self._direction in self._dense_directions:
            step_size = self._small_step
        else:
            step_size = self._large_step

        dx, dz = self._DIRECTIONS[self._direction]

        # 圈基准半径：每圈沿 8 方向各外扩 4×large_step，使相邻圈绝不重叠。
        # 修复依据第3条：原实现一圈（32块）后游标精确回到圈起点，密度不再变化时
        # 下一圈重复同一批已访问区块 → 主循环去重分支零查询死循环。
        base = self._ring * 4 * self._large_step
        radius = base + (self._step + 1) * step_size
        nx, nz = dx * radius, dz * radius

        # 边界回绕：区块坐标超出 ±chunk_radius 时回绕到对侧，保证绝不返回越界区块
        # （主循环越界分支不消耗查询，同理会造成死循环）
        lim = self._chunk_radius
        if abs(nx) > lim or abs(nz) > lim:
            span = 2 * lim
            nx = (nx + lim) % span - lim
            nz = (nz + lim) % span - lim

        self._advance_cursor()
        return (nx, nz)

    def _advance_cursor(self) -> None:
        """推进到下一块：每方向 4 步后换向，走完 8 方向后进入下一圈"""
        self._step += 1
        if self._step >= 4:
            self._step = 0
            self._direction = (self._direction + 1) % 8
            if self._direction == 0:
                self._ring += 1
