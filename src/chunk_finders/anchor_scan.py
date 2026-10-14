"""
anchor_scan.py
[宏观] 锚点重定位扫描：发现深海点后，以该点为新锚点向外 BFS 扩散。

对应需求：
  · 第一节第4条：发现一个深海点后，其周围阈值范围内视为同一生成网格区域，
                 后续搜索应从该点向外继续。
  · 第二节：若当前区块按选点策略查询后均非深海点，则跳转到一定距离外继续搜索。
对应论文创新点3：master-slave 协同——BFS 保证覆盖完整性 + RL 指导方向。
  本 finder 负责 Stage1/2 的 BFS 式系统覆盖（贴着已知深海区外缘推进），
  Stage3 由 Q-learning + 温度导向偏向接管方向决策，二者构成主从协同。

为什么有效（实测依据）：
  深海场空间自相关在 lag=416格(阈值) 仍达 0.23，聚集性强；
  故从已知深海锚点向外缘扩散，命中率远高于从原点盲目辐射。
  这打破了 ray_scan 的预算瓶颈——后者发现深海点后仍以原点为中心辐射，
  大量查询浪费在已探明区域，预算耗尽也凑不满 40 点。
"""

from collections import deque
from typing import Tuple, List, Set

from .base_finder import BaseFinder


class AnchorScanFinder(BaseFinder):
    """
    锚点重定位扫描（BFS 覆盖 + 射线开疆）

    工作机制：
        1. BFS frontier 队列：每当发现新深海锚点，向 8 方向、距离≈间距阈值处
           派生探索起点入队，使搜索始终贴着已知深海区外缘系统性推进。
        2. 回退射线扫描：frontier 为空时，以分层射线（同 ray_scan）开疆拓土，
           寻找新的深海锚点。
        3. 锚点去重：update_deep_sea_density 每次收到全量深海点，仅对新锚点派生 frontier。
    """

    # 8方向，顺序与 main_algorithm._DIRECTION_DELTAS / RayScanFinder._DIRECTIONS 一致
    _DIRECTIONS: List[Tuple[int, int]] = [
        (1, 0), (1, -1), (0, -1), (-1, -1),
        (-1, 0), (-1, 1), (0, 1), (1, 1),
    ]

    # 射线回退的步长档位缺省值，仅在 config 缺失 action_steps 时兜底
    _DEFAULT_STEPS: List[int] = [1, 2, 4, 8]

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        # 间距阈值(格) -> 区块；派生跳跃距离取略大于阈值，确保跳出排斥区(需求第一节第4条)
        threshold_blocks = self.config.get("deep_sea_distance_threshold", 416)
        self._jump_chunks = max(int(threshold_blocks) // 16 + 1, 1)

        # 射线回退步长档位由配置注入，严禁硬编码（需求第十节）
        steps = self.config.get("action_steps") or self._DEFAULT_STEPS
        self._ray_steps: List[int] = [int(s) for s in steps]

        # frontier 容量上限：防止极端情况下队列无限增长（深海点上限40 x 8方向 = 320）
        self._max_frontier = int(self.config.get("anchor_max_frontier", 4096))

        self.reset()

    @property
    def name(self) -> str:
        return "anchor_scan"

    def reset(self) -> None:
        """重置全部内部状态，确保每个 episode 从原点重新开始（需求第一节第3条）"""
        self._frontier: deque = deque()
        self._anchors: Set[Tuple[int, int]] = set()
        # 回退射线扫描游标
        self._ray_dir = 0
        self._ray_step_idx = 0
        self._ray_layer = 1
        self._emitted_origin = False

    def update_deep_sea_density(self, chunks: List[Tuple[int, int]]) -> None:
        """
        外部注入已发现深海点的区块坐标（由 main_algorithm._sync_finder_density 调用）。
        对每个【新】锚点，向 8 方向、距离≈阈值处派生 BFS 探索起点。
        Args:
            chunks: 全量深海点区块坐标列表 [(cx, cz), ...]
        """
        for anchor in chunks:
            ax, az = anchor
            if (ax, az) in self._anchors:
                continue
            self._anchors.add((ax, az))
            for dx, dz in self._DIRECTIONS:
                if len(self._frontier) >= self._max_frontier:
                    break
                self._frontier.append(
                    (ax + dx * self._jump_chunks, az + dz * self._jump_chunks)
                )

    def _next_ray_chunk(self) -> Tuple[int, int]:
        """frontier 空时的回退：分层射线扫描开疆拓土（逻辑同 ray_scan）"""
        if not self._emitted_origin:
            self._emitted_origin = True
            return (0, 0)

        dx, dz = self._DIRECTIONS[self._ray_dir]
        step = self._ray_steps[self._ray_step_idx % len(self._ray_steps)] * self._ray_layer
        nx, nz = dx * step, dz * step

        # 推进：步长(内层) -> 方向(中层) -> 层数(外层)
        self._ray_step_idx += 1
        if self._ray_step_idx % len(self._ray_steps) == 0:
            self._ray_dir += 1
            if self._ray_dir >= len(self._DIRECTIONS):
                self._ray_dir = 0
                self._ray_layer += 1
        return (nx, nz)

    def get_next_chunk(self, current_x: int, current_z: int) -> Tuple[int, int]:
        """
        返回下一个待查区块：优先 BFS frontier（贴着已知深海区外缘系统推进），
        frontier 空时回退到射线扫描开疆。
        重复/已访问区块由 main_algorithm 的 _visited_chunks 去重，此处不重复判断。
        """
        if self._frontier:
            return self._frontier.popleft()
        return self._next_ray_chunk()

    @property
    def anchor_count(self) -> int:
        """当前已登记的深海锚点数（供调试与收敛分析）"""
        return len(self._anchors)

    @property
    def frontier_size(self) -> int:
        """当前 BFS 待探索队列长度"""
        return len(self._frontier)
