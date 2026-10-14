"""
bfs.py
[BFS 基线] 广度优先搜索：从出生点 (0,0) 出发，按 8 邻域逐环向外系统扫描。

需求.txt 第十一节 L4：
    · 独立算法，不依赖 main_algorithm.py / L1 策略 / 奖励模块
    · 只依赖 L0（cubiomes_wrapper + database）
    · 返回格式与 main_algorithm.run_single_episode() 保持一致

与主框架的差异（作为"无智能"对照）：
    · 无温度导向、无 Q-learning、无奖励信号，纯几何覆盖
    · 每区块固定查询中心点（温度探针 + 深海判定各 1 次，共 2 次黑盒查询）
"""

from collections import deque
from typing import Dict, List

from ._common import (make_env, within_radius, query_chunk_center,
                      build_result, DIRECTION_DELTAS)


def run_bfs(config: dict, seed: int) -> dict:
    """
    执行一次 BFS 基线 episode。

    Args:
        config: 配置字典，读取 search_radius / max_queries_per_episode /
                target_deep_sea_points / deep_sea_distance_threshold / temperature_enabled
        seed: 世界种子（cubiomes 地形种子）
    Returns:
        结果字典（格式与 run_single_episode 一致）
    """
    wrapper, db = make_env(config, seed)
    radius = int(config.get("search_radius", 10000))
    max_queries = int(config.get("max_queries_per_episode", 10000))
    target = int(config.get("target_deep_sea_points", 40))
    temp_enabled = bool(config.get("temperature_enabled", True))

    visited = {(0, 0)}
    frontier = deque([(0, 0)])
    convergence: List[dict] = []
    temperature_history: List[float] = []
    temperature_samples: List[tuple] = []
    temperature_probes = 0

    while (wrapper.query_count < max_queries and len(db) < target
           and frontier):
        cx, cz = frontier.popleft()
        if not within_radius(cx, cz, radius):
            continue

        temperature_probes += 1
        _, _, found = query_chunk_center(
            wrapper, db, config, cx, cz,
            temperature_history=temperature_history,
            temperature_samples=temperature_samples,
            temperature_enabled=temp_enabled,
        )
        convergence.append({
            "query_count": wrapper.query_count,
            "deep_sea_points": len(db),
            "stage": 1,
        })

        # 8 邻域入队（广度优先：先扫完近处再向外扩散）
        for dx, dz in DIRECTION_DELTAS:
            nx, nz = cx + dx, cz + dz
            if (nx, nz) not in visited:
                visited.add((nx, nz))
                frontier.append((nx, nz))

    return build_result(seed, wrapper, db, convergence,
                        temperature_history, temperature_samples,
                        visited_chunks=len(visited),
                        temperature_probes=temperature_probes)
