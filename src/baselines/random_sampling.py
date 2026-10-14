"""
random_sampling.py
[Random 基线] 纯随机采样：在搜索范围内随机抽取区块进行查询。

需求.txt 第十一节 L4：
    · 独立算法，不依赖 main_algorithm.py / L1 策略 / 奖励模块
    · 只依赖 L0（cubiomes_wrapper + database）
    · 返回格式与 main_algorithm.run_single_episode() 保持一致

与主框架的差异（作为"无先验"对照）：
    · 无任何空间连续性 / 温度导向 / 学习信号，每次跳跃独立随机
    · 每区块固定查询中心点（温度探针 + 深海判定各 1 次，共 2 次黑盒查询）
"""

import random
from typing import Dict, List

from ._common import make_env, within_radius, query_chunk_center, build_result


def run_random_sampling(config: dict, seed: int) -> dict:
    """
    执行一次纯随机采样基线 episode。

    Args:
        config: 配置字典，读取 search_radius / max_queries_per_episode /
                target_deep_sea_points / deep_sea_distance_threshold / temperature_enabled
        seed: 世界种子；算法随机性由 config["algo_seed"]（缺省取 seed）驱动，
              保证同一世界种子的多次重复结果不同（分离环境/算法随机性）
    Returns:
        结果字典（格式与 run_single_episode 一致）
    """
    wrapper, db = make_env(config, seed)
    radius = int(config.get("search_radius", 10000))
    max_queries = int(config.get("max_queries_per_episode", 10000))
    target = int(config.get("target_deep_sea_points", 40))
    temp_enabled = bool(config.get("temperature_enabled", True))

    algo_seed = int(config.get("algo_seed", seed))
    rng = random.Random((seed * 1000003 + algo_seed) & 0xFFFFFFFF)

    max_chunk = radius // 16
    visited = set()
    convergence: List[dict] = []
    temperature_history: List[float] = []
    temperature_samples: List[tuple] = []
    temperature_probes = 0

    while wrapper.query_count < max_queries and len(db) < target:
        # 随机抽取一个未访问区块（重试次数封顶，防止极端情况下死循环）
        for _ in range(64):
            cx = rng.randint(-max_chunk, max_chunk)
            cz = rng.randint(-max_chunk, max_chunk)
            if (cx, cz) not in visited:
                break
        if (cx, cz) in visited:
            break   # 64 次重试仍无新区块（几乎不可能），视为已扫尽

        visited.add((cx, cz))
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

    return build_result(seed, wrapper, db, convergence,
                        temperature_history, temperature_samples,
                        visited_chunks=len(visited),
                        temperature_probes=temperature_probes)
