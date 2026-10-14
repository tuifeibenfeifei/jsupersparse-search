"""
go_explore.py
[Go-Explore 基线] 单元档案 + 覆盖驱动的分层探索。

核心思想（Ecoffet et al., 2021 的简化适配版）：
    1. 第一阶段"探索"：把访问过的所有区块存入档案（archive），
       遇到深海点后，将其邻域标记为高优先级扩展目标（紧贴已知深海区外缘推进）；
    2. 第二阶段"回退利用"：定期随机回到档案中的某个高价值单元，从该处继续扩展，
       避免单条路径陷入局部区域，实现大范围覆盖与长程转移的结合。

需求.txt 第十一节 L4：
    · 独立算法，不依赖 main_algorithm.py / L1 策略 / 奖励模块
    · 只依赖 L0（cubiomes_wrapper + database）
    · 返回格式与 main_algorithm.run_single_episode() 保持一致

与主框架的差异（作为"无学习信号"对照）：
    · 无温度导向、无 Q-learning、无奖励权重，仅靠"档案 + 邻域扩展"的几何先验
    · 每区块固定查询中心点（温度探针 + 深海判定各 1 次，共 2 次黑盒查询）
"""

import random
from typing import Dict, List

from ._common import (make_env, within_radius, query_chunk_center,
                      build_result, DIRECTION_DELTAS)


def run_go_explore(config: dict, seed: int) -> dict:
    """
    执行一次 Go-Explore 基线 episode。

    Args:
        config: 配置字典，读取 search_radius / max_queries_per_episode /
                target_deep_sea_points / deep_sea_distance_threshold /
                temperature_enabled / go_explore_return_prob / go_explore_expand_step
        seed: 世界种子；算法随机性由 config["algo_seed"]（缺省取 seed）驱动
    Returns:
        结果字典（格式与 run_single_episode 一致）
    """
    wrapper, db = make_env(config, seed)
    radius = int(config.get("search_radius", 10000))
    max_queries = int(config.get("max_queries_per_episode", 10000))
    target = int(config.get("target_deep_sea_points", 40))
    temp_enabled = bool(config.get("temperature_enabled", True))

    # 参数（配置驱动，严禁硬编码；缺省值仅为 YAML 缺失时的兜底）
    return_prob = float(config.get("go_explore_return_prob", 0.15))   # 每步回退档案概率
    expand_step = int(config.get("go_explore_expand_step", 2))        # 深海点邻域扩展步长（区块）

    algo_seed = int(config.get("algo_seed", seed))
    rng = random.Random((seed * 1000003 + algo_seed) & 0xFFFFFFFF)

    archive = {(0, 0)}                 # 全部访问过的区块（覆盖档案）
    high_value = [(0, 0)]              # 高价值单元（深海点邻域扩展目标）
    visited = set()
    current = (0, 0)
    convergence: List[dict] = []
    temperature_history: List[float] = []
    temperature_samples: List[tuple] = []
    temperature_probes = 0
    last_point_chunk = None

    while wrapper.query_count < max_queries and len(db) < target:
        # --- 阶段决策：回退档案 / 邻域扩展 / 随机开疆 ---
        if high_value and rng.random() < return_prob:
            current = rng.choice(high_value)
        elif last_point_chunk is not None and rng.random() < 0.6:
            # 优先围绕最近深海锚点外扩（贴边推进）
            px, pz = last_point_chunk
            dx, dz = rng.choice(DIRECTION_DELTAS)
            current = (px + dx * expand_step, pz + dz * expand_step)
        else:
            # 覆盖开疆：随机跳向档案边缘附近
            max_chunk = radius // 16
            current = (rng.randint(-max_chunk, max_chunk),
                       rng.randint(-max_chunk, max_chunk))

        if not within_radius(current[0], current[1], radius):
            current = (0, 0)
        if current in visited:
            # 重复访问：不消耗查询预算，继续下一轮决策（与主框架去重语义一致）
            continue

        visited.add(current)
        archive.add(current)
        cx, cz = current

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

        if found:
            last_point_chunk = (cx, cz)
            # 命中深海点 → 8 邻域入列高价值单元（紧贴深海区外缘扩展）
            for dx, dz in DIRECTION_DELTAS:
                nxt = (cx + dx * expand_step, cz + dz * expand_step)
                if within_radius(nxt[0], nxt[1], radius):
                    high_value.append(nxt)

    return build_result(seed, wrapper, db, convergence,
                        temperature_history, temperature_samples,
                        visited_chunks=len(visited),
                        temperature_probes=temperature_probes)
