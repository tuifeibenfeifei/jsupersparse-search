"""
bfs.py
[BFS baseline] breadth-first search: starting from the spawn point (0,0), systematically scan outward ring by ring along the 8-neighborhood.

Req. Sec. 11, L4:
    · standalone algorithm, does not depend on main_algorithm.py / L1 strategy / reward modules
    · depends only on L0 (cubiomes_wrapper + database)
    · return format is identical to main_algorithm.run_single_episode()

Differences from the main framework (as a "no-intelligence" control):
    · no temperature guidance, no Q-learning, no reward signal; pure geometric coverage
    · each chunk fixes its center-point query (one temperature probe + one deep-sea determination, 2 black-box queries total)
"""

from collections import deque
from typing import Dict, List

from ._common import (make_env, within_radius, query_chunk_center,
                      build_result, DIRECTION_DELTAS)


def run_bfs(config: dict, seed: int) -> dict:
    """
    Run one BFS baseline episode.

    Args:
        config: config dict; reads search_radius / max_queries_per_episode /
                target_deep_sea_points / deep_sea_distance_threshold / temperature_enabled
        seed: world seed (cubiomes terrain seed)
    Returns:
        result dict (format identical to run_single_episode)
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

        # enqueue the 8-neighborhood (breadth-first: finish nearby cells before expanding outward)
        for dx, dz in DIRECTION_DELTAS:
            nx, nz = cx + dx, cz + dz
            if (nx, nz) not in visited:
                visited.add((nx, nz))
                frontier.append((nx, nz))

    return build_result(seed, wrapper, db, convergence,
                        temperature_history, temperature_samples,
                        visited_chunks=len(visited),
                        temperature_probes=temperature_probes)
