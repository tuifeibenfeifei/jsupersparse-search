"""
random_sampling.py
[Random baseline] pure random sampling: randomly pick chunks within the search range and query them.

Req. Sec. 11, L4:
    · standalone algorithm, does not depend on main_algorithm.py / L1 strategy / reward modules
    · depends only on L0 (cubiomes_wrapper + database)
    · return format is identical to main_algorithm.run_single_episode()

Differences from the main framework (as a "no-prior" control):
    · no spatial continuity / temperature guidance / learning signal; each jump is independently random
    · each chunk fixes its center-point query (one temperature probe + one deep-sea determination, 2 black-box queries total)
"""

import random
from typing import Dict, List

from ._common import make_env, within_radius, query_chunk_center, build_result


def run_random_sampling(config: dict, seed: int) -> dict:
    """
    Run one pure random-sampling baseline episode.

    Args:
        config: config dict; reads search_radius / max_queries_per_episode /
                target_deep_sea_points / deep_sea_distance_threshold / temperature_enabled
        seed: world seed; algorithm randomness is driven by config["algo_seed"] (defaults to seed),
              ensuring that repeated runs with the same world seed differ (separating environment / algorithm randomness)
    Returns:
        result dict (format identical to run_single_episode)
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
        # randomly pick an unvisited chunk (cap retries to prevent an infinite loop in extreme cases)
        for _ in range(64):
            cx = rng.randint(-max_chunk, max_chunk)
            cz = rng.randint(-max_chunk, max_chunk)
            if (cx, cz) not in visited:
                break
        if (cx, cz) in visited:
            break   # after 64 retries still no new chunk (almost impossible), treat as fully swept

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
