"""
go_explore.py
[Go-Explore baseline] Cell archive + coverage-driven hierarchical exploration.

Core idea (simplified adaptation of Ecoffet et al., 2021):
    1. Stage 1 "exploration": store all visited chunks in an archive,
       upon encountering a deep-sea point, mark its neighborhood as a high-priority expansion target (advancing along the edge of the known deep-sea region);
    2. Stage 2 "return and exploit": periodically return to a random high-value cell in the archive and continue expanding from there,
       avoiding getting stuck in a local area along a single path, combining large-scale coverage with long-horizon transitions.

Req. Sec. 11 L4:
    - Standalone algorithm, does not depend on main_algorithm.py / L1 strategies / reward modules
    - Depends only on L0 (cubiomes_wrapper + database)
    - Return format is consistent with main_algorithm.run_single_episode()

Differences from the main framework (as a "no learning signal" control):
    - No temperature guidance, no Q-learning, no reward weights; relies only on the geometric prior of "archive + neighborhood expansion"
    - Fixed center-point query per chunk (1 temperature probe + 1 deep-sea determination, 2 black-box queries total)
"""

import random
from typing import Dict, List

from ._common import (make_env, within_radius, query_chunk_center,
                      build_result, DIRECTION_DELTAS)


def run_go_explore(config: dict, seed: int) -> dict:
    """
    Run one Go-Explore baseline episode.

    Args:
        config: configuration dict, reads search_radius / max_queries_per_episode /
                target_deep_sea_points / deep_sea_distance_threshold /
                temperature_enabled / go_explore_return_prob / go_explore_expand_step
        seed: world seed; algorithmic randomness is driven by config["algo_seed"] (defaults to seed)
    Returns:
        Result dict (format consistent with run_single_episode)
    """
    wrapper, db = make_env(config, seed)
    radius = int(config.get("search_radius", 10000))
    max_queries = int(config.get("max_queries_per_episode", 10000))
    target = int(config.get("target_deep_sea_points", 40))
    temp_enabled = bool(config.get("temperature_enabled", True))

    # Parameters (config-driven; hardcoding forbidden; defaults are fallback only when YAML is missing)
    return_prob = float(config.get("go_explore_return_prob", 0.15))   # per-step probability of returning to archive
    expand_step = int(config.get("go_explore_expand_step", 2))        # deep-sea point neighborhood expansion step size (chunks)

    algo_seed = int(config.get("algo_seed", seed))
    rng = random.Random((seed * 1000003 + algo_seed) & 0xFFFFFFFF)

    archive = {(0, 0)}                 # all visited chunks (coverage archive)
    high_value = [(0, 0)]              # high-value cells (deep-sea point neighborhood expansion targets)
    visited = set()
    current = (0, 0)
    convergence: List[dict] = []
    temperature_history: List[float] = []
    temperature_samples: List[tuple] = []
    temperature_probes = 0
    last_point_chunk = None

    while wrapper.query_count < max_queries and len(db) < target:
        # --- Stage decision: return to archive / neighborhood expansion / random frontier expansion ---
        if high_value and rng.random() < return_prob:
            current = rng.choice(high_value)
        elif last_point_chunk is not None and rng.random() < 0.6:
            # Prefer expanding around the nearest deep-sea anchor (edge-following advance)
            px, pz = last_point_chunk
            dx, dz = rng.choice(DIRECTION_DELTAS)
            current = (px + dx * expand_step, pz + dz * expand_step)
        else:
            # Coverage frontier expansion: randomly jump near the archive edge
            max_chunk = radius // 16
            current = (rng.randint(-max_chunk, max_chunk),
                       rng.randint(-max_chunk, max_chunk))

        if not within_radius(current[0], current[1], radius):
            current = (0, 0)
        if current in visited:
            # Repeat visit: does not consume query budget, proceed to next decision round (consistent with main framework dedup semantics)
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
            # Deep-sea point hit -> enqueue 8-neighborhood as high-value cells (expanding along the deep-sea region edge)
            for dx, dz in DIRECTION_DELTAS:
                nxt = (cx + dx * expand_step, cz + dz * expand_step)
                if within_radius(nxt[0], nxt[1], radius):
                    high_value.append(nxt)

    return build_result(seed, wrapper, db, convergence,
                        temperature_history, temperature_samples,
                        visited_chunks=len(visited),
                        temperature_probes=temperature_probes)
