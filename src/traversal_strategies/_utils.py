# src/traversal_strategies/_utils.py
"""
Common utility functions for micro-strategies, to avoid code duplication
(Req. Sec. 9: extract the common parts shared by grid_3x3.py and adaptive_sampling.py)
"""
from typing import List, Tuple


def chunk_bounds(chunk_x: int, chunk_z: int) -> Tuple[int, int, int, int]:
    """
    Return the world-coordinate bounds of a chunk (x_min, z_min, x_max, z_max)
    A chunk is 16 cells per side; the coordinate range is chunk*16 to chunk*16+15
    """
    x0, z0 = chunk_x * 16, chunk_z * 16
    return x0, z0, x0 + 15, z0 + 15


def chunk_center(chunk_x: int, chunk_z: int) -> Tuple[int, int]:
    """return the world coordinates of a chunk's center (consistent with the point-picking method of the 9 micro-strategies and the reference implementation)"""
    return chunk_x * 16 + 8, chunk_z * 16 + 8


def generate_grid_points(chunk_x: int, chunk_z: int,
                         rows: int = 3, cols: int = 3) -> List[Tuple[int, int]]:
    """
    Generate uniform grid coordinates inside a chunk
    Args:
        chunk_x/chunk_z: chunk coordinates
        rows/cols: number of grid rows/columns (default 3x3)
    Returns:
        list of grid coordinates, ordered by "exploration priority descending" — i.e., the closer to the chunk center, the earlier,
        the first element is the highest-priority candidate (satisfying base_strategy's interface constraint)
    """
    x0, z0, x1, z1 = chunk_bounds(chunk_x, chunk_z)
    cx, cz = chunk_center(chunk_x, chunk_z)

    points = []
    # prevent division by 0 when rows=1 or cols=1
    row_div = max(rows - 1, 1)
    col_div = max(cols - 1, 1)

    for i in range(rows):
        for j in range(cols):
            x = x0 + int((x1 - x0) * i / row_div)
            z = z0 + int((z1 - z0) * j / col_div)
            points.append((x, z))

    # sort by Manhattan distance ascending => center point first => exploration priority descending
    points.sort(key=lambda p: abs(p[0] - cx) + abs(p[1] - cz))
    return points


def generate_corner_points(chunk_x: int, chunk_z: int) -> List[Tuple[int, int]]:
    """generate the four corner coordinates of a chunk, ordered by clockwise priority"""
    x0, z0, x1, z1 = chunk_bounds(chunk_x, chunk_z)
    return [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]
