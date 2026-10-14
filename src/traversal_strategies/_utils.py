# src/traversal_strategies/_utils.py
"""
微观策略公共工具函数，避免代码重复
（需求第九节：把 grid_3x3.py 和 adaptive_sampling.py 共同部分挑出来）
"""
from typing import List, Tuple


def chunk_bounds(chunk_x: int, chunk_z: int) -> Tuple[int, int, int, int]:
    """
    返回区块的世界坐标边界 (x_min, z_min, x_max, z_max)
    区块边长为16格，坐标范围 chunk*16 ~ chunk*16+15
    """
    x0, z0 = chunk_x * 16, chunk_z * 16
    return x0, z0, x0 + 15, z0 + 15


def chunk_center(chunk_x: int, chunk_z: int) -> Tuple[int, int]:
    """返回区块中心点世界坐标（与 9 种微观策略及参考实现取点方式一致）"""
    return chunk_x * 16 + 8, chunk_z * 16 + 8


def generate_grid_points(chunk_x: int, chunk_z: int,
                         rows: int = 3, cols: int = 3) -> List[Tuple[int, int]]:
    """
    生成区块内均匀网格坐标
    Args:
        chunk_x/chunk_z: 区块坐标
        rows/cols: 网格行列数（默认3x3）
    Returns:
        网格坐标列表，按「探索优先级降序」排列 —— 即离区块中心越近越靠前，
        首个元素为优先级最高的候选点（符合 base_strategy 的接口约束）
    """
    x0, z0, x1, z1 = chunk_bounds(chunk_x, chunk_z)
    cx, cz = chunk_center(chunk_x, chunk_z)

    points = []
    # 防止 rows=1 或 cols=1 时除以 0
    row_div = max(rows - 1, 1)
    col_div = max(cols - 1, 1)

    for i in range(rows):
        for j in range(cols):
            x = x0 + int((x1 - x0) * i / row_div)
            z = z0 + int((z1 - z0) * j / col_div)
            points.append((x, z))

    # 按曼哈顿距离升序排序 => 中心点在前 => 探索优先级降序
    points.sort(key=lambda p: abs(p[0] - cx) + abs(p[1] - cz))
    return points


def generate_corner_points(chunk_x: int, chunk_z: int) -> List[Tuple[int, int]]:
    """生成区块四角坐标，按顺时针优先级排列"""
    x0, z0, x1, z1 = chunk_bounds(chunk_x, chunk_z)
    return [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]
