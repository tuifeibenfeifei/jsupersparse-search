"""traversal_strategies/__init__.py - 遍历策略插件包"""
from .base_strategy import BaseStrategy
from .center_point import CenterPointStrategy as CenterPoint
from .corner_points import CornerPointsStrategy as CornerPoints
from .random_two_points import RandomTwoPointsStrategy as RandomTwoPoints
from .grid_3x3 import Grid3x3Strategy as Grid3x3
from .corners_and_center import CornersAndCenterStrategy as CornersAndCenter
from .any_three_corners import AnyThreeCornersStrategy as AnyThreeCorners
from .t_shape import TShapeStrategy as TShape
from .adaptive_sampling import AdaptiveSamplingStrategy as AdaptiveSampling
from .diagonal_two_points import DiagonalTwoPointsStrategy as DiagonalTwoPoints

__all__ = [
    "BaseStrategy",
    "CenterPoint",
    "CornerPoints",
    "RandomTwoPoints",
    "Grid3x3",
    "CornersAndCenter",
    "AnyThreeCorners",
    "TShape",
    "AdaptiveSampling",
    "DiagonalTwoPoints",
]