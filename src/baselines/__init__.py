"""src/baselines/__init__.py - baseline algorithm plugin package (L4 independent comparison algorithms)."""
from .bfs import run_bfs
from .go_explore import run_go_explore
from .icm import run_icm
from .random_sampling import run_random_sampling
from .rnd import run_rnd

__all__ = ["run_bfs", "run_go_explore", "run_icm",
           "run_random_sampling", "run_rnd"]
