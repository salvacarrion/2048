"""Baselines — the simplest possible players, your reference line."""
from .greedy import GreedyStrategy
from .manual import ManualStrategy
from .random import RandomStrategy

__all__ = ["RandomStrategy", "GreedyStrategy", "ManualStrategy"]
