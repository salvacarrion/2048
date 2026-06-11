"""Players, grouped by technique. See each sub-package's README for the idea
behind the family. Every player implements :class:`~playbook.strategies.base.Strategy`.
"""
from .base import Strategy, Trainable

__all__ = ["Strategy", "Trainable"]
