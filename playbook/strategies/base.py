"""The one interface every player implements.

A strategy decides a move given the current board and the set of legal moves.
That is the whole contract — copy the closest existing strategy and rewrite
:meth:`Strategy.select_move` to try a new idea.

Strategies that *learn* additionally mix in :class:`Trainable`.
"""
from abc import ABC, abstractmethod


class Strategy(ABC):
    #: short, human-friendly name (used in reports and the registry)
    name = "strategy"

    #: Per-move scores recorded by the most recent :meth:`select_move`, as a
    #: ``{Move: float}`` mapping — the *why* behind the chosen move, surfaced by
    #: the live ``play`` view (``--explain``). Strategies that can explain
    #: themselves (search, ``greedy``, ``ntuple``) fill this in; ``None`` means
    #: no explanation is available (e.g. ``random``).
    last_scores = None

    @abstractmethod
    def select_move(self, board, legal):
        """Return the chosen :class:`~playbook.game.board.Move`.

        ``board`` is a 4x4 numpy array of log2 exponents; ``legal`` is the set
        of moves that actually change the board (never empty unless the game is
        over, in which case the runner stops before calling this).
        """

    def reset(self):
        """Hook called at the start of each game. Default: no-op."""


class Trainable(ABC):
    """Mixin for strategies that improve from experience.

    The training loop and checkpoint format are up to each implementation; this
    just marks the capability and documents the expected methods.
    """

    @abstractmethod
    def observe(self, transition):
        """Update from a single :class:`~playbook.game.env` transition."""

    def train(self, env, episodes, **kwargs):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    @classmethod
    def load(cls, path):
        raise NotImplementedError
