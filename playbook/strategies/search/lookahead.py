"""Expectimax on top of a *learned* afterstate value, for a whole batch of boards.

The learning strategies (``ntuple``, ``dqn``) learn ``V(afterstate)`` and play
greedily by ``Q(s, a) = r(s, a) + V(afterstate(s, a))``. Searching a little
before trusting ``V`` makes them markedly stronger: with ``depth=1`` each
afterstate is scored by the average, over every possible spawn, of the best
``Q`` from the resulting board.

Unlike the tree searches next door, this one works level by level on numpy
batches (via :mod:`playbook.game.vector`), so a neural ``value_fn`` scores every
leaf of the tree in a single forward pass.
"""
import numpy as np

from ...game.vector import expand, spawn_children


def expectimax_q(boards, value_fn, depth=0):
    """``Q(s, a)`` for every board and move; ``-inf`` where a move is illegal.

    ``value_fn`` maps a ``(M, 4, 4)`` batch of afterstates to ``(M,)`` values (in
    points). ``depth=0`` is the greedy ``r + V(afterstate)``; each extra level
    averages over the random spawn and maximizes over the next move.
    """
    after, rewards, legal = expand(boards)
    rows, moves = np.nonzero(legal)
    legal_after = after[rows, moves]
    values = np.zeros(legal.shape)
    if len(legal_after):
        values[rows, moves] = (value_fn(legal_after) if depth == 0
                               else _chance(legal_after, value_fn, depth))
    return np.where(legal, rewards + values, -np.inf)


def _chance(afterstates, value_fn, depth):
    """Average, over every possible spawn, of the best move one level deeper."""
    children, owner, prob = spawn_children(afterstates)
    best = expectimax_q(children, value_fn, depth - 1).max(axis=1)
    best[np.isneginf(best)] = 0.0          # game over: nothing more to score
    return np.bincount(owner, weights=prob * best, minlength=len(afterstates))
