"""N-tuple network trained by temporal-difference learning.

This is the classic, lightweight reinforcement-learning approach for 2048
(Szubert & Jaśkowski, 2014) — no neural net, just lookup tables, and it learns
strong play on a CPU. It is the canonical example of *value-based RL on
afterstates*:

  * A board is scored by summing small lookup tables, one per "tuple" (a fixed
    group of cells). Each table maps the tile pattern in those cells to a value.
  * The agent plays greedily w.r.t. ``reward(s, a) + V(afterstate)``.
  * After each move it nudges ``V(afterstate)`` toward the value it actually saw
    next — TD(0) on afterstates.

Afterstates (the board *after* the slide/merge but *before* the random spawn)
are exactly what :class:`~playbook.game.env.SimEnv` exposes in ``info``.
"""
import random

import numpy as np

from playbook.game.board import simulate_move
from playbook.strategies.base import Strategy, Trainable

MAX_EXP = 16  # tile exponents are clipped to this many distinct values


def default_tuples():
    """A simple, didactic tuple set: every row, every column, every 2x2 square."""
    rows = [[(r, c) for c in range(4)] for r in range(4)]
    cols = [[(r, c) for r in range(4)] for c in range(4)]
    squares = [[(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
               for r in range(3) for c in range(3)]
    return rows + cols + squares


class NTupleNetwork:
    """Sum of lookup tables, one per tuple. The whole value function."""

    def __init__(self, tuples=None, max_exp=MAX_EXP):
        self.tuples = [tuple(t) for t in (tuples or default_tuples())]
        self.max_exp = max_exp
        self.tables = [np.zeros(max_exp ** len(t), dtype=np.float64) for t in self.tuples]

    def _index(self, board, k):
        idx, stride = 0, 1
        for i, j in self.tuples[k]:
            exp = min(int(board[i][j]), self.max_exp - 1)
            idx += exp * stride
            stride *= self.max_exp
        return idx

    def value(self, board):
        return float(sum(self.tables[k][self._index(board, k)] for k in range(len(self.tuples))))

    def update(self, board, delta):
        """Distribute a value correction across the active table entries."""
        per = delta / len(self.tables)
        for k in range(len(self.tables)):
            self.tables[k][self._index(board, k)] += per

    def save(self, path):
        np.savez(path, tuples=np.array(self.tuples, dtype=object),
                 max_exp=self.max_exp, *self.tables)

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=True)
        net = cls(tuples=list(data["tuples"]), max_exp=int(data["max_exp"]))
        net.tables = [data[f"arr_{i}"] for i in range(len(net.tuples))]
        return net


class NTupleStrategy(Strategy, Trainable):
    name = "ntuple"

    def __init__(self, tuples=None, alpha=0.1, net=None, seed=None):
        self.net = net if net is not None else NTupleNetwork(tuples)
        self.alpha = alpha
        self.rng = random.Random(seed)

    # -- playing -------------------------------------------------------------
    def _best(self, board, legal):
        """Greedy move by ``reward + V(afterstate)``.

        Returns ``(move, afterstate, reward)`` for the chosen move.
        """
        best = None  # (val, move, afterstate, reward)
        for move in sorted(legal):
            after, _, reward = simulate_move(board, move)
            val = reward + self.net.value(after)
            if best is None or val > best[0]:
                best = (val, move, after, reward)
        return best[1], best[2], best[3]

    def select_move(self, board, legal):
        move, _, _ = self._best(board, legal)
        return move

    # -- learning ------------------------------------------------------------
    def observe(self, transition):
        """Single TD(0) afterstate update (online API). See :meth:`train`."""
        if transition.done:
            target = 0.0
        else:
            target = transition.reward + self.net.value(transition.next_state)
        delta = self.alpha * (target - self.net.value(transition.afterstate))
        self.net.update(transition.afterstate, delta)

    def _train_episode(self, env):
        board = env.reset()
        while True:
            legal = env.legal_moves()
            if not legal:
                break
            move, after, _ = self._best(board, legal)
            next_board, _, done, _ = env.step(move)

            next_legal = env.legal_moves()
            if done or not next_legal:
                target = 0.0
            else:
                # value of this afterstate = next reward + value of the following afterstate
                _, next_after, next_reward = self._best(next_board, next_legal)
                target = next_reward + self.net.value(next_after)
            delta = self.alpha * (target - self.net.value(after))
            self.net.update(after, delta)

            board = next_board
            if done:
                break
        return env.score

    def train(self, env, episodes=10_000, verbose=False, report_every=500):
        scores = []
        for ep in range(1, episodes + 1):
            scores.append(self._train_episode(env))
            if verbose and ep % report_every == 0:
                window = scores[-report_every:]
                print(f"  episode {ep:>6}  avg score (last {report_every}): "
                      f"{sum(window) / len(window):.0f}")
        return scores

    # -- persistence ---------------------------------------------------------
    def save(self, path):
        self.net.save(path)

    @classmethod
    def load(cls, path):
        return cls(net=NTupleNetwork.load(path))
