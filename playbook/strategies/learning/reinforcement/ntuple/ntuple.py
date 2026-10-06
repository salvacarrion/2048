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

from playbook.game.board import Move, simulate_move
from playbook.game.rules import legal_moves
from playbook.strategies.base import Strategy, Trainable
from playbook.strategies.search.lookahead import expectimax_q

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

    def values(self, boards):
        """:meth:`value` of a ``(N, 4, 4)`` batch at once (numpy, no Python loop
        over boards) -- what batched search and data collection use."""
        if not hasattr(self, "_cells"):
            self._cells = np.array([[i * 4 + j for i, j in t] for t in self.tuples])
            self._strides = self.max_exp ** np.arange(self._cells.shape[1])
        flat = np.asarray(boards, dtype=np.int64).reshape(len(boards), 16)
        idx = (np.minimum(flat, self.max_exp - 1)[:, self._cells] * self._strides).sum(axis=2)
        return sum(table[idx[:, k]] for k, table in enumerate(self.tables))

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

    def __init__(self, tuples=None, alpha=0.1, net=None, depth=0, seed=None):
        self.net = net if net is not None else NTupleNetwork(tuples)
        self.alpha = alpha
        self.depth = depth   # expectimax levels on top of V when playing (0 = greedy)
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

    def move_values(self, boards):
        """``reward + V(afterstate)`` of every move (``-inf`` if illegal) for a
        batch of boards, with ``depth`` levels of expectimax on top (see
        :mod:`~playbook.strategies.search.lookahead`)."""
        return expectimax_q(boards, self.net.values, self.depth)

    def select_moves(self, boards):
        return self.move_values(boards).argmax(axis=1)

    def select_move(self, board, legal):
        if self.depth:
            q = self.move_values(board[None])[0]
            self.last_scores = {Move(m): float(q[m]) for m in legal}
            return max(sorted(legal), key=lambda m: q[m])
        # Greedy by reward + V(afterstate) — the same ranking _best uses while
        # training — recording each move's value so the live `play` view (and
        # --explain) can show why this move won.
        scores, best = {}, None  # best = (value, move)
        for move in sorted(legal):
            after, _, reward = simulate_move(board, move)
            value = reward + self.net.value(after)
            scores[move] = value
            if best is None or value > best[0]:
                best = (value, move)
        self.last_scores = scores
        return best[1]

    # -- learning ------------------------------------------------------------
    def observe(self, transition):
        """Single TD(0) afterstate update (online API): the same rule :meth:`train`
        applies after every move.

        The afterstate is worth what the board that followed it offered: the
        best ``reward + V(afterstate)`` from ``next_state`` (0 if that board is
        game over). ``transition.reward`` is *not* part of it -- those points were
        scored by the move that produced the afterstate, so they are already
        behind it.
        """
        target = 0.0
        legal = legal_moves(transition.next_state)
        if not transition.done and legal:
            _, next_after, next_reward = self._best(transition.next_state, legal)
            target = next_reward + self.net.value(next_after)
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
    def load(cls, path, **config):
        return cls(net=NTupleNetwork.load(path), **config)
