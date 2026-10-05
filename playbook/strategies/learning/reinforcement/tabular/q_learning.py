"""Tabular Q-learning: the simplest RL, and where it breaks.

A table ``Q[state, action]``, updated after every move by

    Q(s, a) += alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))

and played epsilon-greedily while learning.

The catch is the word *state*. 2048 has more boards than any table can hold, and
a table never generalizes: what it learns about one board says nothing about the
board next to it. So the table is indexed by an **abstraction**, a coarse summary
of the board:

  * ``"coarse"`` (default): whether the biggest tile sits in a corner, on an
    edge or inside, how many cells are free and, for each move, whether it is
    legal, merges a little or a lot, or drags the biggest tile out of place.
    About 13k possible keys, so they are visited often and the table does learn
    (about twice the score of random play) -- but very different boards share a
    key, so it plateaus below even the one-line ``greedy`` player.
  * ``"raw"``: the exact board. The table keeps growing (a new entry for nearly
    every board it meets) and never learns anything useful.

That trade-off -- generalize badly or not at all -- is the lesson, and the reason
for the n-tuple network (many small tables over overlapping cell groups) and
deep RL (a network that generalizes by itself).

With so many different boards behind one key, the targets are very noisy, so by
default (``alpha=None``) the step size is ``1 / N(s, a)``: each entry is simply
the average of the targets it has seen, where a fixed step would never settle.
"""
import pickle
import random
from collections import defaultdict

import numpy as np

from playbook.game.board import Move, simulate_move
from playbook.game.env import Transition
from playbook.strategies.base import Strategy, Trainable


#: where a cell sits: 0 corner, 1 edge, 2 inside
_CELL_KIND = (0, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1, 0, 1, 1, 0)


def _move_outcome(board, move, biggest):
    """0 illegal, 1 slides only, 2 small merge, 3 big merge (> 16 points),
    4 drags the biggest tile off its cell."""
    after, _, reward = simulate_move(board, move)
    if np.array_equal(after, board):
        return 0
    if after.flat[biggest] < board.flat[biggest]:
        return 4
    if reward == 0:
        return 1
    return 2 if reward <= 16 else 3


def coarse_key(board):
    """A coarse summary of the board: the 'state' the table actually sees.

    What kind of cell the biggest tile sits in, how much room is left, and what
    each of the four moves would do -- but not *which* tiles are where.
    """
    b = np.asarray(board)
    biggest = int(b.argmax())
    free = min(int(np.count_nonzero(b == 0)), 6)        # 0..5, 6 = "plenty"
    return (_CELL_KIND[biggest], free) + tuple(_move_outcome(b, m, biggest) for m in Move)


def raw_key(board):
    return np.asarray(board).tobytes()


ABSTRACTIONS = {"coarse": coarse_key, "raw": raw_key}


class QLearningStrategy(Strategy, Trainable):
    name = "qlearning"

    def __init__(self, alpha=None, gamma=0.5, epsilon=0.05, abstraction="coarse", seed=None):
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.abstraction = abstraction
        self._key = ABSTRACTIONS[abstraction]
        self.q = defaultdict(float)            # (state_key, action) -> value
        self.visits = defaultdict(int)         # (state_key, action) -> updates so far
        self.rng = random.Random(seed)

    # -- playing -------------------------------------------------------------
    def _greedy(self, key, legal):
        return max(sorted(legal), key=lambda m: self.q[(key, int(m))])

    def select_move(self, board, legal):
        key = self._key(board)
        self.last_scores = {Move(m): self.q[(key, int(m))] for m in legal}
        return self._greedy(key, legal)

    # -- learning ------------------------------------------------------------
    def observe(self, transition, next_legal=None):
        """One Q-learning update. ``next_legal`` (the legal moves of
        ``transition.next_state``) restricts the max to moves that exist."""
        key = self._key(transition.state)
        target = transition.reward
        if not transition.done:
            next_key = self._key(transition.next_state)
            moves = next_legal if next_legal is not None else range(4)
            target += self.gamma * max(self.q[(next_key, int(m))] for m in moves)
        sa = (key, int(transition.move))
        self.visits[sa] += 1
        # alpha=None: step 1/N, i.e. each entry is the plain average of its targets
        step = self.alpha if self.alpha is not None else 1.0 / self.visits[sa]
        self.q[sa] += step * (target - self.q[sa])

    def train(self, env, episodes=10_000, verbose=False, report_every=500, **kwargs):
        """Epsilon-greedy self-play on ``env``, one Q update per move."""
        scores = []
        for ep in range(1, episodes + 1):
            board = env.reset()
            while True:
                legal = env.legal_moves()
                if not legal:
                    break
                if self.rng.random() < self.epsilon:
                    move = self.rng.choice(sorted(legal))
                else:
                    move = self._greedy(self._key(board), legal)
                next_board, reward, done, info = env.step(move)
                self.observe(Transition(board, move, reward, info["afterstate"], next_board, done),
                             next_legal=info["legal_moves"])
                board = next_board
                if done:
                    break
            scores.append(env.score)
            if verbose and ep % report_every == 0:
                window = scores[-report_every:]
                print(f"  episode {ep:>6}  avg score (last {report_every}): "
                      f"{sum(window) / len(window):.0f}  table entries: {len(self.q):,}",
                      flush=True)
        return scores

    # -- persistence ---------------------------------------------------------
    def save(self, path):
        config = dict(alpha=self.alpha, gamma=self.gamma, epsilon=self.epsilon,
                      abstraction=self.abstraction)
        with open(path, "wb") as f:
            pickle.dump({"config": config, "q": dict(self.q)}, f)

    @classmethod
    def load(cls, path, **overrides):
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = cls(**{**data["config"], **overrides})
        agent.q.update(data["q"])
        return agent
