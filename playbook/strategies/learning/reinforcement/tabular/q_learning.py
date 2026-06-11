"""Tabular Q-learning — scaffold.

The simplest RL: a table ``Q[state, action]`` updated by
``Q(s,a) += alpha * (r + gamma * max_a' Q(s', a') - Q(s,a))``.

For 2048 the raw state space is astronomically large, so a plain table will not
generalize — that is exactly the lesson that motivates the n-tuple network
(:mod:`..ntuple <playbook.strategies.learning.reinforcement.ntuple>`) and deep
RL. This stub is kept as the didactic starting point: implement it over a coarse
state abstraction (e.g. bucketed features) to see where tabular methods break.
"""
import random
from collections import defaultdict

from playbook.strategies.base import Strategy, Trainable


class QLearningStrategy(Strategy, Trainable):
    name = "qlearning"

    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.1, seed=None):
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.q = defaultdict(float)            # (state_key, action) -> value
        self.rng = random.Random(seed)

    def _state_key(self, board):
        # TODO: replace with a coarse abstraction; the raw board does not generalize.
        return tuple(int(x) for x in board.flatten())

    def select_move(self, board, legal):
        key = self._state_key(board)
        return max(sorted(legal), key=lambda m: self.q[(key, int(m))])

    def observe(self, transition):
        raise NotImplementedError("TODO: apply the Q-learning update rule here")

    def train(self, env, episodes=10_000, verbose=False, **kwargs):
        raise NotImplementedError(
            "TODO: epsilon-greedy rollouts on SimEnv + the tabular Q-learning update."
        )
