"""Environments: the same strategy plays against either of these.

``Env`` is a small Gym-like interface. ``SimEnv`` is a fast pure-Python 2048
used for training and offline evaluation; ``BrowserEnv`` drives the real game in
Chrome. Because both expose the same ``reset``/``step``/``legal_moves`` API, a
strategy never knows (or cares) which one it is running against.
"""
import math
import random
from abc import ABC, abstractmethod

import numpy as np

from . import board as B
from .board import Move, empty_board, free_cells, simulate_move
from .rules import is_terminal, legal_moves


class Env(ABC):
    """A 2048 environment.

    ``step`` returns ``(board, reward, done, info)`` where ``reward`` is the true
    in-game score delta and ``info`` carries ``"afterstate"`` (the board after
    the slide+merge but *before* the random spawn — what value-based RL learns
    on) and ``"legal_moves"``.
    """

    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def step(self, move):
        ...

    @abstractmethod
    def legal_moves(self):
        ...

    @property
    @abstractmethod
    def score(self):
        ...


class SimEnv(Env):
    """In-memory 2048 simulator. Fast enough to train on (thousands of games/s)."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.board = empty_board()
        self._score = 0

    def reset(self):
        self.board = empty_board()
        self._score = 0
        B.add_random_tile(self.board, self.rng)
        B.add_random_tile(self.board, self.rng)
        return self.board.copy()

    def legal_moves(self):
        return legal_moves(self.board)

    def step(self, move):
        afterstate, _, reward = simulate_move(self.board, move)
        info = {"afterstate": afterstate, "valid": True}

        if np.array_equal(afterstate, self.board):
            # Illegal move: no change, no spawn. Caller should pick a legal move.
            info["valid"] = False
            info["legal_moves"] = self.legal_moves()
            return self.board.copy(), 0.0, is_terminal(self.board), info

        self.board = afterstate.copy()
        B.add_random_tile(self.board, self.rng)
        self._score += reward
        done = is_terminal(self.board)
        info["legal_moves"] = self.legal_moves()
        return self.board.copy(), float(reward), done, info

    @property
    def score(self):
        return self._score


class BrowserEnv(Env):
    """Drives the live game at play2048.co through Chrome's debugger.

    Thin adapter over :class:`Fast2048Control`; the heavy lifting (reading the
    grid, executing moves, UDLR->URDL remapping) lives there.
    """

    def __init__(self, port=9222, settle=0.0):
        # Imported lazily so the simulator works without the browser stack.
        from ..browser.chrome.chromectrl import ChromeDebuggerControl
        from ..browser.gamectrl import Fast2048Control

        self.ctrl = Fast2048Control(ChromeDebuggerControl(port))
        self.settle = settle

    def reset(self):
        self.ctrl.restart_game()
        return self._read_board()

    def _read_board(self):
        return np.asarray(self.ctrl.get_board(), dtype=np.int8)

    def legal_moves(self):
        return legal_moves(self._read_board())

    def step(self, move):
        before = self.score
        self.ctrl.execute_move(int(move))
        if self.ctrl.get_status() == "won":
            self.ctrl.continue_game()
        board = self._read_board()
        reward = float(self.score - before)
        done = self.ctrl.get_status() == "ended"
        info = {"afterstate": board, "legal_moves": legal_moves(board), "valid": True}
        return board, reward, done, info

    @property
    def score(self):
        return self.ctrl.get_score()
