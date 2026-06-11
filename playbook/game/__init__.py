"""The 2048 "world": board mechanics, rules and environments."""
from .board import (
    Move,
    MOVE_NAMES,
    SIZE,
    add_random_tile,
    count_empty,
    empty_board,
    free_cells,
    make_move,
    move_name,
    move_reward,
    render,
    set_tile,
    simulate_move,
    to_value,
    to_values,
)
from .env import BrowserEnv, Env, SimEnv
from .rules import is_blocked, is_terminal, legal_moves

__all__ = [
    "Move", "MOVE_NAMES", "SIZE",
    "empty_board", "make_move", "simulate_move", "move_reward",
    "free_cells", "count_empty", "set_tile", "add_random_tile",
    "to_value", "to_values", "render", "move_name",
    "legal_moves", "is_terminal", "is_blocked",
    "Env", "SimEnv", "BrowserEnv",
]
