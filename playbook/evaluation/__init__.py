"""Evaluation: run games, aggregate metrics, compare strategies."""
from .compare import compare, compare_and_print
from .evaluate import evaluate
from .report import EvalReport, GameResult, compare_table
from .runner import play_game

__all__ = [
    "play_game", "evaluate", "compare", "compare_and_print",
    "GameResult", "EvalReport", "compare_table",
]
