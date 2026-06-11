"""Monte-Carlo rollout search.

For each candidate move, play ``runs`` random games of length ``depth`` and
average the heuristic of the boards reached. The move with the best average
wins. Simple, embarrassingly parallel, and surprisingly strong.
"""
from ...game.board import add_random_tile, simulate_move
from ...game.rules import legal_moves
from .base import NEG_INF, SearchStrategy


class MctsStrategy(SearchStrategy):
    name = "mcts"

    def __init__(self, heuristic=None, depth=10, runs=30, seed=None):
        super().__init__(heuristic, depth, seed)
        self.runs = runs

    def _rollout(self, board):
        b = board
        for _ in range(self.depth):
            moves = legal_moves(b)
            if not moves:
                break
            b, _, _ = simulate_move(b, self.rng.choice(sorted(moves)))
            add_random_tile(b, self.rng)
        return self.heuristic(b)

    def select_move(self, board, legal):
        best_move, best = None, NEG_INF
        for move in sorted(legal):
            child, _, _ = simulate_move(board, move)
            add_random_tile(child, self.rng)
            avg = sum(self._rollout(child) for _ in range(self.runs)) / self.runs
            if best_move is None or avg > best:
                best_move, best = move, avg
        return best_move
