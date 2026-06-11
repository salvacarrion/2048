"""Maximization search: plan ``depth`` plies ahead assuming a single sampled
random spawn between moves, and take the move leading to the best board."""
from ...game.board import add_random_tile, simulate_move
from ...game.rules import is_blocked, legal_moves
from .base import NEG_INF, SearchStrategy


class MaximizationStrategy(SearchStrategy):
    name = "maximization"

    def _maximize(self, board, depth):
        if depth == 0 or is_blocked(board):
            return self.heuristic(board)
        best = NEG_INF
        for move in legal_moves(board):
            child, _, _ = simulate_move(board, move)
            add_random_tile(child, self.rng)
            best = max(best, self._maximize(child, depth - 1))
        return best if best != NEG_INF else self.heuristic(board)

    def select_move(self, board, legal):
        best_move, best = None, NEG_INF
        for move in sorted(legal):
            child, _, _ = simulate_move(board, move)
            add_random_tile(child, self.rng)
            score = self._maximize(child, self.depth)
            if best_move is None or score > best:
                best_move, best = move, score
        return best_move
