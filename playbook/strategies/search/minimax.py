"""Minimax with alpha-beta pruning.

Treats the random-tile placement as an adversary: the player maximizes, the
"machine" places the spawn in the worst cell for the player. Unlike the original
(which sampled a random tile value inside the min node), this drops a deterministic
2-tile, so the search is reproducible.
"""
from ...game.board import free_cells, set_tile, simulate_move
from ...game.rules import is_blocked, legal_moves
from .base import NEG_INF, POS_INF, SearchStrategy


class MinimaxStrategy(SearchStrategy):
    name = "minimax"

    def _maximize(self, board, depth, alpha, beta):
        if depth == 0 or is_blocked(board):
            return self.heuristic(board)
        best = NEG_INF
        for move in legal_moves(board):
            child, _, _ = simulate_move(board, move)
            best = max(best, self._minimize(child, depth - 1, alpha, beta))
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        return best if best != NEG_INF else self.heuristic(board)

    def _minimize(self, board, depth, alpha, beta):
        cells = free_cells(board)
        if depth == 0 or not cells:
            return self.heuristic(board)
        worst = POS_INF
        for i, j in cells:
            child = set_tile(board, i, j, 1)   # adversary drops a 2 in the worst spot
            worst = min(worst, self._maximize(child, depth - 1, alpha, beta))
            beta = min(beta, worst)
            if alpha >= beta:
                break
        return worst

    def select_move(self, board, legal):
        best_move, best = None, NEG_INF
        for move in sorted(legal):
            child, _, _ = simulate_move(board, move)
            score = self._minimize(child, self.depth, NEG_INF, POS_INF)
            if best_move is None or score > best:
                best_move, best = move, score
        return best_move
