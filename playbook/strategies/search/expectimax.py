"""Expectimax — the principled search for 2048.

The spawn is chance, not an adversary: chance nodes average over every empty
cell and over the real spawn distribution (a 2 with prob 0.9, a 4 with prob 0.1).
"""
from ...game.board import free_cells, set_tile, simulate_move
from ...game.rules import is_blocked, legal_moves
from .base import NEG_INF, SearchStrategy


class ExpectimaxStrategy(SearchStrategy):
    name = "expectimax"

    def _maximize(self, board, depth):
        if depth == 0 or is_blocked(board):
            return self.heuristic(board)
        best = NEG_INF
        for move in legal_moves(board):
            child, _, _ = simulate_move(board, move)
            best = max(best, self._chance(child, depth - 1))
        return best if best != NEG_INF else self.heuristic(board)

    def _chance(self, board, depth):
        cells = free_cells(board)
        if depth == 0 or not cells:
            return self.heuristic(board)
        total = 0.0
        for i, j in cells:
            total += 0.9 * self._maximize(set_tile(board, i, j, 1), depth - 1)  # spawn 2
            total += 0.1 * self._maximize(set_tile(board, i, j, 2), depth - 1)  # spawn 4
        return total / len(cells)

    def select_move(self, board, legal):
        best_move, best = None, NEG_INF
        for move in sorted(legal):
            child, _, _ = simulate_move(board, move)
            score = self._chance(child, self.depth)
            if best_move is None or score > best:
                best_move, best = move, score
        return best_move
