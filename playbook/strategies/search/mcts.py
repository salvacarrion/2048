"""Monte-Carlo rollout search.

For each candidate move, play ``runs`` random games of length ``depth`` and
average the *return* of each rollout — the in-game points it scored plus the
heuristic of the board it ended on. The move with the best average wins. Simple,
embarrassingly parallel, and surprisingly strong.

Scoring a rollout by the points it collects (not just the heuristic of the final
board) is what makes the random tail informative: after a dozen-plus random
moves the board is always near-dead and messy, so its heuristic says little about
which *first* move was good — but the score racked up on the way there does.
"""
from ...game.board import add_random_tile, simulate_move
from ...game.rules import legal_moves
from .base import SearchStrategy


class MctsStrategy(SearchStrategy):
    name = "mcts"

    def __init__(self, heuristic=None, depth=10, runs=30, seed=None):
        super().__init__(heuristic, depth, seed)
        self.runs = runs

    def _rollout(self, board):
        """One random playout: return the points scored plus the leaf heuristic."""
        b = board
        total = 0.0
        for _ in range(self.depth):
            moves = legal_moves(b)
            if not moves:
                break
            b, _, reward = simulate_move(b, self.rng.choice(sorted(moves)))
            total += reward
            add_random_tile(b, self.rng)
        return total + self.heuristic(b)

    def select_move(self, board, legal):
        scores = {}
        for move in sorted(legal):
            child, _, _ = simulate_move(board, move)
            add_random_tile(child, self.rng)
            scores[move] = sum(self._rollout(child) for _ in range(self.runs)) / self.runs
        return self._pick(scores)
