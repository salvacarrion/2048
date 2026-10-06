"""Monte Carlo Tree Search (UCT), with chance nodes for the random tile.

:mod:`rollouts` gives every move the same number of random playouts and forgets
them. MCTS spends the same budget *adaptively*: it grows a search tree, one node
per simulation, and at each decision balances **exploiting** the moves that have
scored well so far with **exploring** the ones it knows little about -- the
UCB1 rule, ``mean + c * sqrt(ln N / n)``. Statistics accumulate in the tree, so
the decisions *below* the root get refined too.

2048 is not a two-player game: after each move the game drops a random tile. So
the tree alternates two kinds of node:

  * **decision** nodes (a board, our turn): pick a move by UCB1;
  * **chance** nodes (an afterstate): the game picks the tile. We *sample* it
    from the real distribution (an empty cell uniformly, a 2 with probability
    0.9) and keep one child per outcome seen, so likely spawns get visited more.

One simulation: walk down the tree, expand one new move, score the board it leads
to with one random playout (points over ``depth`` moves + the leaf heuristic,
exactly as in :mod:`rollouts`), and add the return to every node on the path.
Returns are in points, so UCB compares siblings on a 0-1 scale (their min-max).

``runs`` is the number of simulations per move (at least the number of moves,
so each gets tried). ``mcts --runs 80`` spends about
the same playouts as ``rollouts --runs 20`` (20 for each of ~4 moves); the
difference between the two is only *where* the playouts go.
"""
import math

from ...game.board import Move, add_random_tile, free_cells, set_tile, simulate_move
from ...game.rules import legal_moves
from .base import SearchStrategy


class _Decision:
    """A board where we choose the move."""

    __slots__ = ("board", "untried", "children", "visits")

    def __init__(self, board):
        self.board = board
        self.untried = sorted(legal_moves(board), reverse=True)  # expanded UP first
        self.children = {}                                       # move -> _Chance
        self.visits = 0


class _Chance:
    """An afterstate where the game places the random tile."""

    __slots__ = ("after", "reward", "children", "visits", "total")

    def __init__(self, after, reward):
        self.after, self.reward = after, reward
        self.children = {}                     # (cell, exponent) -> _Decision
        self.visits, self.total = 0, 0.0

    def mean(self):
        return self.total / self.visits


class MctsStrategy(SearchStrategy):
    name = "mcts"

    def __init__(self, heuristic=None, depth=10, runs=80, c=1.0, seed=None):
        super().__init__(heuristic, depth, seed)
        self.runs, self.c = runs, c

    def _spawn(self, chance):
        """Sample the random tile; return the decision node it leads to."""
        cells = free_cells(chance.after)
        i, j = cells[self.rng.randrange(len(cells))]
        exp = 1 if self.rng.random() < 0.9 else 2
        child = chance.children.get((i, j, exp))
        if child is None:
            child = chance.children[(i, j, exp)] = _Decision(set_tile(chance.after, i, j, exp))
        return child

    def _rollout(self, board):
        """One random playout: the points scored plus the leaf heuristic."""
        b, total = board, 0.0
        for _ in range(self.depth):
            moves = legal_moves(b)
            if not moves:
                break
            b, _, reward = simulate_move(b, self.rng.choice(sorted(moves)))
            total += reward
            add_random_tile(b, self.rng)
        return total + self.heuristic(b)

    def _select(self, node):
        """UCB1 over the moves already tried, means rescaled to 0-1 among siblings."""
        means = [ch.mean() for ch in node.children.values()]
        lo, span = min(means), (max(means) - min(means)) or 1.0
        log_n = math.log(node.visits)
        return max(node.children.values(),
                   key=lambda ch: (ch.mean() - lo) / span + self.c * math.sqrt(log_n / ch.visits))

    def _simulate(self, root):
        path, node = [], root
        while True:
            if node.untried:                   # expand: try a new move from here
                move = node.untried.pop()
                after, _, reward = simulate_move(node.board, move)
                chance = node.children[move] = _Chance(after, reward)
                path.append((node, chance))
                value = self._rollout(self._spawn(chance).board)
                break
            if not node.children:              # game over: nothing more to score
                value = 0.0
                break
            chance = self._select(node)
            path.append((node, chance))
            node = self._spawn(chance)
        for decision, chance in reversed(path):   # back up the return
            value += chance.reward
            chance.visits += 1
            chance.total += value
            decision.visits += 1

    def select_move(self, board, legal):
        root = _Decision(board)
        for _ in range(self.runs):
            self._simulate(root)
        children = root.children
        # explain as the share of simulations each move received: the most
        # simulated move is the most robust choice (ties: the higher mean)
        self.last_scores = {Move(m): 100 * ch.visits / self.runs for m, ch in children.items()}
        return max(sorted(children), key=lambda m: (children[m].visits, children[m].mean()))
