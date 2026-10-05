"""Batched 2048: thousands of games at once, fast enough to feed a GPU.

:class:`~playbook.game.env.SimEnv` plays one game in plain Python (~3k moves/s):
perfect for reading and for evaluation, far too slow for a neural network that
learns from hundreds of millions of moves. This module plays the *same game* on
a whole batch of boards with numpy:

  * A row of four exponents (each 0..15) packs into one 16-bit number, so there
    are only 65,536 possible rows. Each of them is collapsed to the left
    **once**, with the canonical rule from :mod:`board`, and stored in a table.
    A move on a batch of boards is then just table lookups.
  * Every direction is that LEFT collapse on a re-oriented board, exactly as in
    :func:`board.simulate_move`.

Boards keep the package-wide encoding, batched: ``(N, 4, 4)`` int8 arrays of
exponents. Tiles up to 32768 (exponent 15) are supported.
"""
import numpy as np

from .board import Move, _collapse_left_line

MAX_PACKED_EXP = 15   # a row cell must fit in 4 bits

_LEFT = None     # (65536, 4) int8: the row after a LEFT collapse
_REWARD = None   # (65536,) int64: in-game points scored by that collapse


def _tables():
    global _LEFT, _REWARD
    if _LEFT is None:
        rows = np.arange(1 << 16)
        cells = (rows[:, None] >> np.array([0, 4, 8, 12])) & 0xF
        left = np.empty((1 << 16, 4), dtype=np.int8)
        reward = np.empty(1 << 16, dtype=np.int64)
        for r, line in enumerate(cells.tolist()):
            left[r], _, reward[r] = _collapse_left_line(line)
        _LEFT, _REWARD = left, reward
    return _LEFT, _REWARD


def _collapse_left(boards):
    """LEFT collapse of every row of a ``(N, 4, 4)`` batch -> (boards, rewards)."""
    left, reward = _tables()
    b = boards.astype(np.int32)
    rows = b[..., 0] | (b[..., 1] << 4) | (b[..., 2] << 8) | (b[..., 3] << 12)
    return left[rows], reward[rows].sum(axis=1)


# Same (view, unview) trick as board._ORIENT, with a leading batch axis.
_ORIENT = {
    Move.UP:    (lambda b: b.transpose(0, 2, 1),
                 lambda o: o.transpose(0, 2, 1)),
    Move.DOWN:  (lambda b: b.transpose(0, 2, 1)[:, :, ::-1],
                 lambda o: o[:, :, ::-1].transpose(0, 2, 1)),
    Move.LEFT:  (lambda b: b,
                 lambda o: o),
    Move.RIGHT: (lambda b: b[:, :, ::-1],
                 lambda o: o[:, :, ::-1]),
}


def simulate_moves(boards, move):
    """Batched :func:`board.simulate_move`: the same ``move`` on every board.

    Returns ``(afterstates, rewards)`` with shapes ``(N, 4, 4)`` and ``(N,)``.
    """
    view, unview = _ORIENT[Move(move)]
    out, reward = _collapse_left(view(boards))
    return np.ascontiguousarray(unview(out)), reward


def expand(boards):
    """Every move from every board at once.

    Returns ``(afterstates, rewards, legal)`` shaped ``(N, 4, 4, 4)``,
    ``(N, 4)`` and ``(N, 4)``; the second axis is the move (UDLR). A move is
    legal when it changes the board.
    """
    boards = np.asarray(boards, dtype=np.int8)
    after = np.empty((len(boards), 4, 4, 4), dtype=np.int8)
    rewards = np.empty((len(boards), 4), dtype=np.int64)
    for move in Move:
        after[:, move], rewards[:, move] = simulate_moves(boards, move)
    legal = (after != boards[:, None]).any(axis=(2, 3))
    return after, rewards, legal


def spawn_tiles(boards, rng):
    """In place: one tile on a uniformly random empty cell of every board that
    has one -- a 2 (exponent 1) with probability 0.9, else a 4."""
    assert boards.flags.c_contiguous, "spawn_tiles writes through a reshaped view"
    flat = boards.reshape(len(boards), 16)
    empty = flat == 0
    keys = np.where(empty, rng.random(flat.shape), -1.0)
    cell = keys.argmax(axis=1)
    values = np.where(rng.random(len(boards)) < 0.9, 1, 2).astype(np.int8)
    rows = np.flatnonzero(empty.any(axis=1))
    flat[rows, cell[rows]] = values[rows]
    return boards


def spawn_children(afterstates):
    """Every possible spawn after each afterstate: the chance node of expectimax.

    Returns ``(children, owner, prob)``: ``children[k]`` is ``afterstates[owner[k]]``
    with one new tile, and ``prob[k]`` the chance of that spawn (``prob`` sums to
    1 for each afterstate that has an empty cell).
    """
    flat = np.asarray(afterstates, dtype=np.int8).reshape(len(afterstates), 16)
    owner, cell = np.nonzero(flat == 0)
    k = len(owner)
    children = np.concatenate([flat[owner], flat[owner]])
    children[np.arange(k), cell] = 1           # a 2 ...
    children[k + np.arange(k), cell] = 2       # ... or a 4
    n_empty = np.bincount(owner, minlength=len(flat))[owner]
    prob = np.concatenate([0.9 / n_empty, 0.1 / n_empty])
    return children.reshape(-1, 4, 4), np.concatenate([owner, owner]), prob


# --------------------------------------------------------------------------- #
# Symmetry: the 8 rotations/reflections of the board play the same game
# --------------------------------------------------------------------------- #
def _symmetries():
    base = np.arange(16).reshape(4, 4)
    perms = []
    for grid in (base, base.T):
        for k in range(4):
            perms.append(np.rot90(grid, k).ravel())
    perms = np.array(perms)
    # Which move on the transformed board matches each original move? Found by
    # probing a board where all four moves give different results.
    probe = np.array([[1, 2, 3, 1], [2, 0, 0, 3], [3, 0, 0, 2], [1, 3, 2, 4]], np.int8)
    after, _, _ = expand(probe[None])
    move_maps = np.empty((8, 4), dtype=np.int64)
    for s, perm in enumerate(perms):
        t_after, _, _ = expand(probe.ravel()[perm].reshape(1, 4, 4))
        for m in range(4):
            mapped = after[0, m].ravel()[perm]
            move_maps[s, m] = next(m2 for m2 in range(4)
                                   if np.array_equal(t_after[0, m2].ravel(), mapped))
    return perms, move_maps


SYMMETRY_PERMS, SYMMETRY_MOVES = _symmetries()


def random_symmetry(boards, rng, moves=None, values=None):
    """Apply an independent random symmetry to each board (data augmentation).

    If ``moves`` (one per board) or ``values`` (one per board and move, shape
    ``(N, 4)``) are given, they are mapped consistently, so the pairs stay correct
    (e.g. a horizontal flip swaps LEFT and RIGHT). Returns the boards, followed by
    the mapped ``moves`` and/or ``values`` if given.
    """
    n = len(boards)
    sym = rng.integers(8, size=n)
    flat = np.asarray(boards).reshape(n, 16)
    out = [flat[np.arange(n)[:, None], SYMMETRY_PERMS[sym]].reshape(n, 4, 4)]
    if moves is not None:
        out.append(SYMMETRY_MOVES[sym, np.asarray(moves)])
    if values is not None:
        mapped = np.empty_like(values)
        mapped[np.arange(n)[:, None], SYMMETRY_MOVES[sym]] = values
        out.append(mapped)
    return out[0] if len(out) == 1 else tuple(out)


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
class VecSimEnv:
    """``n`` independent games stepped together; a finished game restarts at once.

    The batched sibling of :class:`~playbook.game.env.SimEnv` (same rules, same
    spawn distribution, its own random stream). It always knows every move's
    outcome for the current boards, because choosing a move needs it anyway:
    :meth:`afterstates` returns them without recomputation.
    """

    def __init__(self, n, seed=None):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.boards = np.zeros((n, 4, 4), dtype=np.int8)
        self.scores = np.zeros(n, dtype=np.int64)
        self.moves = np.zeros(n, dtype=np.int64)
        self.reset()

    def reset(self):
        self._restart(np.arange(self.n))
        return self.boards.copy()

    def _restart(self, idx):
        fresh = np.zeros((len(idx), 4, 4), dtype=np.int8)
        spawn_tiles(fresh, self.rng)
        spawn_tiles(fresh, self.rng)
        self.boards[idx] = fresh
        self.scores[idx] = 0
        self.moves[idx] = 0
        self._expanded = expand(self.boards)

    def afterstates(self):
        """``(afterstates, rewards, legal)`` of every move from the current boards
        (see :func:`expand`)."""
        return self._expanded

    def step(self, moves):
        """Play one (legal) move per game.

        Returns ``(boards, rewards, dones, info)``. ``info["afterstate"]`` holds the
        board after each slide+merge, before the spawn. Finished games are restarted
        in ``boards``; their final ``board``/``score``/``moves`` are in
        ``info["final_*"]`` (indexed by ``np.flatnonzero(dones)``).
        """
        after_all, reward_all, legal = self._expanded
        rows = np.arange(self.n)
        if not legal[rows, moves].all():
            raise ValueError("VecSimEnv.step got an illegal move")
        after = after_all[rows, moves]
        rewards = reward_all[rows, moves]
        if after.max() > MAX_PACKED_EXP:
            raise OverflowError("tile above 32768: beyond the packed-row tables")

        self.boards = after.copy()
        spawn_tiles(self.boards, self.rng)
        self.scores += rewards
        self.moves += 1
        self._expanded = expand(self.boards)

        dones = ~self._expanded[2].any(axis=1)
        info = {"afterstate": after}
        if dones.any():
            done = np.flatnonzero(dones)
            info["final_board"] = self.boards[done].copy()
            info["final_score"] = self.scores[done].copy()
            info["final_moves"] = self.moves[done].copy()
            self._restart(done)
        return self.boards.copy(), rewards, dones, info
