"""The batched simulator must play exactly the same game as the scalar one."""
import json
import os

import numpy as np

from playbook.game import Move, simulate_move
from playbook.game.vector import (
    SYMMETRY_MOVES,
    SYMMETRY_PERMS,
    VecSimEnv,
    expand,
    random_symmetry,
    spawn_children,
    spawn_tiles,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "moves.json")


def test_expand_matches_golden_fixture():
    with open(FIXTURE) as f:
        cases = json.load(f)
    boards = np.array([c["board"] for c in cases], dtype=np.int8)
    after, rewards, legal = expand(boards)
    for i, case in enumerate(cases):
        for move in range(4):
            expected = np.array(case["moves"][str(move)]["board"], dtype=np.int8)
            assert np.array_equal(after[i, move], expected)
            assert rewards[i, move] == simulate_move(boards[i], move)[2]
        assert sorted(np.flatnonzero(legal[i]).tolist()) == case["valid_moves"]


def test_spawn_adds_exactly_one_tile_where_there_is_room():
    rng = np.random.default_rng(0)
    boards = rng.integers(0, 4, size=(500, 4, 4)).astype(np.int8)
    boards[0] = 1                                  # a full board gets nothing
    before = boards.copy()
    spawn_tiles(boards, rng)
    added = (boards != before).sum(axis=(1, 2))
    has_room = (before == 0).any(axis=(1, 2))
    assert np.array_equal(added, has_room.astype(int))
    assert set(np.unique(boards[boards != before])) <= {1, 2}


def test_spawn_children_is_a_distribution():
    board = np.array([[1, 2, 0, 0], [3, 0, 0, 0], [0] * 4, [0] * 4], dtype=np.int8)
    children, owner, prob = spawn_children(board[None])
    assert len(children) == 2 * 13                 # 13 empty cells x {2, 4}
    assert np.isclose(prob.sum(), 1.0) and set(owner) == {0}


def test_symmetries_commute_with_moves():
    rng = np.random.default_rng(1)
    boards = rng.integers(0, 6, size=(200, 4, 4)).astype(np.int8)
    after, rewards, _ = expand(boards)
    for s, perm in enumerate(SYMMETRY_PERMS):
        t_after, t_rewards, _ = expand(boards.reshape(-1, 16)[:, perm].reshape(-1, 4, 4))
        for move in Move:
            mapped = SYMMETRY_MOVES[s, move]
            assert np.array_equal(after[:, move].reshape(-1, 16)[:, perm],
                                  t_after[:, mapped].reshape(-1, 16))
            assert np.array_equal(rewards[:, move], t_rewards[:, mapped])
    b, m = random_symmetry(boards, rng, moves=np.zeros(len(boards), dtype=int))
    assert b.shape == boards.shape and m.shape == (len(boards),)


def test_vec_env_plays_and_restarts_games():
    env = VecSimEnv(64, seed=0)
    assert ((env.boards != 0).sum(axis=(1, 2)) == 2).all()
    rng = np.random.default_rng(0)
    finished = 0
    for _ in range(3000):
        _, _, legal = env.afterstates()
        moves = np.where(legal, rng.random(legal.shape), -1).argmax(axis=1)
        boards, rewards, dones, info = env.step(moves)
        assert (rewards >= 0).all() and info["afterstate"].shape == (64, 4, 4)
        finished += int(dones.sum())
        if dones.any():
            assert len(info["final_score"]) == dones.sum()
            assert ((boards[dones] != 0).sum(axis=(1, 2)) == 2).all()   # restarted
    assert finished > 0


def test_n_step_window_sums_the_right_rewards():
    from playbook.strategies.learning.reinforcement.deep.dqn import NStepWindow
    n, envs = 4, 8
    env, window = VecSimEnv(envs, seed=3), NStepWindow(n, envs)
    rng = np.random.default_rng(3)
    history = [[] for _ in range(envs)]   # per game: (afterstate, reward)
    expected, emitted = [], []
    for _ in range(2000):
        _, _, legal = env.afterstates()
        moves = np.where(legal, rng.random(legal.shape), -1).argmax(axis=1)
        boards, rewards, dones, info = env.step(moves)
        score_now = env.scores.copy()
        score_now[dones] = info.get("final_score", [])
        for i in range(envs):
            history[i].append((info["afterstate"][i].tobytes(), int(rewards[i])))
            h = history[i]
            if dones[i]:      # every pending afterstate: rewards until the end
                for k in range(max(0, len(h) - n), len(h)):
                    expected.append((h[k][0], sum(r for _, r in h[k + 1:]), True))
                history[i] = []
            elif len(h) >= n:  # the n-moves-old afterstate: the n-1 rewards after it
                expected.append((h[-n][0], sum(r for _, r in h[-n + 1:]), False))
        after, points, _, done = window.push(info["afterstate"], score_now, boards, dones)
        emitted += [(a.tobytes(), int(p), bool(d)) for a, p, d in zip(after, points, done)]
    assert sorted(emitted) == sorted(expected)


def test_symmetry_maps_move_values_like_moves():
    boards = np.random.default_rng(2).integers(0, 6, size=(300, 4, 4)).astype(np.int8)
    moves = np.random.default_rng(3).integers(4, size=300)
    _, mapped_moves = random_symmetry(boards, np.random.default_rng(7), moves=moves)
    _, mapped_values = random_symmetry(boards, np.random.default_rng(7), values=np.eye(4)[moves])
    assert np.array_equal(mapped_values.argmax(axis=1), mapped_moves)
