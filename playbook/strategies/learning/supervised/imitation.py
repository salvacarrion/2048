"""Imitation learning — scaffold.

Supervised learning instead of reward: collect (board -> move) examples from a
strong teacher (e.g. run :class:`ExpectimaxStrategy` and log its choices), then
train a classifier to predict the teacher's move. A cheap way to "distill" a slow
search player into a fast reactive one.

Provided: the interface and a teacher-driven data-collection helper. Left as the
exercise: fit a model in :meth:`train` (sklearn or a small torch net) and use it
in :meth:`select_move`.
"""
from playbook.strategies.base import Strategy, Trainable


def collect_demonstrations(teacher, env_factory, games=50, max_moves=2000):
    """Roll out ``teacher`` and return a list of (board, move) examples."""
    examples = []
    for _ in range(games):
        env = env_factory()
        board = env.reset()
        for _ in range(max_moves):
            legal = env.legal_moves()
            if not legal:
                break
            move = teacher.select_move(board, legal)
            examples.append((board.copy(), int(move)))
            board, _, done, _ = env.step(move)
            if done:
                break
    return examples


class ImitationStrategy(Strategy, Trainable):
    name = "imitation"

    def __init__(self):
        self.model = None

    def select_move(self, board, legal):
        if self.model is None:
            raise RuntimeError("ImitationStrategy is untrained; call train() first")
        raise NotImplementedError("TODO: predict a move from self.model and clamp to `legal`")

    def observe(self, transition):
        raise NotImplementedError("Imitation learns from a dataset, not transitions")

    def train(self, env, episodes=None, teacher=None, **kwargs):
        raise NotImplementedError(
            "TODO: collect_demonstrations(teacher, SimEnv) then fit self.model "
            "to predict the teacher's move from the board."
        )
