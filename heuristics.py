import numpy as np

from game import *

GRADIENT_TL = np.array(
    [[3, 2, 1, 0],
     [2, 1, 0, -1],
     [1, 0, -1, -2],
     [0, -1, -2, -3]])

GRADIENT_TR = np.array(
    [[0, 1, 2, 3],
     [-1, 0, 1, 2],
     [-2, -1, 0, 1],
     [-3, -2, -1, -0]])

GRADIENT_BL = np.array(
    [[0, -1, -2, -3],
     [1, 0, -1, -2],
     [2, 1, 0, -1],
     [0, -1, -2, -3]])

GRADIENT_BR = np.array(
    [[-3, -2, -1, 0],
     [-2, -1, 0, 1],
     [-1, 0, 1, 2],
     [0, 1, 2, 3]])

GRADIENT_LIST = [GRADIENT_TL, GRADIENT_TR, GRADIENT_BL, GRADIENT_BR]

GRAD_OPT = np.array([
    [0.135759, 0.121925, 0.102812, 0.099937],
    [0.0997992, 0.0888405, 0.076711, 0.0724143],
    [0.060654, 0.0562579, 0.037116, 0.0161889],
    [0.0125498, 0.00992495, 0.00575871, 0.00335193]
])

GRADIENT_SORT = np.array(
    [[16, 15, 14, 13],
     [9, 10, 11, 12],
     [8, 7, 6, 5],
     [1, 2, 3, 4]])


H_SCORES = []


def h_combined(*args, **kwargs):
    score = 0
    for h_func, s in H_SCORES:
        score += s * h_func(*args, **kwargs)
    return score


def h_aprox_score(board):
    b = board[np.nonzero(board > 1)]  # Exclude 2s
    subscores = [pow(2, int(p))*(int(p)-1) for p in b]
    score = sum(subscores)
    return score


def h_free_tiles(board):
    return count_zeros(board)


def h_max_tile(board):
    return int(np.max(board))


def h_potential_merges(board):
    r_merges = row_merges(board)
    c_merges = col_merges(board)
    return sum(r_merges) + sum(c_merges)


def h_total_sum(board):
    return int(np.sum(board))


def h_grad_corners(board):
    """

    """
    score = 0
    for grad in GRADIENT_LIST:
        score = max(score, np.sum(np.multiply(board, grad)))
    return score


def h_grad(board, grad=GRADIENT_TL):
    return np.sum(np.multiply(board, grad))


def h_grad_sort(board):
    return np.sum(np.multiply(board, GRADIENT_SORT))


def h_max_free(board):
    return h_free_tiles(board) * h_max_tile(board)


def h_grad_max_free(board):
    return np.sum(np.multiply(board, GRADIENT_TL))


class MyHeuristic:
    def __init__(self, *args):
        self.weights=args[0][0]

    def score_func(self, board):
        args = self.weights
        return float(args[0]) * h_grad(board, GRAD_OPT) + float(args[1]) * smoothness(board) + float(args[2]) * h_free_tiles(board) * h_max_tile(board)


def smoothness(board):
    values = set()
    for i in range(4):
          for j in range(4):
              values.add(board[i][j])
    return 1/len(values)

# def smoothness(board):
#     smoothness = 0
#     for i in range(4):
#         for j in range(4):
#             if board[i][j] != 0:
#                 value = board[i][j]
#     return smoothness

# expectimax, d=3, 3times, avg. score
def normalize_h_scores():
    h_scores_temp = [
        # (h_free_tiles, 5773),
        # (h_max_tile, 1202),
        (h_total_sum, 260),
        # (h_grad_corners, 5881),
        # (h_potential_merges, 1150),
        (h_grad, 7989),
        (h_max_free, 6117)
    ]
    # max_free = 5918
    # h_grad_tl + h_max_free = 9454

    # Get total scores
    total_scores = 0
    for f, s in h_scores_temp:
        total_scores += s
    for f, s in h_scores_temp:
        H_SCORES.append((f, s/total_scores))


# Run function to normalize scores
normalize_h_scores()


if __name__ == "__main__":
    b = [[16,15,2,0],
        [2,3,1,0],
        [4,0,0,0],
        [2,0,0,0]]

    board = np.array(b)
    print(h_aprox_score(board))
    asds = 3
    # rows = [2, 0, 1, 1]
    # cols = [1, 0, 0, 2]


