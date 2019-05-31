import numpy as np
from copy import copy, deepcopy
import random


def _to_val(c):
    if c == 0: return 0
    return 2**c


def to_val(m):
    return [[_to_val(c) for c in row] for row in m]


def _to_score(c):
    if c <= 1:
        return 0
    return (c-1) * (2**c)


def to_score(m):
    return np.array([[_to_score(c) for c in row] for row in m])


def score_with_move(board, move):
    pass


def score_to_val(score):
    return 2 ** score if score > 0 else 0


def make_move(board, move):
    grid = deepcopy(board)
    if move == 0:  # UP
        new_board = moveUp(grid)
        new_board, score = moveUpAdd(new_board)
    elif move == 1:  # DOWN
        new_board = moveDown(grid)
        new_board, score = moveDownAdd(new_board)
    elif move == 2:  # LEFT
        new_board = moveLeft(grid)
        new_board, score = moveLeftAdd(new_board)
    else: # RIGHT
        new_board = moveRight(grid)
        new_board, score = moveRightAdd(new_board)
    return new_board, score


def moveUp(grid):
    i = 0
    for j in range(0, 4):
        if grid[i][j] != 0 or grid[i + 1][j] != 0 or grid[i + 2][j] != 0 or grid[i + 3][j] != 0:  # are there non-zeros
            if grid[i][j] == 0:
                while grid[i][j] == 0:  # until first non-zero box
                    grid[i][j] = grid[i + 1][j]
                    grid[i + 1][j] = grid[i + 2][j]
                    grid[i + 2][j] = grid[i + 3][j]
                    grid[i + 3][j] = 0
            if grid[i + 1][j] == 0 and (
                    grid[i + 2][j] != 0 or grid[i + 3][j] != 0):  # is second box zero and boxes below it non zero
                while grid[i + 1][j] == 0:
                    grid[i + 1][j] = grid[i + 2][j]
                    grid[i + 2][j] = grid[i + 3][j]
                    grid[i + 3][j] = 0
            if grid[i + 2][j] == 0 and grid[i + 3][j] != 0:
                while grid[i + 2][j] == 0:
                    grid[i + 2][j] = grid[i + 3][j]
                    grid[i + 3][j] = 0
    return grid


def moveDown(grid):
    i = 0
    for j in range(0, 4):
        if grid[i][j] != 0 or grid[i + 1][j] != 0 or grid[i + 2][j] != 0 or grid[i + 3][j] != 0:  # are there non-zeros
            if grid[i + 3][j] == 0:
                while grid[i + 3][j] == 0:  # until first non-zero box
                    grid[i + 3][j] = grid[i + 2][j]
                    grid[i + 2][j] = grid[i + 1][j]
                    grid[i + 1][j] = grid[i][j]
                    grid[i][j] = 0
            if grid[i + 2][j] == 0 and (
                    grid[i + 1][j] != 0 or grid[i][j] != 0):  # is second box zero and boxes below it non zero
                while grid[i + 2][j] == 0:
                    grid[i + 2][j] = grid[i + 1][j]
                    grid[i + 1][j] = grid[i][j]
                    grid[i][j] = 0
            if grid[i + 1][j] == 0 and grid[i][j] != 0:
                while grid[i + 1][j] == 0:
                    grid[i + 1][j] = grid[i][j]
                    grid[i][j] = 0
    return grid


def moveLeft(grid):
    j = 0
    for i in range(0, 4):
        if grid[i][j] != 0 or grid[i][j + 1] != 0 or grid[i][j + 2] != 0 or grid[i][j + 3] != 0:  # are there non-zeros
            if grid[i][j] == 0:
                while grid[i][j] == 0:  # until first non-zero box
                    grid[i][j] = grid[i][j + 1]
                    grid[i][j + 1] = grid[i][j + 2]
                    grid[i][j + 2] = grid[i][j + 3]
                    grid[i][j + 3] = 0
            if grid[i][j + 1] == 0 and (
                    grid[i][j + 2] != 0 or grid[i][j + 3] != 0):  # is second box zero and boxes below it non zero
                while grid[i][j + 1] == 0:
                    grid[i][j + 1] = grid[i][j + 2]
                    grid[i][j + 2] = grid[i][j + 3]
                    grid[i][j + 3] = 0
            if grid[i][j + 2] == 0 and (grid[i][j + 3] != 0):
                while grid[i][j + 2] == 0:
                    grid[i][j + 2] = grid[i][j + 3]
                    grid[i][j + 3] = 0
    return grid


def moveRight(grid):
    j = 0
    for i in range(0, 4):
        if grid[i][j] != 0 or grid[i][j + 1] != 0 or grid[i][j + 2] != 0 or grid[i][j + 3] != 0:  # are there non-zeros
            if grid[i][j + 3] == 0:
                while grid[i][j + 3] == 0:  # until first non-zero box
                    grid[i][j + 3] = grid[i][j + 2]
                    grid[i][j + 2] = grid[i][j + 1]
                    grid[i][j + 1] = grid[i][j]
                    grid[i][j] = 0
            if grid[i][j + 2] == 0 and (
                    grid[i][j + 1] != 0 or grid[i][j] != 0):  # is second box zero and boxes below it non zero
                while grid[i][j + 2] == 0:
                    grid[i][j + 2] = grid[i][j + 1]
                    grid[i][j + 1] = grid[i][j]
                    grid[i][j] = 0
            if grid[i][j + 1] == 0 and (grid[i][j] != 0):
                while grid[i][j + 1] == 0:
                    grid[i][j + 1] = grid[i][j]
                    grid[i][j] = 0
    return grid


def moveUpAdd(grid):

    i = 0
    points = 0
    for j in range(0, 4):
        if grid[i][j] == grid[i + 1][j]:  # are first and second boxes the same?
            grid[i][j] += 1 if grid[i][j] > 0 else 0   # first and second boxes storing in the first
            points += grid[i][j]  # yay they get points
            grid[i + 1][j] = grid[i + 2][j]  # move third to 2nd pos
            grid[i + 2][j] = grid[i + 3][j]  # move 4th to 3rd pos
            grid[i + 3][j] = 0  # reset box
        if grid[i + 1][j] == grid[i + 2][j]:  # are second and third boxes the same
            grid[i + 1][j] += 1 if grid[i + 1][j] > 0 else 0   # add 2nd and 3rd boxes
            points += grid[i + 1][j]  # yay they get points
            grid[i + 2][j] = grid[i + 3][j]  # move 4th to 3rd pos
            grid[i + 3][j] = 0  # reset box
        if grid[i + 2][j] == grid[i + 3][j]:  # are third and fouth boxes the same
            grid[i + 2][j] += 1 if grid[i + 2][j] > 0 else 0   # add 3rd and 4th boxes
            points += grid[i + 2][j]  # yay they get points
            grid[i + 3][j] = 0  # reset box
    return grid, points


def moveDownAdd(grid):
    i = 0
    points = 0
    for j in range(0, 4):
        if grid[i + 3][j] == grid[i + 2][j]:  # are first and second boxes the same?
            grid[i + 3][j] += 1 if grid[i + 3][j] > 0 else 0   # first and second boxes storing in the first
            points += grid[i + 3][j]  # yay they get points
            grid[i + 2][j] = grid[i + 1][j]  # move third to 2nd pos
            grid[i + 1][j] = grid[i][j]  # move 4th to 3rd pos
            grid[i][j] = 0  # reset box
        if grid[i + 2][j] == grid[i + 1][j]:  # are second and third boxes the same
            grid[i + 2][j] += 1 if grid[i + 2][j] > 0 else 0   # add 2nd and 3rd boxes
            points += grid[i + 2][j]  # yay they get points
            grid[i + 1][j] = grid[i][j]  # move 4th to 3rd pos
            grid[i][j] = 0  # reset box
        if grid[i + 1][j] == grid[i][j]:  # are third and fouth boxes the same
            grid[i + 1][j] += 1 if grid[i + 1][j] > 0 else 0   # add 3rd and 4th boxes
            points += grid[i + 1][j]  # yay they get points
            grid[i][j] = 0  # reset box

    return grid, points


def moveLeftAdd(grid):

    j = 0
    points = 0
    for i in range(0, 4):
        if grid[i][j] == grid[i][j + 1]:  # are first and second boxes the same?
            grid[i][j] += 1 if grid[i][j] > 0 else 0   # first and second boxes storing in the first
            points += grid[i][j]  # yay they get points
            grid[i][j + 1] = grid[i][j + 2]  # move third to 2nd pos
            grid[i][j + 2] = grid[i][j + 3]  # move 4th to 3rd pos
            grid[i][j + 3] = 0  # reset box
        if grid[i][j + 1] == grid[i][j + 2]:  # are second and third boxes the same
            grid[i][j + 1] += 1 if grid[i][j + 1] > 0 else 0   # add 2nd and 3rd boxes
            points += grid[i][j + 1]  # yay they get points
            grid[i][j + 2] = grid[i][j + 3]  # move 4th to 3rd pos
            grid[i][j + 3] = 0  # reset box
        if grid[i][j + 2] == grid[i][j + 3]:  # are third and fouth boxes the same
            grid[i][j + 2] += 1 if grid[i][j + 2] > 0 else 0   # add 3rd and 4th boxes
            points += grid[i][j + 2]  # yay they get points
            grid[i][j + 3] = 0  # reset box
    return grid, points


def moveRightAdd(grid):
    j = 0
    points = 0
    for i in range(0, 4):
        if grid[i][j + 3] == grid[i][j + 2]:  # are first and second boxes the same?
            grid[i][j + 3] += 1 if grid[i][j + 3] > 0 else 0  # first and second boxes storing in the first
            points += grid[i][j + 3]  # yay they get points
            grid[i][j + 2] = grid[i][j + 1]  # move third to 2nd pos
            grid[i][j + 1] = grid[i][j]  # move 4th to 3rd pos
            grid[i][j] = 0  # reset box
        if grid[i][j + 2] == grid[i][j + 1]:  # are second and third boxes the same
            grid[i][j + 2] += 1 if grid[i][j + 2] > 0 else 0  # add 2nd and 3rd boxes
            points += grid[i][j + 2]  # yay they get points
            grid[i][j + 1] = grid[i][j]  # move 4th to 3rd pos
            grid[i][j] = 0  # reset box
        if grid[i][j + 1] == grid[i][j]:  # are third and fouth boxes the same
            grid[i][j + 1] += 1 if grid[i][j + 1] > 0 else 0   # add 3rd and 4th boxes
            points += grid[i][j + 1]  # yay they get points
            grid[i][j] = 0  # reset box
    return grid, points


def col_merges(board):
    merges = [0, 0, 0, 0]
    i = 0

    # Create copy a move up
    grid = deepcopy(board)
    grid = moveUp(grid)

    for j in range(0, 4):
        if grid[i][j] == grid[i + 1][j]:  # are first and second boxes the same?
            merges[j] += 1 if grid[i][j] > 0 else 0   # first and second boxes storing in the first
        if grid[i + 2][j] == grid[i + 3][j]:  # are second and third boxes the same
            merges[j] += 1 if grid[i + 2][j] > 0 else 0   # add 2nd and 3rd boxes
    return merges


def row_merges(board):
    # Create copy a move up
    grid = np.transpose(board)
    grid = moveUp(grid)
    merges = col_merges(grid)
    return merges

def get_free_tiles(board):
    free_tiles = []
    for i in range(0, 4):
        for j in range(0, 4):
            if board[i][j] == 0:
                free_tiles.append((i, j))
    return free_tiles


def count_zeros(board, axis=None):
    return 16 - int(np.count_nonzero(board, axis=axis))


def set_tile(board, i, j, value=None, dcopy=True):
    new_board = deepcopy(board) if dcopy else board
    if not value:
        new_board[i][j] = 1 if random.random() < 0.9 else 2
    else:
        new_board[i][j] = value
    return new_board


def add_random_tile(board):
    free_tiles = get_free_tiles(board)
    if len(free_tiles) > 0:
        i, j = free_tiles[random.randint(0, len(free_tiles)-1)]
        set_tile(board, i, j)


def print_board(m):
    for row in m:
        for c in row:
            print('%8d' % c, end=' ')
        print('')


def movename(move):
    return ['UP', 'DOWN', 'LEFT', 'RIGHT'][move]


def is_row_valid(board, b_move, i):
    for j in range(4):
        if board[i][j] != b_move[i][j]:
            return True
    return False


def get_valid_moves(board):
    board_up, _ = make_move(board, 0)
    board_down, _ = make_move(board, 1)
    board_left, _ = make_move(board, 2)
    board_right, _ = make_move(board, 3)

    all_b = [board_up, board_down, board_left, board_right]
    valid_moves = set()

    for i, b_move in enumerate(all_b):
        for r in range(4):
            if is_row_valid(board, b_move, r):
                valid_moves.add(i)
                break
    return valid_moves


def is_blocked(board):
    for i in range(4):
        for j in range(4):
            if j+1 <= 3:
                if board[i][j] == board[i][j+1]:
                    return False

            if i+1 <= 3:
                if board[i][j] == board[i+1][j]:
                    return False
    return True


if __name__ == "__main__":

    b = [[1,3,1,3],
        [2,1,2,1],
        [1,2,1,2],
        [5,1,3,3]]

    print(is_blocked(b))
