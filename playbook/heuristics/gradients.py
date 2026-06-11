"""Positional weight matrices used by gradient-style heuristics.

These reward keeping large tiles anchored in a corner/edge, which is the single
most effective idea for 2048. Because the move encoding is fixed, these
orientations are fixed too.
"""
import numpy as np

# Decreasing weights from each of the four corners.
GRADIENT_TL = np.array([
    [3, 2, 1, 0],
    [2, 1, 0, -1],
    [1, 0, -1, -2],
    [0, -1, -2, -3],
])
GRADIENT_TR = np.array([
    [0, 1, 2, 3],
    [-1, 0, 1, 2],
    [-2, -1, 0, 1],
    [-3, -2, -1, 0],
])
GRADIENT_BL = np.array([
    [0, -1, -2, -3],
    [1, 0, -1, -2],
    [2, 1, 0, -1],
    [0, -1, -2, -3],
])
GRADIENT_BR = np.array([
    [-3, -2, -1, 0],
    [-2, -1, 0, 1],
    [-1, 0, 1, 2],
    [0, 1, 2, 3],
])

GRADIENT_CORNERS = [GRADIENT_TL, GRADIENT_TR, GRADIENT_BL, GRADIENT_BR]

# A snake/boustrophedon ordering — the ideal place for a monotonic chain.
GRADIENT_SNAKE = np.array([
    [16, 15, 14, 13],
    [9, 10, 11, 12],
    [8, 7, 6, 5],
    [1, 2, 3, 4],
])

# Tuned weights (from an earlier CMA-ES run) for the top-left anchor.
GRADIENT_OPT = np.array([
    [0.135759, 0.121925, 0.102812, 0.099937],
    [0.0997992, 0.0888405, 0.076711, 0.0724143],
    [0.060654, 0.0562579, 0.037116, 0.0161889],
    [0.0125498, 0.00992495, 0.00575871, 0.00335193],
])
