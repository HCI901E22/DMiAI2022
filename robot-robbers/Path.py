from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import numpy as np


def MakeMatrix(fromTile, toTile, state, dodge=True):
    if fromTile == toTile:
        return []

    matrix = np.ones((128, 128))
    obstacles = [(x, y, w, h)
                 for (x, y, w, h) in state[4] if x >= 0 and y >= 0]

    if dodge:
        for scrooge in state[1]:
            for i in range(30):
                for j in range(30):
                    if 0 < scrooge[1] - 15 + i < 128 and 0 < scrooge[0] - 15 + j < 128:
                        matrix[scrooge[1] - 15 + i][scrooge[0] - 15 + j] = 5

    for cx in range(0, 127):
        for cy in range(0, 127):
            for x, y, w, h in obstacles:
                if x <= cx <= x + w and y <= cy <= y + h:
                    matrix[cy][cx] = 0

    grid = Grid(matrix=matrix)

    start = grid.node(fromTile[0], fromTile[1])
    end = grid.node(toTile[0], toTile[1])

    finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
    path, runs = finder.find_path(start, end, grid)

    # print(path)
    # print(grid.grid_str(path=path, start=start, end=end))
    return path
