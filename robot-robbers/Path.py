from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import numpy as np


def MakeMatrix(fromTile, toTile, state):
    matrix = np.ones((128, 128))
    obstacles = [(x, y, w, h)
                 for (x, y, w, h) in state[4] if x >= 0 and y >= 0]
    #for x in obstacles:
    #    for h in range(x[3]):
    #        for w in range(x[2]):
    #            matrix[x[0] + w][x[1] + h] = 0

    for cx in range(0, 127):
        for cy in range(0, 127):
            for x, y, w, h in obstacles:
                if cx >= x and cx <= x + w and cy >= y and cy <= y + h:
                    matrix[cy][cx] = 0

    for scrooge in state[1]:
        if(scrooge[0] < 118 and scrooge[1] < 118 and scrooge[1] > 10 and scrooge[0] > 10):
            for i in range(20):
                for j in range(20):
                    matrix[scrooge[1]  - 10 + i][scrooge[0]  - 10 + j] = 0

    grid = Grid(matrix=matrix)

    start = grid.node(fromTile[0], fromTile[1])
    end = grid.node(toTile[0], toTile[1])

    finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
    path, runs = finder.find_path(start, end, grid)
            

    #print(path)
    #print(grid.grid_str(path=path, start=start, end=end))
    return path
