from typing import Tuple
import numpy as np
from fastapi import APIRouter
from models.dtos import RobotRobbersPredictResponseDto, RobotRobbersPredictRequestDto
import Path

router = APIRouter()


@router.post('/predict', response_model=RobotRobbersPredictResponseDto)
def predict(request: RobotRobbersPredictRequestDto):
    robots = [(x, y, w, h)
              for (x, y, w, h) in request.state[0] if x >= 0 and y >= 0]
    scrooges = [(x, y, w, h)
                for (x, y, w, h) in request.state[1] if x >= 0 and y >= 0]
    cashbags = [(x, y, w, h)
                for (x, y, w, h) in request.state[2] if x >= 0 and y >= 0]
    dropspots = [(x, y, w, h)
                 for (x, y, w, h) in request.state[3] if x >= 0 and y >= 0]
    obstacles = request.state[4]
    print(request.total_reward)
    # Your moves go here!
    path = []
    if (path == []):
        if (request.state[5][0][0] > 0):
            path = Path.MakeMatrix(
                (robots[0][0], robots[0][1]),
                (dropspots[0][0], dropspots[0][1]), request.state)
        else:
            path = Path.MakeMatrix(
                (robots[0][0], robots[0][1]),
                (cashbags[0][0], cashbags[0][1]), request.state)

    n_robbers = 5
    moves = doMove((robots[0][0], robots[0][1]), path)
    # print(moves)

    return RobotRobbersPredictResponseDto(
        moves=moves
    )


def ItemPos(idIndex, itemNum, state):
    return (state[idIndex][itemNum][0], state[idIndex][itemNum][1])


def doMove(robotPos, path):
    move = []
    #print("RobotPos: {}", robotPos)
    #print("path: {}", path[0])
    if (robotPos[0] > path[1][0]):
        move += [-1]
    elif (robotPos[0] < path[1][0]):
        move += [1]
    elif (robotPos[0] == path[1][0]):
        move += [0]

    if (robotPos[1] > path[1][1]):
        move += [-1]
    elif (robotPos[1] < path[1][1]):
        move += [1]
    elif (robotPos[1] == path[1][1]):
        move += [0]
    move += [0, 0, 0, 0, 0, 0, 0, 0]
    return move
