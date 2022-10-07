from typing import Tuple
import numpy as np
from fastapi import APIRouter
from models.dtos import RobotRobbersPredictResponseDto, RobotRobbersPredictRequestDto
import Path
import math

router = APIRouter()

paths = [[],[],[],[],[]]

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

    print(paths)

    moves = []
    # Make path towards cash and then deposit
    for x in range(3):
        if(len(paths[x]) < 2 or checkScroogeNearby(roboPos(robots, x), scrooges)):
            if (request.state[5][x][0] > 0):
                paths[x] = Path.MakeMatrix(
                    roboPos(robots, x),
                    closestDeposit(dropspots, roboPos(robots, x), request.state[1]), request.state)
            else:
                paths[x] = Path.MakeMatrix(
                    roboPos(robots, x),
                    closestBag(cashbags, roboPos(robots, x), request.state[1]), request.state)
        moves += doMove((robots[x][0], robots[x][1]), paths[x])

    moves += [0, 0, 0, 0]

    return RobotRobbersPredictResponseDto(
        moves=moves
    )


def roboPos(robots, robotNum):
    return (robots[robotNum][0], robots[robotNum][1])


def ItemPos(idIndex, itemNum, state):
    return (state[idIndex][itemNum][0], state[idIndex][itemNum][1])


def closestBag(bags, robotPos, scrooges):
    dist = 100000
    pos = robotPos
    for x in bags:
        if (x[0] == -1):
            continue
        if (math.dist(robotPos, (x[0], x[1])) < dist and checkScroogeNearby((x[0], x[1]), scrooges)):
            pos = (x[0], x[1])
            dist = math.dist(robotPos, (x[0], x[1]))
    return pos


def closestDeposit(depos, robotPos, scrooges):
    dist = 100000
    pos = robotPos
    for x in depos:
        if (math.dist(robotPos, (x[0], x[1])) < dist and checkScroogeNearby((x[0], x[1]), scrooges)):
            pos = (x[0], x[1])
            dist = math.dist(robotPos, (x[0], x[1]))
    return pos


def checkScroogeNearby(item, scrooges):
    for x in range(len(scrooges)):
        if (math.dist(item, (scrooges[x][0], scrooges[x][1])) < 15):
            return False
    return True


def doMove(robotPos, path):
    move = []
    #Movement in X
    if (len(path) < 2):
        return [0, 0]
    if (robotPos[0] > path[1][0]):
        move += [-1]
    elif (robotPos[0] < path[1][0]):
        move += [1]
    elif (robotPos[0] == path[1][0]):
        move += [0]

    #Movement in Y
    if (robotPos[1] > path[1][1]):
        move += [-1]
    elif (robotPos[1] < path[1][1]):
        move += [1]
    elif (robotPos[1] == path[1][1]):
        move += [0]

    del path[0]
    return move
