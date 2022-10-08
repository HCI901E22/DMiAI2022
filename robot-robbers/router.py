from typing import Tuple
import numpy as np
from fastapi import APIRouter
from models.dtos import RobotRobbersPredictResponseDto, RobotRobbersPredictRequestDto
import Path
import math
import pickle

router = APIRouter()

file_name = "paths.pkl"
paths = []


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

    # open_file = open(file_name, "rb")
    global paths
    # print(paths)

    with open('logs/log.txt', 'a') as file:
        file.write(str(request.state) + '\n')

    # print(request.total_reward)
    moves = []
    # Make path towards cash and then deposit
    for x in range(5):
        if (request.state[5][x][0] > 0):
            if (veryClosebag(cashbags, roboPos(robots, x), scrooges)[0] != 0 and request.state[5][x][0] < 2):
                paths[x] = Path.MakeMatrix(
                    roboPos(robots, x),
                    closestBag(cashbags, roboPos(robots, x), request.state[1], request.state), request.state)
            elif (request.game_ticks % 2 == 0):
                paths[x] = Path.MakeMatrix(
                    roboPos(robots, x),
                    closestDeposit(dropspots, roboPos(robots, x), request.state[1]), request.state)
        else:
            if (request.game_ticks % 5 == 0):
                paths[x] = Path.MakeMatrix(
                    roboPos(robots, x),
                    closestBag(cashbags, roboPos(robots, x), request.state[1], request.state), request.state)
        moves += doMove((robots[x][0], robots[x][1]), paths[x])
    # open(file_name, 'w').close()
    # open_file = open(file_name, "wb")
    # pickle.dump(paths, open_file)
    # open_file.close()
    # moves += [0, 0, 0, 0]
    return RobotRobbersPredictResponseDto(
        moves=moves
    )


def roboPos(robots, robotNum):
    return (robots[robotNum][0], robots[robotNum][1])


def ItemPos(idIndex, itemNum, state):
    return (state[idIndex][itemNum][0], state[idIndex][itemNum][1])


def closestBag(bags, robotPos, scrooges, state):
    dist = 100000
    pos = robotPos
    for x in bags:
        if (x[0] == -1):
            continue
        if (math.dist(robotPos, (x[0], x[1])) < dist and checkScroogeNearby((x[0], x[1]),
                                                                            scrooges) and checkCarrierNearby(robotPos,
                                                                                                             state[0], (
                                                                                                             x[0],
                                                                                                             x[1]),
                                                                                                             state)):
            pos = (x[0], x[1])
            dist = math.dist(robotPos, (x[0], x[1]))
    return pos


def veryClosebag(bags, robotPos, scrooges):
    dist = 30
    pos = (0, 0)
    for x in bags:
        if (x[0] == -1):
            continue
        if (math.dist(robotPos, (x[0], x[1])) < dist and checkScroogeNearby((x[0], x[1]), scrooges)):
            pos = (x[0], x[1])
            dist = math.dist(robotPos, (x[0], x[1]))
    return pos


def checkCarrierNearby(robotPos, robots, bagPos, state):
    for i in range(len(robots)):
        if (math.dist(robotPos, bagPos) > math.dist((robots[i][0], robots[i][1]), bagPos) and state[5][i][0] > 0):
            return False
    return True


def closestDeposit(depos, robotPos, scrooges):
    dist = 100000
    pos = robotPos
    for x in depos:
        if (math.dist(robotPos, (x[0], x[1])) < dist and (
                checkScroogeNearby((x[0], x[1]), scrooges) or robotCloser(robotPos, (x[0], x[1]), scrooges))):
            pos = (x[0], x[1])
            dist = math.dist(robotPos, (x[0], x[1]))
    return pos


def checkScroogeNearby(item, scrooges):
    for x in range(len(scrooges)):
        if (math.dist(item, (scrooges[x][0], scrooges[x][1])) < 15):
            return False
    return True


def robotCloser(robotPos, itemPos, scrooges):
    for scrooge in scrooges:
        if (math.dist(robotPos, itemPos) > math.dist((scrooge[0], scrooge[1]), itemPos)):
            return False
    return True


def doMove(robotPos, path):
    move = []
    # Movement in X
    if (len(path) < 2):
        return [0, 0]
    if (robotPos[0] > path[1][0]):
        move += [-1]
    elif (robotPos[0] < path[1][0]):
        move += [1]
    elif (robotPos[0] == path[1][0]):
        move += [0]

    # Movement in Y
    if (robotPos[1] > path[1][1]):
        move += [-1]
    elif (robotPos[1] < path[1][1]):
        move += [1]
    elif (robotPos[1] == path[1][1]):
        move += [0]

    del path[0]
    return move


@router.get('/reset')
def reset():
    global paths
    paths = [[], [], [], [], []]
    # open(file_name, 'w').close()
    # open_file = open(file_name, "wb")
    # pickle.dump(paths, open_file)
    # open_file.close()
