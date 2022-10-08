import random
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
trap = False
compute_dodge_path = True
trap_corner = (0, 0)
dodge_corner = (127, 127)


@router.post('/predict', response_model=RobotRobbersPredictResponseDto)
def predict(request: RobotRobbersPredictRequestDto):
    robots = [(x, y)
              for (x, y, _, _) in request.state[0] if x >= 0 and y >= 0]
    scrooges = [(x, y)
                for (x, y, _, _) in request.state[1] if x >= 0 and y >= 0]
    cashbags = [(x, y, w, h)
                for (x, y, w, h) in request.state[2] if x >= 0 and y >= 0]
    dropspots = [(x, y)
                 for (x, y, _, _) in request.state[3] if x >= 0 and y >= 0]
    obstacles = [(x, y, w, h) for (x, y, w, h) in request.state[4] if x >= 0 ]
    # open_file = open(file_name, "rb")
    global paths
    # print(paths)

    #with open('logs/log.txt', 'a') as file:
    #    file.write(str(request.state) + '\n')
    global compute_dodge_path
    if compute_dodge_path:
        decide_corner(dropspots)
        compute_dodge_path = False

    global dodge_corner

    # print(request.total_reward)
    moves = []
    if all_scrooges_close(scrooges, [robots[0], robots[1]]):
        global trap_corner
        moves += move_towards(robots[0], trap_corner, obstacles)
        moves += move_towards(robots[1], trap_corner, obstacles)
    else:
        moves += distract_scrooges(robots, scrooges, 0, obstacles)
        moves += distract_scrooges(robots, scrooges, 1, obstacles)

    if scrooges_trapped(scrooges):
    #Make path towards cash and then deposit
        for x in range(2,5):
            if (request.state[5][x][0] > 0):
                if (veryClosebag(cashbags, roboPos(robots, x), scrooges)[0] != 0 and request.state[5][x][0] < 4):
                    paths[x] = Path.MakeMatrix(
                        roboPos(robots, x),
                        closestBag(cashbags, roboPos(robots, x), request.state[1], request.state), request.state)
                elif (request.game_ticks % 10 == 0):
                    paths[x] = Path.MakeMatrix(
                        roboPos(robots, x),
                        closestDeposit(dropspots, roboPos(robots, x), request.state[1]), request.state)
            else:
                if (request.game_ticks % 15 == 0):
                    paths[x] = Path.MakeMatrix(
                        roboPos(robots, x),
                        closestBag(cashbags, roboPos(robots, x), request.state[1], request.state), request.state)
            moves += doMove((robots[x][0], robots[x][1]), paths[x])
    #open(file_name, 'w').close()
    #open_file = open(file_name, "wb")
    #pickle.dump(paths, open_file)
    #open_file.close()
    #moves += [0, 0, 0, 0]
    else:
        for i in range(2, 5):
            moves += move_towards(robots[i], dodge_corner, obstacles)


    #for i in range(5):
    #    moves += doMove(robots[i], paths[i])


















    # Make path towards cash and then deposit
    # for x in range(5):
    #     if (request.state[5][x][0] > 0):
    #         if (veryClosebag(cashbags, roboPos(robots, x), scrooges)[0] != 0 and request.state[5][x][0] < 2):
    #             paths[x] = Path.MakeMatrix(
    #                 roboPos(robots, x),
    #                 closestBag(cashbags, roboPos(robots, x), request.state[1], request.state), request.state)
    #         elif (request.game_ticks % 2 == 0):
    #             paths[x] = Path.MakeMatrix(
    #                 roboPos(robots, x),
    #                 closestDeposit(dropspots, roboPos(robots, x), request.state[1]), request.state)
    #     else:
    #         if (request.game_ticks % 5 == 0):
    #             paths[x] = Path.MakeMatrix(
    #                 roboPos(robots, x),
    #                 closestBag(cashbags, roboPos(robots, x), request.state[1], request.state), request.state)
    #     moves += doMove((robots[x][0], robots[x][1]), paths[x])
    # open(file_name, 'w').close()
    # open_file = open(file_name, "wb")
    # pickle.dump(paths, open_file)
    # open_file.close()
    # moves += [0, 0, 0, 0]
    return RobotRobbersPredictResponseDto(
        moves=moves
    )


def distract_scrooges(robots, scrooges, robot_idx, obstacles):
    ### Find scrooge furthest from trap
    furthest_scrooge = 512
    max_dist = -1
    robot = robots[robot_idx]
    for i, scrooge in enumerate(scrooges):
        old_dist = max_dist
        global trap_corner
        max_dist = max(max_dist, math.dist(trap_corner, scrooge))
        robo_dist = math.dist(robot, scrooge)
        if old_dist < max_dist and robo_dist >= 12:
            furthest_scrooge = i
        else:
            max_dist = old_dist
    scrooge = scrooges[furthest_scrooge]
    return move_towards(robot, scrooge, obstacles)


def all_scrooges_close(scrooges, robots):
    result = True
    for s in scrooges:
        trapped = False
        for robot in robots:
            trapped |= math.dist(robot, s) < 12
        result &= trapped
    return result


def decide_corner(dropspots):
    corners = [(0, 0), (0, 127), (127, 0), (127, 127)]
    opposite = corners.copy()
    opposite.reverse()

    best_dist = -1
    best_idx = -1

    for idx, c in enumerate(corners):
        best = 512
        for s in dropspots:
            d = math.dist(c, s)
            best = min(best, d)
        if best > best_dist:
            best_dist = best
            best_idx = idx

    global trap_corner
    global dodge_corner
    trap_corner = corners[best_idx]
    dodge_corner = opposite[best_idx]


def move_towards(robot, destination, obstacles):
    x = -1 if destination[0] < robot[0] else 1 if destination[0] > robot[0] else 0
    y = -1 if destination[1] < robot[1] else 1 if destination[1] > robot[1] else 0
    for (ox, oy, w, h) in obstacles:
        if robot[0] + x == ox and oy <= robot[1] + y <= oy + h:
            #x = random.choice([0,1]) if x == -1 else random.choice([0,-1])
            #y = random.choice([1,-1]) if y == 0 else y
            x = 0
            top = oy
            bottom = oy + h
            top_dist = abs(top - destination[1])
            bottom_dist = abs(bottom - destination[1])
            y = -1 if top_dist < bottom_dist else 1
        elif ox <= robot[0] + x <= ox + w and oy == robot[1] + y:
            y = random.choice([0, 1]) if y == -1 else random.choice([0, -1])
            x = random.choice([1, -1]) if x == 0 else x
            y = 0
            left = ox
            right = ox + w
            left_dist = abs(left - destination[0])
            right_dist = abs(right - destination[0])
            x = -1 if left_dist < right_dist else 1
    return [x, y]


def scrooges_trapped(scrooges):
    result = True
    for s in scrooges:
        global trap_corner
        result &= math.dist(s, trap_corner) < 50
    return result

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
    global trap
    trap = False
    global compute_dodge_path
    compute_dodge_path = True
    global trap_corner
    trap_corner = None
    global dodge_corner
    dodge_corner = None
    # open(file_name, 'w').close()
    # open_file = open(file_name, "wb")
    # pickle.dump(paths, open_file)
    # open_file.close()
