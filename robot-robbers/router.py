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
    cashbags = [(x, y)
                for (x, y, _, _) in request.state[2] if x >= 0 and y >= 0]
    dropspots = [(x, y)
                 for (x, y, _, _) in request.state[3] if x >= 0 and y >= 0]
    obstacles = [(x, y, w, h) for (x, y, w, h) in request.state[4] if x >= 0]
    padded_obstacles = [(x - 6, y - 6, w + 12, h + 12) for (x, y, w, h) in request.state[4] if x >= 0]
    # open_file = open(file_name, "rb")
    global paths
    # print(paths)

    with open('logs/log.txt', 'a') as file:
        file.write(str(request.state) + '\n')
    global compute_dodge_path
    if (compute_dodge_path and len(dropspots) == 3):
        decide_corner(dropspots, cashbags, robots[0])
        compute_dodge_path = False

    global dodge_corner

    # print(request.total_reward)
    global trap_corner
    moves = []
    trapped = scrooges_trapped(scrooges)
    n_distractors = 3
    if all_scrooges_close(scrooges, robots[0:n_distractors]):
        decide_corner(dropspots, cashbags, robots[0])
        global trap_corner
        for i in range(n_distractors):
            moves += move_towards(robots[i], trap_corner, padded_obstacles, cashbags, False)
    else:
        for i in range(n_distractors):
            moves += distract_scrooges(robots, scrooges, i, padded_obstacles, cashbags)

    if trapped:
        # Make path towards cash and then deposit
        for x in range(n_distractors, 5):
            if request.state[5][x][0] > 0:
                if veryClosebag(cashbags, roboPos(robots, x), scrooges, request.state, request.state[5][x][0]) and request.state[5][x][0] < 32:
                    moves += move_towards(robots[x], closestBag(cashbags, roboPos(robots, x), request.state[1],
                                                                request.state), obstacles, scrooges)

                else:
                    moves += move_towards(robots[x], closestDeposit(dropspots, roboPos(robots, x), request.state[1]),
                                          obstacles, scrooges)
            else:
                moves += move_towards(robots[x], closestBag(cashbags, roboPos(robots, x), request.state[1],
                                                            request.state), obstacles, scrooges)
            # moves += doMove((robots[x][0], robots[x][1]), paths[x])
    # open(file_name, 'w').close()
    # open_file = open(file_name, "wb")
    # pickle.dump(paths, open_file)
    # open_file.close()
    # moves += [0, 0, 0, 0]
    else:
        for i in range(n_distractors, 5):
            moves += move_towards(robots[i], dodge_corner, obstacles, scrooges, False)

    # for i in range(5):
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


def distract_scrooges(robots, scrooges, robot_idx, obstacles, cashbags):
    ### Find scrooge closest to robot
    closest_scrooge = -1
    min_dist = 512
    robot = robots[robot_idx]
    for i, scrooge in enumerate(scrooges):
        old_dist = min_dist
        global trap_corner
        min_dist = min(min_dist, math.dist(trap_corner, scrooge))
        robo_dist = math.dist(robot, scrooge)
        if old_dist > min_dist and robo_dist > 12:
            closest_scrooge = i
        else:
            min_dist = old_dist
    scrooge = scrooges[closest_scrooge]
    return move_towards(robot, scrooge, obstacles, cashbags, False)


def all_scrooges_close(scrooges, robots):
    result = True
    for s in scrooges:
        trapped = False
        for robot in robots:
            trapped |= math.dist(robot, s) < 14
        result &= trapped
    return result


def decide_corner(dropspots, cashbags, robot):
    corners = [(0, 0), (0, 127), (127, 0), (127, 127)]

    def sort_fn(cor):
        result = math.dist(cor, robot)
        return result

    corners.sort(key=sort_fn, reverse=False)

    global trap_corner
    global dodge_corner
    trap_corner = corners[0]
    dodge_corner = corners[3]


def move_towards(robot, destination, obstacles, money_bags, take_money=True):
    r = random.random()

    x = -1 if destination[0] < robot[0] else 1 if destination[0] > robot[0] else 0
    y = -1 if destination[1] < robot[1] else 1 if destination[1] > robot[1] else 0
    for (ox, oy, w, h) in obstacles:
        corners = [(ox - 1, oy - 1), (ox - 1, oy + h + 1), (ox + w + 1, oy - 1), (ox + w + 1, oy + h + 1)]
        if ox < robot[0] < (ox + w) and oy < robot[y] < (oy + h):
            x = random.choice([-1, 1])
            y = random.choice([1, -1])
        elif (robot[0] + x == ox or robot[0] + x == ox + w) and (oy == robot[1] + y or robot[1] + y == oy + h):
            if ox <= robot[0] <= ox + w:
                y = 0
            elif oy <= robot[1] <= oy + h:
                x = 0
            elif robot in corners:
                if ox <= destination[0] <= ox + w:
                    x = 0
                elif oy <= destination[1] <= oy + h:
                    y = 0
                else:
                    x = random.choice([-1, 1])
                    y = random.choice([1, -1])
            elif ox <= destination[0] <= ox + w:
                y = 0
                if x == 0:
                    left = ox
                    right = ox + w
                    x = 1 if abs(destination[1] - right) < abs(destination[1] - left) else -1
            elif oy <= destination[1] <= oy + h:
                x = 0
                if y == 0:
                    top = oy
                    bottom = oy + h
                    y = 1 if abs(destination[1] - bottom) < abs(destination[1] - top) else -1
            elif h < w:
                x = 0
            else:
                y = 0
        elif robot in corners:
            if ox <= destination[0] <= ox + w:
                x = 0
            elif oy <= destination[1] <= oy + h:
                y = 0
            else:
                x = random.choice([-1, 1])
                y = random.choice([1, -1])
        elif (robot[0] == ox or robot[0] == ox + w) and oy <= robot[1] <= oy + h:
            x = 0
            if oy <= destination[1] <= oy + h:
                top = oy
                bottom = oy + h
                y = 1 if abs(destination[1] - bottom) < abs(destination[1] - top) else -1
        elif ox <= robot[0] <= ox + w and (oy == robot[1] or robot[1] == oy + h):
            y = 0
            if ox <= destination[0] <= ox + w:
                left = ox
                right = ox + w
                x = 1 if abs(destination[1] - right) < abs(destination[1] - left) else -1
        elif (robot[0] + x == ox or robot[0] + x == ox + w) and oy <= robot[1] + y <= oy + h:
            x = 0
            if oy <= 0:
                y = 1
            elif oy + h >= 127:
                y = -1
            if oy <= destination[1] <= oy + h:
                top = oy
                bottom = oy + h
                y = 1 if abs(destination[1] - bottom) < abs(destination[1] - top) else -1
        elif ox <= robot[0] + x <= ox + w and (oy == robot[1] + y or robot[1] + y == oy + h):
            y = 0
            if ox <= 0:
                x = 1
            elif ox + w >= 127:
                x = -1
            if ox <= destination[0] <= ox + w:
                left = ox
                right = ox + w
                x = 1 if abs(destination[1] - right) < abs(destination[1] - left) else -1
    moves = [-1, 0, 1]
    if not take_money:
        for bag in money_bags:

            if (x, y) == bag:
                x = random.choice([c for c in moves if c != x])
                y = random.choice([c for c in moves if c != y])
    if x == 0 and y == 0:
        x = random.choice(moves)
        y = random.choice(moves)
    return [x, y]


def scrooges_trapped(scrooges):
    result = True
    for s in scrooges:
        global trap_corner
        result &= math.dist(s, trap_corner) < 64
    return result


def roboPos(robots, robotNum):
    return (robots[robotNum][0], robots[robotNum][1])


def ItemPos(idIndex, itemNum, state):
    return (state[idIndex][itemNum][0], state[idIndex][itemNum][1])


def closestBag(bags, robotPos, scrooges, state):
    dist = 100000
    pos = robotPos
    for x in bags:
        if x[0] == -1:
            continue
        if (math.dist(robotPos, (x[0], x[1])) < dist and checkScroogeNearby((x[0], x[1]), scrooges)
                and checkCarrierNearby(robotPos, state[0], (x[0], x[1]), state)):
            pos = (x[0], x[1])
            dist = math.dist(robotPos, (x[0], x[1]))
    return pos


def veryClosebag(bags, robotPos, scrooges, state, r_bags):
    dist = 64
    if r_bags >= 3:
        dist = 32
    for x in bags:
        if (x[0] == -1):
            continue
        if math.dist(robotPos, (x[0], x[1])) < dist and checkScroogeNearby((x[0], x[1]),scrooges) and \
                checkCarrierNearby(robotPos, state[0], (x[0], x[1]), state):
            return True
    return False


def checkCarrierNearby(robotPos, robots, bagPos, state):
    for i in range(len(robots)):
        mydist = math.dist(robotPos, bagPos)
        theirdist = math.dist((robots[i][0], robots[i][1]), bagPos)
        if (mydist > theirdist and theirdist < 64 and state[5][i][0] > 0 and
                state[5][i][0] < 32):
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
        if scrooges[x][0] < 0 or scrooges[x][1] < 0:
            continue
        if (math.dist(item, (scrooges[x][0], scrooges[x][1])) < 17):
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
    trap_corner = (0, 0)
    global dodge_corner
    dodge_corner = (127, 127)
    open("logs/log.txt", 'w').close()
    # open_file = open(file_name, "wb")
    # pickle.dump(paths, open_file)
    # open_file.close()
    return "State reset"
