from game.environment import RobotRobbersEnv
from models.dtos import RobotRobbersPredictRequestDto, RobotRobbersPredictResponseDto
import router
import random
import requests
import json
import numpy as np
import time

env = RobotRobbersEnv()
env.reset(random.randint(0, 10000))
env.render()
sample = env.observation_space.sample()

router.reset()

# My variables
moveStart = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
firstRun = True
predictDTO = 0

file = open("logs/log.txt")

if firstRun:
    state, reward, is_done, info = env.step(moveStart)
    firstRun = False

for state in file.readlines():
    state = np.asarray(json.loads(state))
    env.setState(state)
    env.render()
    time.sleep(0.01)

file.close()
