from game.environment import RobotRobbersEnv
from models.dtos import RobotRobbersPredictRequestDto
from models.dtos import RobotRobbersPredictResponseDto
import router
import requests
import json

env = RobotRobbersEnv()
env.reset(4)
env.render()
sample = env.observation_space.sample()

# My variables
moveStart = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
firstRun = True
predictDTO = 0

while True:
    if firstRun:
        state, reward, is_done, info = env.step(moveStart)
        predict_data = {
        'state': state.tolist(),
        'reward': reward,
        'is_terminal': is_done,
        'total_reward': info['total_reward'],
        'game_ticks': info['game_ticks']
        }
        predictDTO = RobotRobbersPredictRequestDto(**predict_data)
        firstRun = False
    else:
        moveDTO = requests.post('http://kurtskammerater.norwayeast.cloudapp.azure.com:4343/predict', json=predict_data)
        state, reward, is_done, info = env.step(moveDTO.json()['moves'])
    
    predict_data = {
        'state': state.tolist(),
        'reward': reward,
        'is_terminal': is_done,
        'total_reward': info['total_reward'],
        'game_ticks': info['game_ticks']
    }

    predictDTO = RobotRobbersPredictRequestDto(**predict_data)

    # If you want to render the game as it runs, it's recommended to
    # run this script locally:
    #
    # ```bash
    # python3 -m venv .venv
    # source .venv/bin/activate
    # pip install -r requirements.txt
    # python run_game.py
    # ```
    #
    env.render()
