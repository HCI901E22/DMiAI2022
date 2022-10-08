from game.environment import RobotRobbersEnv
from models.dtos import RobotRobbersPredictRequestDto, RobotRobbersPredictResponseDto
import router
import random
import requests
import json

env = RobotRobbersEnv()
env.reset(random.randint(0, 10000))
env.render()
sample = env.observation_space.sample()

router.reset()

# My variables
moveStart = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
firstRun = True
predictDTO = 0

while True:
    if firstRun:
        state, reward, is_done, info = env.step(moveStart)
        firstRun = False
    else:
        moveDTO = router.predict(predictDTO)
        #js = {'state': predictDTO.state, 'reward': predictDTO.reward, 'is_terminal': predictDTO.is_terminal, 'total_reward': predictDTO.total_reward, 'game_ticks': predictDTO.game_ticks}
        #responce = json.loads(requests.post("http://kurtskammerater.westeurope.cloudapp.azure.com:4343/predict", json=js).text)
        #moveDTO = RobotRobbersPredictResponseDto(moves=responce['moves'])
        state, reward, is_done, info = env.step(moveDTO.moves)
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
