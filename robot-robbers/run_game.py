from game.environment import RobotRobbersEnv
from models.dtos import RobotRobbersPredictRequestDto
import router


env = RobotRobbersEnv()
env.reset(4)
env.render()
sample = env.observation_space.sample()

# My variables
moveStart = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
firstRun = True
predictDTO = 0

current_key = None


while True:
    if firstRun:
        state, reward, is_done, info = env.step(moveStart)
        firstRun = False
    else:
        moveDTO = router.predict(predictDTO)
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
