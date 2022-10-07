from game.environment import RobotRobbersEnv
import numpy as np

env = RobotRobbersEnv()
state = env.reset()
env.render()
sample = env.observation_space.sample()

while True:
    
    if (env._cashbag_carriers.any() > 0):
        moves = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    else:
        moves = [np.random.randint(-1, 2) for _ in range(env.n_robbers * 2)]
    state, reward, is_done, info = env.step(moves)
    print(moves)
    
    env.render()
