from typing_extensions import Self
from game.environment import RobotRobbersEnv
import router

env = RobotRobbersEnv()
env.reset(42)
env.render()
sample = env.observation_space.sample()

print('Number of states: {}'.format(env.observation_space))
print('Number of actions: {}'.format(env.action_space))

while True:
    move = router.predict(env._get_observation())

    state, reward, is_done, info = env.step([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ])

    env.render()
