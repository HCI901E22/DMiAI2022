from argparse import Action
from random import random
from turtle import left, right, up
from game.environment import RobotRobbersEnv
from models.dtos import RobotRobbersPredictRequestDto
import numpy as np
import nn
import ActionType
import random

train_frames = 500
observe = 100
game_ticks = 0
GAMMA = 0.9

env = RobotRobbersEnv()
# request = RobotRobbersPredictRequestDto()
state = env.reset(4)
env.render()
sample = env.observation_space.sample()
model = nn.Model()
actionType = ActionType.ActionType()


nn_param = [128, 128]
params = {
    "batchSize": 64,
    "buffer": 100,
    "nn": nn_param
}
replay = []
batchSize = params['batchSize']
buffer = params['buffer']

model.neural_net(params['nn'])
def process_minibatch2(minibatch, model):

    # by Microos, improve this batch processing function
    #   and gain 50~60x faster speed (tested on GTX 1080)
    #   significantly increase the training FPS

    # instead of feeding data to the model one by one,
    #   feed the whole batch is much more efficient

    mb_len = len(minibatch)

    old_states = np.zeros(shape=(mb_len, env.observation_space))
    actions = np.zeros(shape=(mb_len, 10))
    rewards = np.zeros(shape=(mb_len,))
    new_states = np.zeros(shape=(mb_len, env.observation_space))

    for i, m in enumerate(minibatch):
        # print("index: ", i)
        old_state_m, action_m, reward_m, new_state_m = m
        old_states[i, :] = old_state_m[i, 1, 1]
        actions[i] = action_m
        rewards[i] = reward_m
        new_states[i, :] = new_state_m[i, 1, 1]

    # print("Actions list: ", actions)

    old_qvals = model.predict(old_states, batch_size=mb_len)
    new_qvals = model.predict(new_states, batch_size=mb_len)

    maxQs = np.max(new_qvals, axis=1)
    y = old_qvals
    non_term_inds = np.where(rewards != -5000)[0]
#  term_inds = np.where(rewards == -5000)[0]
#
    #y[non_term_inds, actions[non_term_inds].astype(
        #int)] = rewards[non_term_inds] + (GAMMA * maxQs[non_term_inds])
#  y[term_inds, actions[term_inds].astype(int)] = rewards[term_inds]
    #y = GAMMA * maxQs
    X_train = old_states
    y_train = y
    return X_train, y_train

def FormatQval(qval):
    moves = []
    for actor in range(len(qval[0]) - 5):
        moves += [qval[0][actor][0] - model.state[0][actor][0],
                  qval[0][actor][1] - model.state[0][actor][1]]

    return moves
        

while True:
    print(model.epsilon)
    # Choose an action.
    # moveActions = [actionType.up, actionType.down, actionType.left, actionType.right,
                        # actionType.upLeft, actionType.upRight, actionType.downLeft, actionType.downRight]

    if np.random.random() < model.epsilon:
        #print("Random action")
        #print("startStateCheck: ", model.startStateCheck)
        randomRow = np.random.randint(1, size=1)
        #moves = np.asarray([random.choice(moveActions) for _ in range(env.n_robbers)])
        moves = [np.random.choice(actionType.action) for _ in range(env.n_robbers * 2)]
        #moves = np.asarray([(x[0], x[1]) for x in moves])
    else:
        #print("Decision")
        qval = model.predict(model.state, batch_size=1)
        #print("qval: ", qval)
        #print("qval[0]: ", qval[0])

        moves = FormatQval(qval)
        print("action: ", moves)
        # moves = [np.random(moveActions[randomRow[0], :])) for _ in range(env.n_robbers)]
    new_state, reward, is_done, info = env.step(moves)
    #print(moves)

    # Take action, observe new state and get reward.
    #print("state: ", model.state, "new state: ", new_state, "reward: ", reward)
    replay.append((model.state, moves, reward, new_state))
    print("Size of replay: ", len(replay))
    # If we're done observing, start training.
    # if we've stored enough in our buffer, pop the oldest.
    if (game_ticks > observe):
        if len(replay) > buffer - 1:
            replay.pop(0)
            print("Training..")
            # randomly sample our experience replay memory
            minibatch = random.sample(replay, batchSize)

            # get training values.
            X_train, y_train = process_minibatch2(minibatch, model)

            # train the model on this batch
            history = nn.LossHistory()
            model.fit(X_train, y_train, batch_size=batchSize,
                      epochs=1, verbose=0, callbacks=[history])
            # loss_log.append(history.losses)
        else:
            print("Not training...")

    # Update the starting state S'.
    model.state = new_state
    game_ticks += 1

    
    if model.epsilon > 0.1:
        model.epsilon -= (1.0 / train_frames)

    env.render()
    

