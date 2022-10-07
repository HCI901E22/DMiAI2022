"""
The design of this comes from here:
http://outlace.com/Reinforcement-Learning-Part-3/
"""


import numpy as np
import tensorflow as tf
#from tensorflow.keras.layers import Dense, Activation, Dropout
#from tensorflow.keras.optimizers import RMSprop
#from tensorflow.keras.callbacks import Callback
from fastapi import FastAPI
from game.environment import RobotRobbersEnv


class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

env = RobotRobbersEnv()
class Model(tf.keras.Sequential):

    app = FastAPI()

    def __init__(self):
        super().__init__()
        self.epsilon = 1
        self.startStateCheck = True
        self.state = env.observation_space
        self.movesActions = [-1, 0, 1]

    def neural_net(self, params):

        # First layer.
        self.add(tf.keras.layers.Dense(
            params[0], kernel_initializer='lecun_uniform', input_shape=(6, 10, 4)
        ))
        self.add(tf.keras.layers.Activation('relu'))
        self.add(tf.keras.layers.Dropout(0.2))

        # Second layer.
        self.add(tf.keras.layers.Dense(params[1], kernel_initializer='lecun_uniform'))
        self.add(tf.keras.layers.Activation('relu'))
        self.add(tf.keras.layers.Dropout(0.2))

        # Output layer.
        self.add(tf.keras.layers.Dense(5, kernel_initializer='lecun_uniform'))
        self.add(tf.keras.layers.Activation('linear'))

        rms = tf.keras.optimizers.RMSprop()
        self.compile(loss='mse', optimizer=rms)

        #if load:
        #   self.load_weights(load)
