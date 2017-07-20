# -*- coding: utf-8 -*-
from keras.utils import plot_model
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from dqn import DQNAgent
from keras.models import model_from_json
from keras.models import load_model

EPISODES = 100

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.epsilon = 0.01

    # agent.model = model_from_json(open('cartpole.json').read())
    agent.load('cp2.h5')
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False

    for t in range( 10000 ):
        env.render()
        # action = agent.act(state)

        act_values = agent.model.predict(state)
        action = np.argmax(act_values[0])  # returns action

        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        state = next_state
        if done:
            print("score: {}, e: {:.2}"
                  .format(t, agent.epsilon))
            break