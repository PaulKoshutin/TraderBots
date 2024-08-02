import gymnasium as gym
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


def preprocess(data, divide=False):
    res = []
    for i in range(len(data) - 1):
        if divide:
            diff = (data[i + 1] - data[i]) / 20000
        else:
            diff = data[i + 1] - data[i]
        if diff > 3:
            res.append(3)
        elif diff < -3:
            res.append(-3)
        elif diff == 0:
            res.append(1e-7)
        else:
            res.append(diff)
    return res

context_size = 240

close = pd.read_csv('dataTCSG.csv', sep=';').Close.values.tolist()
i_close = preprocess(close)

starting_money = 100000.0
min_increment = 0.5

"""
'Current price': [0 for i in range(context_size)],       0
'Inventory price': [0 for i in range(context_size)],     1
'Inventory occupied': [0 for i in range(context_size)],  2
'Current money': [0 for i in range(context_size)],       3
'Time step': [0 for i in range(context_size)]            4
"""

state = {
    'data': [0 for i in range(context_size + 5)],
}

def next_minute(state):
    state['data'][4] += 1
    state['data'][0] = close[int(state['data'][4])]
    state['data'][5:] = i_close[int(state['data'][4] - context_size):int(state['data'][4])]

def sell(state):
    if state['data'][2]:
        state['data'][3] += state['data'][0] - min_increment
        state['data'][1] = 0.0
        state['data'][2] = 0

def buy(state):
    if state['data'][3] >= state['data'][0] and not state['data'][2]:
        state['data'][3] -= state['data'][0]
        state['data'][1] = state['data'][0]
        state['data'][2] = 1

def do_nothing(state):
    pass


actions = [do_nothing, buy, sell]
action_space = gym.spaces.Discrete(len(actions))

observations = ['data']

std_min = [0,-np.inf, 0, -np.inf, -np.inf] + [-np.inf for i in range(context_size)]
std_max = [np.inf,np.inf, 1, np.inf, np.inf] + [np.inf for i in range(context_size)]

def make_obs_space():
    lower_obs_bound = {
        'data': std_min
    }
    higher_obs_bound = {
        'data': std_max
    }

    low = np.array([lower_obs_bound[o] for o in observations])
    high = np.array([higher_obs_bound[o] for o in observations])
    shape = (len(observations),context_size+5)
    return gym.spaces.Box(low,high,shape)


class TCSG(gym.Env):
    def __init__(self):
        self.actions = actions
        self.observations = observations
        self.action_space = action_space
        self.observation_space = make_obs_space()
        self.log = ''
        self.max_steps = len(i_close)

    def observation(self):
        return np.array([self.state[o] for o in self.observations])

    def reset(self, seed=1):
        self.rand = np.random.randint(0, len(i_close) // 1.11)
        self.max_steps = len(i_close) - (self.rand + context_size)
        std = [close[self.rand + context_size],0, 0, starting_money, self.rand + context_size] + i_close[self.rand:self.rand + context_size]
        self.state = {
            'data': std
        }
        self.steps_left = self.max_steps

        return self.observation(),{}

    def step(self, action):
        if self.state['data'][2]:
            potential_profit = self.state['data'][0] - min_increment
        else:
            potential_profit = 0

        if self.state['data'][4] == context_size:
            old_score = 0
        else:
            old_score = (self.state['data'][3] + potential_profit) / starting_money  - 1

        self.actions[action](self.state)
        self.log += f'Chosen action: {self.actions[action].__name__}\n'

        next_minute(self.state)

        if self.state['data'][2]:
            potential_profit = self.state['data'][0] - min_increment
        else:
            potential_profit = 0

        new_score = (self.state['data'][3] + potential_profit) / starting_money - 1 - 1 * ((self.state['data'][4]-self.rand-context_size)/(len(i_close)-self.rand-context_size))

        reward = new_score - old_score

        if self.state['data'][3] + self.state['data'][1] < self.state['data'][0]:
            self.log += f'Not enough money\n'

            reward -= 100

            self.state['data'][3] += 10000

        self.log += str(self.state) + '\n'

        self.steps_left -= 1
        done = (self.steps_left <= 0)

        return self.observation(), reward, done, done,{}

    def close(self):
        pass

    def render(self, mode=None):
        print(self.log)
        self.log = ''

tcsg = TCSG()
model = DQN.load("GymModel3", env=tcsg)

done, truncated = False, False
observation, info = tcsg.reset()
while not done and not truncated:
    position_index, _states = model.predict(observation)
    observation, reward, done, truncated, info = tcsg.step(position_index)

print(reward)