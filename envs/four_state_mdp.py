import gym
from gym import spaces
import numpy as np

from typing import Optional


class Simple4MDP(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render_mode: Optional[str] = None):
        super(Simple4MDP, self).__init__()
        self.action_space = spaces.Discrete(1)  # Only one action available
        #self.action_space = {'n': 1}
        self.observation_space = spaces.Discrete(4)  # States 0, 1, 2, 3
        self.state = np.array([0])
        self.done = False

        # Define reward structure based on transitions
        # self.rewards = {0: 7, 1: 7, 2: 8, 3: 8}
        # self.alternate_rewards = {0: 3, 1: 3, 2: 2, 3: 2}
        
        # Original values Figure 3.
        self.rewards = {0: 2, 1: 2, 2: 1, 3: 1}
        self.alternate_rewards = {0: -2, 1: -2, 2: -1, 3: -1}
        
    def step(self, action):
        if self.done:
            return self.state, 0, self.done, {}, {} 

        # Decide the reward pattern randomly
        if np.random.rand() > 0.5:
            reward = self.rewards.get(self.state[0], 0)
        else:
            reward = self.alternate_rewards.get(self.state[0], 0)

        # Move to the next state
        self.state += 1

        # Check if terminal state
        if self.state[0] >= 4:
            self.done = True

        return self.state, reward, self.done, {}, {}

    def reset(self, seed=0):
        self.state = np.array([0])
        self.done = False
        return self.state, {}

    def render(self, mode='human'):
        print(f"State: {self.state}")

    def close(self):
        pass