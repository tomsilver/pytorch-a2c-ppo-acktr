from gym import spaces

import numpy as np


class NullAgent(object):

    def __init__(self, action_space):
        self.action_space = action_space
        self.discrete = isinstance(action_space, spaces.Discrete)

    def act(self, observation, reward, done):
        if self.discrete:
            return 0
        return np.zeros(self.action_space.shape[0])
