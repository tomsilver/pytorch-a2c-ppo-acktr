from .pytorch_a2c_ppo_acktr import load_ppo

import os
import numpy as np
import torch


class PPOAgent(object):

    def __init__(self, env_id, action_space, load_dir=None, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        self.actor_critic, self.ob_rms = load_ppo(load_dir, env_id)

        self.clipob = clipob
        self.cliprew = cliprew
        self.gamma = gamma
        self.epsilon = epsilon

        self.recurrent_hidden_states = torch.zeros(1, self.actor_critic.recurrent_hidden_state_size)
        self.masks = torch.zeros(1, 1)

    def act(self, obs, reward, done):
        obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
        obs = torch.from_numpy(np.float32(obs))

        with torch.no_grad():
            _, action, _, self.recurrent_hidden_states = self.actor_critic.act(
                obs, self.recurrent_hidden_states, self.masks)

        action = np.squeeze(action)

        return action
