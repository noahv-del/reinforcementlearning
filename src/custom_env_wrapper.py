import gymnasium as gym
import numpy as np
from gymnasium import RewardWrapper
import math

class CustomRewardWrapper(RewardWrapper):

    def __init__(self, env: gym.Env, cfg: dict):
        super().__init__(env)
        self.cfg = cfg

    def reward(self, reward: float) -> float:
        theta, theta_dot = self.env.unwrapped.state
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        # Example: encourage upright, penalize velocity
        r = cos_theta - 0.1 * theta_dot ** 2

        return r
