import gymnasium as gym
import numpy as np
from gymnasium import RewardWrapper


class CustomRewardWrapper(RewardWrapper):

    def __init__(self, env: gym.Env, cfg: dict):
        super().__init__(env)
        self.cfg = cfg

    def reward(self, reward: float) -> float:
        # Observation: [cos(theta), sin(theta), theta_dot]
        cos_theta, sin_theta, theta_dot = self.env.unwrapped.state
        theta = np.arctan2(sin_theta, cos_theta)

        angle_penalty = self.cfg["reward"]["angle_weight"] * (theta ** 2)
        velocity_penalty = self.cfg["reward"]["velocity_weight"] * (theta_dot ** 2)

        shaped_reward = - (angle_penalty + velocity_penalty)

        return shaped_reward
