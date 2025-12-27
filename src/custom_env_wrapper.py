import gymnasium as gym
import math
from gymnasium import RewardWrapper


class CustomRewardWrapper(RewardWrapper):
    def __init__(self, env: gym.Env, cfg: dict):
        super().__init__(env)
        self.angle_weight = cfg["reward"]["angle_weight"]
        self.velocity_weight = cfg["reward"]["velocity_weight"]

    def reward(self, reward: float) -> float:
        # Pendulum state: [theta, theta_dot]
        theta, theta_dot = self.env.unwrapped.state

        shaped_reward = (
            self.angle_weight * math.cos(theta)
            - self.velocity_weight * (theta_dot ** 2)
        )

        return shaped_reward
