from src.custom_env_wrapper import CustomRewardWrapper
from src.utils import load_config
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import os


def train_custom(cfg_path="config/config_custom.yaml"):
    cfg = load_config(cfg_path)
    os.makedirs("results/custom", exist_ok=True)

    for trial in range(3):
        env = gym.make(cfg["environment"])
        env = CustomRewardWrapper(env, cfg)

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=trial,
            tensorboard_log=f"logs/custom/trial_{trial}"
        )

        cb = CheckpointCallback(
            save_freq=10000,
            save_path=f"logs/custom/trial_{trial}/checkpoints"
        )

        model.learn(total_timesteps=cfg["timesteps"], callback=cb)
        model.save(f"results/custom/ppo_custom_trial_{trial}")

        env.close()


if __name__ == '__main__':
    train_custom()
