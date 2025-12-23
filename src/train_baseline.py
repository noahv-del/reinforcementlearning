from stable_baselines3 import PPO
import gymnasium as gym
from src.utils import load_config
from stable_baselines3.common.callbacks import CheckpointCallback
import os


def train_baseline(cfg_path="config/config_baseline.yaml"):
    cfg = load_config(cfg_path)
    os.makedirs("results/baseline", exist_ok=True)

    for trial in range(cfg["num_trials"]):
        env = gym.make(cfg["environment"])

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=cfg["seed"] + trial,
            tensorboard_log=f"logs/baseline/trial_{trial}"
        )

        cb = CheckpointCallback(
            save_freq=cfg["checkpoint_freq"],
            save_path=f"logs/baseline/trial_{trial}/checkpoints"
        )

        model.learn(total_timesteps=cfg["timesteps"], callback=cb)
        model.save(f"results/baseline/ppo_trial_{trial}")

        env.close()


if __name__ == '__main__':
    train_baseline()
