from src.utils import load_config
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import os


def train_extension(cfg_path="config/config_extension.yaml"):
    cfg = load_config(cfg_path)
    os.makedirs("results/extension", exist_ok=True)

    env = gym.make(cfg["environment"])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=cfg["extension"]["learning_rate"],
        verbose=1,
        tensorboard_log="logs/extension"
    )

    cb = CheckpointCallback(
        save_freq=cfg["checkpoint_freq"],
        save_path="logs/extension/checkpoints"
    )

    model.learn(total_timesteps=cfg["timesteps"], callback=cb)
    model.save("results/extension/ppo_lr_tuned")

    env.close()


if __name__ == '__main__':
    train_extension()
