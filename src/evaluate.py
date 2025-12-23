import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C, SAC


def evaluate(model_path, env_id, episodes=10):
    """
    Evaluate a trained RL agent without exploration.
    """

    # Automatically detect algorithm from saved model
    if "PPO" in model_path or "ppo" in model_path:
        model = PPO.load(model_path)
    elif "DQN" in model_path or "dqn" in model_path:
        model = DQN.load(model_path)
    elif "A2C" in model_path or "a2c" in model_path:
        model = A2C.load(model_path)
    elif "SAC" in model_path or "sac" in model_path:
        model = SAC.load(model_path)
    else:
        raise ValueError("Unknown algorithm in model path")

    env = gym.make(env_id)

    episode_rewards = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            # IMPORTANT: deterministic=True disables exploration
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            total_reward += reward
            steps += 1

        episode_rewards.append(total_reward)
        print(f"Episode {ep + 1}: reward = {total_reward:.2f}, steps = {steps}")

    env.close()

    mean_reward = sum(episode_rewards) / len(episode_rewards)
    std_reward = (sum((r - mean_reward) ** 2 for r in episode_rewards) / len(episode_rewards)) ** 0.5

    print(f"\nEvaluation over {episodes} episodes")
    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Std reward: {std_reward:.2f}")

    return mean_reward, std_reward
