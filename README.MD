To use the project, simply run the following:

```` bash
pip install stable-baselines3 gymnasium tensorboard
python -m src.train_baseline
python -m src.train_custom_reward
python -m src.train_extension
````

To evaluate
````
# Baseline
evaluate("results/baseline/ppo_trial_0", "Pendulum-v1", episodes=20)

# Custom Reward
evaluate("results/custom/ppo_custom_trial_0", "Pendulum-v1", episodes=20)

# Extension
evaluate("results/extension/ppo_lr_tuned", "Pendulum-v1", episodes=20)
````
