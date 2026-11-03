import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("LunarLander-v3", render_mode="human")

model = PPO("MlpPolicy", env , verbose=1)

model.learn(total_timesteps=100000)

env.close()
