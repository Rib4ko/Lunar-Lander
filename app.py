import gymnasium as gym

# 1. Spin up the game window
env = gym.make("LunarLander-v3", render_mode="human")

# 2. Reset the game to the starting state
obs, info = env.reset()

# 3. Play for 500 frames
for _ in range(500):
    # Pick a random button (0, 1, 2, or 3)
    action = env.action_space.sample()

    # Press the button and see what happens
    obs, reward, terminated, truncated, info = env.step(action)

    # If it crashes or lands, start a fresh round
    if terminated or truncated:
        obs, info = env.reset()

# 4. Close the window cleanly
env.close()