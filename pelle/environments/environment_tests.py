import gymnasium as gym
import minigrid
from backwards_wrapper import BackwardsActionWrapper


env = gym.make(
    "MiniGrid-WFC-MazeSimple-v0",
    render_mode="human"
)

env = BackwardsActionWrapper(env)

obs, info = env.reset(seed=42)
print("Action space:", env.action_space)

for _ in range(10):

    action = 3

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

env.close()