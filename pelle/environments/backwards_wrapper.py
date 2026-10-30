import gymnasium as gym
import numpy as np

# This overrides the pickup function!
class BackwardsActionWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

        # Keep the same 7-action space
        self.action_space = gym.spaces.Discrete(7)

    def step(self, action):

        # Actions 0, 1, 2, 4, 5, 6:
        # let MiniGrid handle them normally
        if action != 3:
            return self.env.step(action)

        # Action 3 = backwards

        # Save the direction the agent is currently facing
        original_dir = self.env.unwrapped.agent_dir

        # Turn the agent 180 degrees
        self.env.unwrapped.agent_dir = (original_dir + 2) % 4

        # Tell MiniGrid to move forward
        obs, reward, terminated, truncated, info = self.env.step(2)

        # Restore the original direction
        self.env.unwrapped.agent_dir = original_dir

        return obs, reward, terminated, truncated, info