from environments import make_env

ENV_ID = "MiniGrid-WFC-MazeSimple-v0"

def main():

    # Create the environment.
    env = make_env(
        EnvString=ENV_ID,
        edge_length=10,
        render_mode="human"
    )

    try:
        # Start a new episode.
        obs, info = env.reset(seed=0)
        env.render()
        # Make the random action sequence reproducible.
        env.action_space.seed(0)

        # Print information about the environment.
        print("Environment:", ENV_ID)
        print("Observation space:", env.observation_space)
        print("Action space:", env.action_space)

        # Inspect the observation.
        print("\nInitial observation:")

        if isinstance(obs, dict):
            for key, value in obs.items():

                if hasattr(value, "shape"):
                    print(
                        f"{key}: shape={value.shape}, "
                        f"dtype={value.dtype}"
                    )
                else:
                    print(f"{key}: {value!r}")

        else:
            print("Shape:", obs.shape)
            print("Data type:", obs.dtype)

        # Let the agent take random actions.
        print("\nStarting random actions...")

        for step in range(50):

            # Choose a random action.
            action = env.action_space.sample()

            # Execute the action.
            obs, reward, terminated, truncated, info = env.step(action)

            # Report any non-zero reward.
            if reward != 0:
                print(f"Step {step + 1}: reward = {reward:.3f}")

            # Restart the episode when it ends.
            if terminated or truncated:
                print(f"Episode ended at step {step + 1}.")
                obs, info = env.reset()

            print('Finished Episode')
            
    finally:
        # Close the environment when the program finishes.
        env.close()


if __name__ == "__main__":
    main()