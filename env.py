import gym

def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        """
        Creates a new environment instance based on the provided Gym ID.

        Returns:
            env (gym.Env): The created environment instance.

        Raises:
            None

        Description:
            This function creates a new environment instance using the `gym.make()` function. It then wraps the environment
            with the `gym.wrappers.RecordEpisodeStatistics` wrapper to record episode statistics. If the `capture_video`
            flag is set to `True` and the `idx` parameter is equal to 0, the environment is further wrapped with the
            `gym.wrappers.RecordVideo` wrapper to record video of the environment. The seed values for the environment,
            action space, and observation space are set using the provided `seed` value. Finally, the created environment
            instance is returned.

        Note:
            - The `gym_id` parameter is used to specify the Gym ID of the environment to be created.
            - The `seed` parameter is used to set the seed values for the environment, action space, and observation space.
            - The `idx` parameter is used to determine whether or not to record video of the environment.
            - The `capture_video` parameter is a flag that determines whether or not to record video of the environment.
            - The `run_name` parameter is used to specify the name of the video recording.

        Example Usage:
            env = thunk()
        """
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk