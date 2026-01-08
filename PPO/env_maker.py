
import gymnasium as gym
from env_wrapper import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
    Normalize
)


def make_atari_env(env_id, idx=0, capture_video=False):
    """Helper function to create an environment with some standard wrappers.

    Taken from cleanRL's DQN Atari implementation: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py.
    """
    if capture_video and idx == 0:
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, f"videos/{env_id}")
    else:
        env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    # env.action_space.seed(seed)

    return env


def make_mujoco_env(env_id, idx=0, capture_video=False):
    """Helper function to create an environment with some standard wrappers.

    Taken from cleanRL's DQN Atari implementation: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py.
    """

    env = gym.make(env_id)

    env = gym.wrappers.ClipAction(env)
    # env = Normalize(env)
    # env.action_space.seed(seed)

    return env
