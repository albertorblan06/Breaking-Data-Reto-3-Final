import gymnasium as gym
import ale_py
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

gym.register_envs(ale_py)


def make_env():
    return gym.make("ALE/Boxing-v5", obs_type="ram")


env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=4)

obs = env.reset()
print("Stacked observation shape:", obs.shape)
