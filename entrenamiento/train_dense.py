import os
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from pettingzoo.atari import boxing_v2
from stable_baselines3 import PPO
import torch


class DenseRewardEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self):
        super().__init__()
        self.pz_env = boxing_v2.env(obs_type="rgb_image")
        self.action_space = Discrete(18)
        self.observation_space = Box(low=0, high=255, shape=(128,), dtype=np.uint8)
        self.last_dist = 1000

    def reset(self, seed=None, options=None):
        self.pz_env.reset()
        self.agent_iterator = iter(self.pz_env.agent_iter())
        next(self.agent_iterator)  # Advance to P1

        ram = self._get_ram()
        self.last_dist = self._calc_dist(ram)
        return ram, {}

    def _get_ram(self):
        try:
            return self.pz_env.unwrapped.ale.getRAM()
        except AttributeError:
            return np.zeros(128, dtype=np.uint8)

    def _calc_dist(self, ram):
        mi_x, mi_y = float(ram[32]), float(ram[34])
        su_x, su_y = float(ram[33]), float(ram[35])
        return np.sqrt((mi_x - su_x) ** 2 + (mi_y - su_y) ** 2)

    def step(self, action):
        self.pz_env.step(action)  # P1
        try:
            next(self.agent_iterator)  # Go to P2
        except StopIteration:
            pass

        obs, _, term, trunc, info = self.pz_env.last()
        if term or trunc:
            self.pz_env.step(None)
        else:
            # P2 plays randomly
            self.pz_env.step(self.pz_env.action_space("second_0").sample())

        try:
            next(self.agent_iterator)  # Back to P1
        except StopIteration:
            pass

        obs, reward_p1, term, trunc, info = self.pz_env.last()
        ram = self._get_ram()

        # --- DENSE REWARD SHAPING ---
        dense_reward = float(reward_p1) * 10.0  # Multiply hit rewards!

        # 1. Approach reward
        current_dist = self._calc_dist(ram)
        if current_dist < self.last_dist:
            dense_reward += 0.05
        elif current_dist > self.last_dist:
            dense_reward -= 0.05

        # 2. Stay in combat range reward
        if 20 <= current_dist <= 35:
            dense_reward += 0.1

        # 3. Corner penalty
        mi_x = float(ram[32])
        if mi_x < 35 or mi_x > 115:
            dense_reward -= 0.1

        self.last_dist = current_dist

        return ram, dense_reward, bool(term), bool(trunc), info


def export_to_onnx(model, onnx_path):
    import torch as th

    class OnnxablePolicy(th.nn.Module):
        def __init__(self, extractor, action_net, value_net):
            super().__init__()
            self.extractor = extractor
            self.action_net = action_net
            self.value_net = value_net

        def forward(self, observation):
            action_hidden, value_hidden = self.extractor(observation)
            return self.action_net(action_hidden), self.value_net(value_hidden)

    onnxable_model = OnnxablePolicy(
        model.policy.mlp_extractor, model.policy.action_net, model.policy.value_net
    )
    dummy_input = th.randn(1, 128)
    th.onnx.export(
        onnxable_model,
        dummy_input,
        onnx_path,
        opset_version=11,
        input_names=["input"],
    )


def train():
    env = DenseRewardEnv()
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0005, n_steps=2048)

    print("Training Dense PPO Hunter against Random (150,000 steps)...")
    model.learn(total_timesteps=150000)

    onnx_path = os.path.join(os.path.dirname(__file__), "dense_hunter.onnx")
    export_to_onnx(model, onnx_path)
    print("Done!")


if __name__ == "__main__":
    train()
