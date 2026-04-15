import os
import numpy as np
import gymnasium as gym
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import torch


class BoxingRewardWrapper(gym.Wrapper):
    """
    Reward Shaping Wrapper for Atari Boxing (RAM).
    Incentivizes aggressive play and penalizes passivity.
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 1. Base Game Objective Multiplier
        # Multiply by 10 to ensure landing punches (+10) and getting hit (-10)
        # remain the absolute highest priority over the small positional rewards.
        custom_reward = reward * 10.0

        # 2. Passivity Penalty
        # Bleed a tiny bit of score every frame. Prevents tying 0-0.
        custom_reward -= 0.001

        # In Boxing-v5 single-agent mode, the RL agent ALWAYS trains as Player 1 (White).
        mi_x = int(obs[32])
        su_x = int(obs[33])
        mi_y = int(obs[34])
        su_y = int(obs[35])

        dist_x = abs(su_x - mi_x)
        dist_y = abs(su_y - mi_y)

        # 3. The Sweet Spot Positioning Reward
        if 24 <= dist_x <= 28 and dist_y <= 10:
            custom_reward += 0.005  # Good boy, you are in range!

            # 4. Action 1 (PUNCH) Reward
            # Prevent "Reward Hacking" where it just stands in the sweet spot and never swings
            if action == 1:
                custom_reward += 0.01

        return obs, custom_reward, terminated, truncated, info


def train():
    print(
        "🚀 Iniciando entrenamiento extendido (5M steps) de Aquatic Agents sobre RAM (FRAME STACKING + REWARD SHAPING)..."
    )

    gym.register_envs(ale_py)

    def make_env():
        env = gym.make("ALE/Boxing-v5", obs_type="ram")
        env = BoxingRewardWrapper(env)  # Inject custom rewards before stacking!
        return env

    # The velocity fix: Frame Stacking! We stack 4 consecutive RAM frames
    # so the MlpPolicy can infer direction and velocity.
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4)

    # Guardar checkpoints cada 500,000 pasos
    checkpoint_callback = CheckpointCallback(
        save_freq=500000, save_path="./logs_aquatic/", name_prefix="aquatic_model"
    )

    # Parametrización más agresiva para RL
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.00025,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        tensorboard_log="./aquatic_tensorboard/",
    )

    print("🥊 Entrenando (5,000,000 timesteps)...")
    model.learn(total_timesteps=5000000, callback=checkpoint_callback)

    model.save("aquatic_model_final")
    print("✅ Modelo final guardado como 'aquatic_model_final.zip'")

    # --- EXPORTACIÓN A ONNX ---
    print("📦 Exportando a ONNX...")

    device = torch.device("cpu")
    model.policy.to(device)

    class OnnxablePolicy(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.features_extractor = policy.features_extractor
            self.mlp_extractor = policy.mlp_extractor
            self.action_net = policy.action_net

        def forward(self, observation):
            features = self.features_extractor(observation)
            latent_pi, _ = self.mlp_extractor(features)
            logits = self.action_net(latent_pi)
            return torch.argmax(logits, dim=1)

    onnxable_model = OnnxablePolicy(model.policy)
    onnxable_model.eval()

    # Notice the dummy input is now 512 (128 bytes x 4 frames)
    dummy_input = torch.zeros(1, 512, dtype=torch.float32).to(device)

    torch.onnx.export(
        onnxable_model,
        dummy_input,
        "modelo_aquatic.onnx",
        opset_version=17,
        input_names=["input"],
        output_names=["action"],
    )
    print("🔥 ¡Listo! Tienes tu 'cerebro' en modelo_aquatic.onnx")


if __name__ == "__main__":
    train()
