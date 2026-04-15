import sys
import os
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
from pettingzoo.atari import boxing_v2
from stable_baselines3 import PPO
import torch

# Add inferencia to the path to import the arena agents
ruta_inferencia = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "inferencia")
)
sys.path.append(ruta_inferencia)
from arena import cargar_agente_desde_carpeta


def extraer_ram_segura(env):
    unwrapped = env.unwrapped
    try:
        return unwrapped.ale.getRAM()
    except AttributeError:
        return np.zeros(128, dtype=np.uint8)


class LeagueEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, opponent_name="equipo_random"):
        super().__init__()

        self.opponent_name = opponent_name
        self.opponent = cargar_agente_desde_carpeta(
            os.path.join(ruta_inferencia, f"modelos/{opponent_name}")
        )
        if hasattr(self.opponent, "configurar"):
            self.opponent.configurar()

        self.pz_env = boxing_v2.env(obs_type="rgb_image")

        self.action_space = Discrete(18)
        self.observation_space = Box(low=0, high=255, shape=(128,), dtype=np.uint8)

    def reset(self, seed=None, options=None):
        self.pz_env.reset()
        self.agent_iterator = iter(self.pz_env.agent_iter())
        next(self.agent_iterator)  # Advance to first_0

        obs, reward, term, trunc, info = self.pz_env.last()
        ram = extraer_ram_segura(self.pz_env)
        return ram, {}

    def step(self, action):
        self.pz_env.step(action)  # P1

        try:
            next(self.agent_iterator)  # Go to P2
        except StopIteration:
            pass

        obs, reward_p2, term, trunc, info = self.pz_env.last()

        if term or trunc:
            self.pz_env.step(None)
        else:
            ram = extraer_ram_segura(self.pz_env)
            estado = {"imagen": obs, "ram": ram, "soy_blanco": False}
            try:
                action_p2 = self.opponent.predecir(estado)
            except AttributeError:
                try:
                    action_p2 = self.opponent.predict(estado)
                except Exception:
                    action_p2 = 0
            except Exception:
                action_p2 = 0

            # Ensure action is integer
            if isinstance(action_p2, np.ndarray):
                action_p2 = int(action_p2.item())

            self.pz_env.step(action_p2)

        try:
            next(self.agent_iterator)  # Back to P1
        except StopIteration:
            pass

        obs, reward_p1, term, trunc, info = self.pz_env.last()
        ram = extraer_ram_segura(self.pz_env)

        return ram, float(reward_p1), bool(term), bool(trunc), info


def export_to_onnx(model, onnx_path):
    import torch as th

    class OnnxablePolicy(th.nn.Module):
        def __init__(self, extractor, action_net, value_net):
            super().__init__()
            self.extractor = extractor
            self.action_net = action_net
            self.value_net = value_net

        def forward(self, observation):
            # NOTE: You may have to process (normalize) observation in the correct
            #       way before using this. See `common.preprocessing.preprocess_obs`
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
    print(f"✅ ONNX model saved to {onnx_path}")


def train():
    print("🚀 Iniciando entrenamiento Curriculum Learning (League Play)")

    curriculum = [
        ("equipo_random", 150000),
        ("equipo_onnx", 150000),
        ("Aquatic_Agents", 300000),
    ]

    model = None

    for phase, (opponent, timesteps) in enumerate(curriculum):
        print(
            f"\n--- Fase {phase + 1}: Entrenando contra {opponent} ({timesteps} steps) ---"
        )
        env = LeagueEnv(opponent)

        if model is None:
            model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048)
        else:
            model.set_env(env)

        model.learn(total_timesteps=timesteps)
        model.save(f"boxing_model_phase_{phase + 1}_{opponent}")
        print(
            f"✅ Modelo guardado como 'boxing_model_phase_{phase + 1}_{opponent}.zip'"
        )

    final_model_path = "boxing_model_league_final"
    model.save(final_model_path)
    print(
        f"\n✅ Entrenamiento completado. Modelo final guardado como '{final_model_path}.zip'"
    )

    # Export to ONNX
    onnx_path = os.path.join(
        ruta_inferencia, "modelos", "Aquatic_Agents", "league_model.onnx"
    )
    export_to_onnx(model, onnx_path)


if __name__ == "__main__":
    train()
