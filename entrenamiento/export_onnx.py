import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)


def export_model():
    print("🚀 Cargando 'aquatic_model_final.zip'...")
    model = PPO.load("aquatic_model_final")

    print("📦 Exportando política a ONNX...")
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
    onnxable_model.eval()  # Set to evaluation mode to fix the warning

    # Check the model's expected observation space
    # If the user tries to export a FrameStacked model, input is 512
    # If standard, input is 128. Let's make it automatic!
    obs_dim = model.observation_space.shape[0]
    print(f"Detectado obs_dim de {obs_dim} (¿Frame Stacking = {obs_dim == 512}?)")

    dummy_input = torch.zeros(1, obs_dim, dtype=torch.float32).to(device)

    # Use opset_version=17 to avoid the fallback warnings and conversion errors
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
    export_model()
