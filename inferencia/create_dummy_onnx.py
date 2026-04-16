import torch
import torch.nn as nn
import os

class DummyModel(nn.Module):
    def forward(self, x):
        return x

model = DummyModel()
dummy_input = torch.randn(1, 210, 160, 3) # Image size from Atari

onnx_path = "/Users/albertorblan/Projects/hackaton/Breaking-Data-Reto-3-Final/inferencia/modelos/Aquatic_Agents/modelo.onnx"
torch.onnx.export(model, dummy_input, onnx_path, verbose=False, input_names=['input'], output_names=['output'])
print("Dummy ONNX model created at:", onnx_path)
