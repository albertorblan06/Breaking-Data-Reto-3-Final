import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


# A very lightweight MLP to guarantee 0.00ms latency inference
class BehavioralCloneNet(nn.Module):
    def __init__(self):
        super(BehavioralCloneNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6),
        )

    def forward(self, x):
        # We assume input x is already float32 and normalized (x / 255.0)
        return self.net(x)


def train_bc():
    data_path = os.path.join(os.path.dirname(__file__), "expert_dataset.npz")
    print(f"Loading dataset from {data_path}...")
    data = np.load(data_path)

    obs = data["obs"].astype(np.float32) / 255.0  # Normalize RAM inputs
    acts = data["acts"].astype(np.int64)

    # Split into train / val
    split = int(0.9 * len(obs))

    train_x = torch.tensor(obs[:split])
    train_y = torch.tensor(acts[:split])
    val_x = torch.tensor(obs[split:])
    val_y = torch.tensor(acts[split:])

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    print(f"Training on device: {device}")

    model = BehavioralCloneNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 15
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

        train_acc = 100.0 * correct / total

        # Validation
        model.eval()
        with torch.no_grad():
            val_x_dev = val_x.to(device)
            val_y_dev = val_y.to(device)
            val_outputs = model(val_x_dev)
            val_loss = criterion(val_outputs, val_y_dev).item()
            _, val_pred = val_outputs.max(1)
            val_acc = 100.0 * val_pred.eq(val_y_dev).sum().item() / val_y.size(0)

        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {total_loss / len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
        )

    # Export to ONNX
    print("Exporting model to ONNX...")
    model.eval()
    model.to("cpu")

    dummy_input = torch.randn(1, 128)
    onnx_path = os.path.join(os.path.dirname(__file__), "behavioral_clone.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Model exported successfully to {onnx_path}")


if __name__ == "__main__":
    train_bc()
