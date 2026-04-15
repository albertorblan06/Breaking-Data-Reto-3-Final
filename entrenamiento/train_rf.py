import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def train_rf():
    data_path = os.path.join(os.path.dirname(__file__), "expert_dataset.npz")
    print(f"Loading dataset from {data_path}...")
    data = np.load(data_path)

    # We keep it as float32 because ONNX models typically expect float inputs
    obs = data["obs"].astype(np.float32)
    acts = data["acts"].astype(np.int64)

    print("Training Random Forest Classifier on 128 RAM features...")
    # 50 trees should be plenty to memorize a strict logic table, and it'll run in <1ms
    rf = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1, max_depth=None
    )
    rf.fit(obs, acts)

    acc = rf.score(obs, acts)
    print(f"Training Accuracy: {acc * 100:.2f}%")

    # Convert to ONNX
    print("Converting RF to ONNX...")
    initial_type = [("float_input", FloatTensorType([None, 128]))]

    # Target opset 11 to ensure maximum compatibility with any onnxruntime version
    onx = convert_sklearn(rf, initial_types=initial_type, target_opset=11)

    onnx_path = os.path.join(os.path.dirname(__file__), "rf_behavioral_clone.onnx")
    with open(onnx_path, "wb") as f:
        f.write(onx.SerializeToString())

    print(f"Model exported successfully to {onnx_path}")


if __name__ == "__main__":
    train_rf()
