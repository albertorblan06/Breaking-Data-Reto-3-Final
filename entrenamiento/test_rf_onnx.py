import onnxruntime as ort
import numpy as np
import os

model_path = os.path.join(os.path.dirname(__file__), "rf_behavioral_clone.onnx")
sess = ort.InferenceSession(model_path)
input_name = sess.get_inputs()[0].name
outputs = sess.run(None, {input_name: np.zeros((1, 128), dtype=np.float32)})
print(f"Output 0 shape: {np.array(outputs[0]).shape}, value: {outputs[0]}")
print(f"Output 1: {outputs[1]}")
