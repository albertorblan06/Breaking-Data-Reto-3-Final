import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("Breaking-Data-Reto-3/entrenamiento/dense_hunter.onnx")
input_name = sess.get_inputs()[0].name
dummy_input = np.zeros((1, 128), dtype=np.float32)
outputs = sess.run(None, {input_name: dummy_input})
print(outputs[0].shape)
print(outputs[0])
