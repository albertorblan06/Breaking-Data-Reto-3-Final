import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("Breaking-Data-Reto-3/entrenamiento/unstoppable.onnx")
input_name = sess.get_inputs()[0].name
ram = np.zeros((1, 128), dtype=np.uint8)
ram[0, 32] = 30
ram[0, 34] = 4
ram[0, 33] = 109
ram[0, 35] = 87

outputs = sess.run(None, {input_name: ram})
print("ONNX Output:", np.argmax(outputs[0], axis=1)[0])
