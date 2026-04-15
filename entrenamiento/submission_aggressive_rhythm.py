import os
import onnxruntime as ort
import numpy as np


class AgenteInferencia:
    def __init__(self, ruta_modelo: str = None):
        self.ruta_modelo = ruta_modelo or os.path.dirname(os.path.abspath(__file__))
        self.session = None
        self.configurar()

    def configurar(self):
        modelo_path = os.path.join(self.ruta_modelo, "aggressive_rhythm.onnx")
        if os.path.exists(modelo_path):
            self.session = ort.InferenceSession(modelo_path)
        else:
            raise FileNotFoundError(f"Modelo no encontrado en {modelo_path}")

    def predict(self, estado: dict) -> int:
        ram = np.array(estado["ram"], dtype=np.uint8).reshape(1, -1)
        if self.session is None:
            self.configurar()

        inputs = {self.session.get_inputs()[0].name: ram}
        action_logits = self.session.run(None, inputs)[0]
        return int(np.argmax(action_logits))
