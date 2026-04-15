import os
import numpy as np
import onnxruntime as ort
from interfaz import AgenteBase


class AgenteInferencia(AgenteBase):
    def __init__(self):
        super().__init__(nombre_equipo="Aquatic_Agents_Hunter")
        self.ort_session = None
        self.input_name = None
        self.jab_timer = 0

    def configurar(self):
        model_path = os.path.join(os.path.dirname(__file__), "dense_hunter.onnx")

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1

        try:
            self.ort_session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.ort_session.get_inputs()[0].name
            print(f"[{self.nombre_equipo}] RL Dense Hunter ONNX loaded.")
        except Exception as e:
            print(f"[{self.nombre_equipo}] ONNX Error: {e}")

    def predict(self, estado):
        if self.ort_session is None:
            return 1

        try:
            ram = estado["ram"].copy()
            soy_blanco = estado["soy_blanco"]
            self.jab_timer += 1

            if not soy_blanco:
                ram[32], ram[33] = ram[33], ram[32]
                ram[34], ram[35] = ram[35], ram[34]
                ram[18], ram[19] = ram[19], ram[18]
                ram[107], ram[109] = ram[109], ram[107]
                ram[111], ram[113] = ram[113], ram[111]
                ram[101], ram[105] = ram[105], ram[101]
                ram[103], ram[105] = ram[105], ram[103]

            # Shape for ONNX (Stable Baselines MLP expects float32 typically)
            obs = np.array([ram], dtype=np.float32)

            outputs = self.ort_session.run(None, {self.input_name: obs})
            action_logits = outputs[0]
            action = int(np.argmax(action_logits, axis=1)[0])

            # Apply rhythm to PUNCH to prevent stunning/spamming uselessly
            if action == 1:
                if self.jab_timer >= 3:
                    self.jab_timer = 0
                else:
                    action = 0  # NOOP while winding up jab

            return action

        except Exception as e:
            print(f"[{self.nombre_equipo}] Predict Error: {e}")
            return 0
