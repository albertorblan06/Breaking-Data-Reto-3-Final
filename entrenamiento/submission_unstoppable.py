import os, numpy as np, onnxruntime as ort
from interfaz import AgenteBase

class AgenteInferencia(AgenteBase):
    def __init__(self):
        self.ort_session = None
        self.input_name = None
        self.jab_timer = 0
        super().__init__(nombre_equipo="Aquatic_Agents_Unstoppable")

    def configurar(self):
        model_path = os.path.join(os.path.dirname(__file__), "modelo.onnx")
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        try:
            self.ort_session = ort.InferenceSession(model_path, sess_options=sess_options, providers=["CPUExecutionProvider"])
            self.input_name = self.ort_session.get_inputs()[0].name
        except Exception as e: print(e)

    def predict(self, estado):
        if self.ort_session is None: return 1
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
            obs = np.array([ram], dtype=np.uint8)
            outputs = self.ort_session.run(None, {self.input_name: obs})
            action = int(np.argmax(outputs[0], axis=1)[0])
            
            # Action has 'FIRE' if it's 1 or >= 10
            is_fire = (action == 1) or (action >= 10)
            
            if is_fire:
                if self.jab_timer >= 4:
                    self.jab_timer = 0
                else:
                    # Strip FIRE component
                    if action == 1: action = 0
                    elif action == 10: action = 2
                    elif action == 11: action = 3
                    elif action == 12: action = 4
                    elif action == 13: action = 5
                    elif action == 14: action = 6
                    elif action == 15: action = 7
                    elif action == 16: action = 8
                    elif action == 17: action = 9
            return action
        except: return 0
