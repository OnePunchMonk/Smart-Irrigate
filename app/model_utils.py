# app/model_utils.py

import numpy as np
import onnxruntime as ort

MODEL_PATH = "artifacts/student_model.onnx"

def load_session():
    return ort.InferenceSession(MODEL_PATH)

def predict_from_text(session, raw_text: str):
    features = np.array([list(map(float, raw_text.strip().split(',')))], dtype=np.float32)
    outputs = session.run(["output"], {"input": features})
    return int(np.argmax(outputs[0]))
