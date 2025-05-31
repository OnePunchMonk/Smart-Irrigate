from fastapi import FastAPI, File
import numpy as np
import onnxruntime as ort

app = FastAPI()
session = ort.InferenceSession("student_model.onnx")

@app.post("/predict")
def predict_from_file(file: bytes = File(...)):
    content = file.decode("utf-8").strip()
    features = np.array([list(map(float, content.split(',')))], dtype=np.float32)
    outputs = session.run(["output"], {"input": features})
    predicted_class = int(np.argmax(outputs[0]))
    return {"predicted_class": predicted_class}

