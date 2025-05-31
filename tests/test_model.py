# tests/test_model.py

import os
from app.model_utils import load_session, predict_from_text

def test_onnx_model_prediction():
    session = load_session()

    test_input_path = os.path.join(os.path.dirname(__file__), "test_input.txt")
    with open(test_input_path, "r") as f:
        content = f.read()

    predicted = predict_from_text(session, content)
    assert isinstance(predicted, int)
    assert predicted in [0, 1, 2, 3]  # adjust if more classes

    print(f"Predicted class: {predicted}")