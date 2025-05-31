#!pip install Flask onnxruntime numpy

import flask
import onnxruntime as ort
import numpy as np
import os
import csv
from datetime import datetime
import threading

app = flask.Flask(__name__)

# --- Configuration ---
MODEL_PATH = 'artifiacts/student_model.onnx'  # Make sure this file is deployed with your app
INPUT_NAME = None
OUTPUT_NAME = None
EXPECTED_INPUT_SHAPE = (1, 6)
LOG_FILE = 'data_log.csv'  # <-- CSV log file
# --- End Configuration ---

session = None

def load_model():
    global session, INPUT_NAME, OUTPUT_NAME
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        return False
    try:
        print(f"Loading ONNX model from {MODEL_PATH}...")
        session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        INPUT_NAME = session.get_inputs()[0].name
        OUTPUT_NAME = session.get_outputs()[0].name
        print(f"Model loaded successfully.")
        print(f"Input Name: {INPUT_NAME}, Expected Shape: {EXPECTED_INPUT_SHAPE}")
        print(f"Output Name: {OUTPUT_NAME}")
        return True
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        session = None
        return False

def log_to_csv(input_values, output_data):
    def write_row():
        try:
            file_exists = os.path.isfile(LOG_FILE)
            with open(LOG_FILE, mode='a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow( # noqa
                        ['timestamp'] # noqa
                        + [f'input_{i+1}' for i in range(6)] # noqa
                        + [f'output_{i+1}' for i in range(4)] # noqa
                                    ) # noqa                                               
                writer.writerow([datetime.now()] + input_values + output_data)
        except Exception as e:
            print(f"[LOG ERROR] Failed to write to CSV: {e}")

    # Run logging in a background thread
    threading.Thread(target=write_row).start()

@app.route('/infer', methods=['POST'])
def infer():
    if session is None:
        print("Model not loaded, attempting reload...")
        if not load_model():
            return flask.jsonify({"error": "Model not loaded on server"}), 500

    try:
        data = flask.request.get_json()
        if not data or 'inputs' not in data:
            return flask.jsonify({"error": "Missing 'inputs' key in JSON payload"}), 400

        input_values = data['inputs']

        if not isinstance(input_values, list) or len(input_values) != EXPECTED_INPUT_SHAPE[1]:
            return flask.jsonify({"error":
                                  f"Expected a list of {EXPECTED_INPUT_SHAPE[1]} float values in 'inputs'"
                                  }), 400

        input_array = np.array(input_values, dtype=np.float32).reshape(EXPECTED_INPUT_SHAPE)
        feeds = {INPUT_NAME: input_array}
        results = session.run([OUTPUT_NAME], feeds)
        output_data = results[0].flatten().tolist()

        log_to_csv(input_values, output_data)

        print(f"Received input: {input_values}, Produced output: {output_data}")
        return flask.jsonify({"outputs": output_data})

    except Exception as e:
        print(f"Error during inference: {e}")
        return flask.jsonify({"error": f"Inference error: {str(e)}"}), 500

@app.route('/')
def home():
    status = "Model Loaded" if session else "Model NOT Loaded"
    return f"ONNX Inference Server is running. Status: {status}"

if not load_model():
    print("CRITICAL: Failed to load model on startup.")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# from flask import Flask, request, jsonify
# import onnxruntime as ort
# import numpy as np
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col
# import os

# app = Flask(__name__)

# # --- Configuration ---
# MODEL_PATH = 'student_model.onnx'  # Path to your ONNX model
# EXPECTED_INPUT_SHAPE = (1, 6)  # Expected shape of the input data (batch_size=1, 6 features)
# INPUT_NAME = None  # Will be fetched from the model
# OUTPUT_NAME = None  # Will be fetched from the model
# # --- End Configuration ---

# # Initialize PySpark session
# spark = SparkSession.builder.appName("FlaskAppWithPySpark").getOrCreate()

# # Load ONNX model
# session = None

# def load_model():
#     global session, INPUT_NAME, OUTPUT_NAME
#     if not os.path.exists(MODEL_PATH):
#         print(f"ERROR: Model file not found at {MODEL_PATH}")
#         return False
#     try:
#         print(f"Loading ONNX model from {MODEL_PATH}...")
#         session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
#         INPUT_NAME = session.get_inputs()[0].name
#         OUTPUT_NAME = session.get_outputs()[0].name
#         print(f"Model loaded successfully.")
#         return True
#     except Exception as e:
#         print(f"Error loading ONNX model: {e}")
#         session = None
#         return False

# @app.route('/infer', methods=['POST'])
# def infer():
#     if session is None:
#         print("Model not loaded, attempting reload...")
#         if not load_model():
#             return jsonify({"error": "Model not loaded on server"}), 500

#     try:
#         # 1. Receive the input data as JSON
#         data = request.get_json()
#         if not data or 'inputs' not in data:
#             return jsonify({"error": "Missing 'inputs' key in JSON payload"}), 400

#         input_values = data['inputs']

#         # 2. Apply PySpark transformations (in-memory processing)
#         # Convert input data into a PySpark DataFrame (Simulating real-time data processing)
#         input_df = spark.createDataFrame([(input_values,)], ['inputs'])

#         # Apply any transformations you need here, for example:
#         # For example, if you need to normalize/scale features, apply transformations.
#         transformed_df = input_df.withColumn('normalized_inputs', col('inputs') / 100)  # Example transformation

#         # Collect the processed data back into a list (after transformation)
#         processed_input = transformed_df.collect()[0]['normalized_inputs']

#         # 3. Convert the processed data into the format required by the ONNX model (numpy array)
#         input_array = np.array(processed_input, dtype=np.float32).reshape(EXPECTED_INPUT_SHAPE)

#         # 4. Run inference using the ONNX model
#         feeds = {INPUT_NAME: input_array}
#         results = session.run([OUTPUT_NAME], feeds)
#         output_data = results[0].flatten().tolist()  # Flatten and convert to a list

#         print(f"Received input: {input_values}, Processed input: {processed_input}, Produced output: {output_data}")

#         # 5. Return the output as JSON response
#         return jsonify({"outputs": output_data})

#     except Exception as e:
#         print(f"Error during inference: {e}")
#         return jsonify({"error": f"Inference error: {str(e)}"}), 500

# @app.route('/')
# def home():
#     status = "Model Loaded" if session else "Model NOT Loaded"
#     return f"ONNX Inference Server is running. Status: {status}"

# # Load the model when the Flask app starts
# if not load_model():
#     print("CRITICAL: Failed to load model on startup.")

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

