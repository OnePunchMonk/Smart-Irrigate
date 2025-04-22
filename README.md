# Step 0

.\venv\Scripts\activate


# Step 1: Train and export ONNX model
python model_train_and_export.py

# Step 2: Build and run Docker
docker build -t irrigation-model .
docker run -p 8000:8000 irrigation-model

# Step 3: Test with your own input
curl -X POST http://localhost:8000/predict -H "Content-Type: multipart/form-data" -F "file=@test_input.txt"



### Test the API from CLI
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_input.txt"


