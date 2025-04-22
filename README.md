
# AIHT-Project

An AI-powered irrigation scheduling system designed to optimize water usage in agriculture by leveraging machine learning, data analytics, and cross-language support including R and PySpark.

## 🌱 Overview

This project aims to enhance agricultural productivity by providing intelligent irrigation recommendations.  
By analyzing environmental and crop data, the system determines optimal watering schedules, ensuring efficient water utilization and promoting sustainable farming practices.

## 🧠 Features

- **Machine Learning Models**: Implements both teacher and student models using PyTorch and ONNX for efficient inference.
- **Data Analysis**: Utilizes Jupyter notebooks and R scripts for exploratory data analysis to understand and preprocess the dataset.
- **Big Data Processing**: Employs PySpark for training large-scale teacher models.
- **API Service**: Provides a FastAPI-based backend to serve model predictions.
- **Docker Integration**: Includes Docker support for containerized deployment.
- **Streamlit GUI**: Offers a GUI for user to specify sensor readings and recieve prediction.
- **Power BI Dashboard**: Offers a Power BI report for visualizing irrigation schedules and model insights.

## 📁 Project Structure

```plaintext
├── app.py                       # FastAPI application
├── Dockerfile                   # Docker configuration
├── eda-aihtproject.ipynb        # Exploratory Data Analysis notebook (Python)
├── analysis_r_code.R            # Exploratory Data Analysis in R
├── frontend.py                  # Frontend interface (Streamlit)
├── irrigation_aiht_powerbi.pbix # Power BI dashboard file
├── Irrigation Scheduling.csv    # Dataset for training and analysis
├── model_train_and_export.py    # Script to train and export models
├── pyspark_teacher.py           # PySpark implementation for teacher model
├── requirements.txt             # Python dependencies
├── student_model.onnx           # Exported student model in ONNX format
├── student_model.pt             # Trained student model in PyTorch format
├── teacher.py                   # Teacher model training script
├── teacher_model.pt             # Trained teacher model in PyTorch format
├── test_input.txt               # Sample input for testing
└── README.md                    # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher  
- Docker (for containerized deployment)  
- R (for R-based data analysis)  
- Apache Spark (for running PySpark)
- Streamlit (for running GUI)  

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/OnePunchMonk/AIHT-Project.git
   cd AIHT-Project
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install R packages (for R analysis)**

   ```r
   install.packages("tidyverse")
   install.packages("data.table")
   ```

## 🏋️‍♂️ Training and Exporting the Model

- **Train and export to ONNX**  
  ```bash
  python model_train_and_export.py
  ```

- **Run the teacher model using PySpark**  
  ```bash
  spark-submit pyspark_teacher.py
  ```

## 🐳 Docker Deployment

1. **Build the Docker image**

   ```bash
   docker build -t irrigation-model .
   ```

2. **Run the Docker container**

   ```bash
   docker run -p 8000:8000 irrigation-model
   ```

## 🧪 API Testing

You can test the prediction endpoint using `curl` or any API testing client:

```bash
curl -X POST http://localhost:8000/predict   -H "Content-Type: multipart/form-data"   -F "file=@test_input.txt"
```

## 📊 Data Analysis

- `eda-aihtproject.ipynb`: R script for additional statistical analysis and visualizations.
(Run on Kaggle env)

## 📈 Visualization

- `irrigation_aiht_powerbi.pbix`: Power BI dashboard for irrigation schedules and model insights.
- `frontend.py`: GUI for non-technical users.

## 🔗 Wokwi Simulation

For a hardware simulation of the irrigation system, refer to the [AIHT Wokwi Simulation Repository](https://github.com/chaitanyadav69/AIHT).

https://wokwi.com/projects/428958546126055425

## Streamlit Deployment on HF Spaces , refer[this link](https://huggingface.co/spaces/OnePunchMonk101010/smart-irrigation).
---

*Note: Ensure all environment variables and configurations are set appropriately before deploying the application.*
