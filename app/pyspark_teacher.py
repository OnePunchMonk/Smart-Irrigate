from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np

# Initialize Spark
spark = SparkSession.builder.appName("IrrigationPipeline").getOrCreate()

# Load dataset
df_spark = spark.read.csv("dataset/Irrigation Scheduling.csv", header=True, inferSchema=True)

# Drop unnecessary columns and handle nulls
df_spark = df_spark.drop("id", "date", "time")
df_spark = df_spark.na.fill({"altitude": df_spark.select("altitude").agg({"altitude": "mean"}).first()[0]})

# Convert to Pandas for label encoding and scaling
df = df_spark.toPandas()

# Encode labels
le = LabelEncoder()
df["class_encoded"] = le.fit_transform(df["class"])

# Feature scaling
X = df.drop(columns=["class", "class_encoded"])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = df["class_encoded"]

# Create final DataFrame
df_final = pd.DataFrame(X_scaled, columns=X.columns)
df_final["label"] = y

# Save scaled test set to simulate Kafka stream
train_df = df_final.sample(frac=0.8, random_state=42)
test_df = df_final.drop(train_df.index)

train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_stream.csv", index=False)

