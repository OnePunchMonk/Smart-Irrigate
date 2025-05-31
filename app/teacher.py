import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Load dataset
df = pd.read_csv("dataset/Irrigation Scheduling.csv")
df = df.drop(columns=["id", "date", "time"])
df["altitude"].fillna(df["altitude"].mean(), inplace=True)

# Encode class labels
le = LabelEncoder()
df["class_encoded"] = le.fit_transform(df["class"])

# Features and labels
X = df.drop(columns=["class", "class_encoded"])
y = df["class_encoded"]
X = StandardScaler().fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)

# Define a deeper teacher model
class TeacherNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TeacherNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.out = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)

# Instantiate model
teacher = TeacherNet(X_train.shape[1], len(np.unique(y)))
optimizer = torch.optim.Adam(teacher.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train the teacher
for epoch in range(25):
    teacher.train()
    total_loss = 0
    for xb, yb in train_loader:
        logits = teacher(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Save the model
torch.save(teacher, "teacher_model.pt")
print("✅ Teacher model saved as 'teacher_model.pt'")
