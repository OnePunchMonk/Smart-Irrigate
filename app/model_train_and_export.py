import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Load data
df = pd.read_csv("dataset/Irrigation Scheduling.csv")
df = df.drop(columns=["id", "date", "time"])
df["altitude"].fillna(df["altitude"].mean(), inplace=True)
label_encoder = LabelEncoder()
df["class_encoded"] = label_encoder.fit_transform(df["class"])

X = df.drop(columns=["class", "class_encoded"])
y = df["class_encoded"]
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)

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

# Define student model
class StudentNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(StudentNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.out = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.out(x)

# Distillation training
student = StudentNet(X_train.shape[1], len(np.unique(y)))
# teacher = torch.load("teacher_model.pt")

import torch
from torch.serialization import add_safe_globals

# Register the class if it's a custom class
add_safe_globals([TeacherNet])

teacher = torch.load("artifacts/teacher_model.pt", weights_only=False)
opt = torch.optim.Adam(student.parameters(), lr=0.001)

T, alpha = 3.0, 0.7

def distill_loss(s_logit, t_logit, labels, T, alpha):
    h = F.cross_entropy(s_logit, labels)
    t_soft = F.log_softmax(t_logit / T, dim=1)
    s_soft = F.log_softmax(s_logit / T, dim=1)
    s_loss = F.kl_div(s_soft, t_soft, log_target=True, reduction='batchmean') * (T**2)
    return alpha * s_loss + (1 - alpha) * h

for epoch in range(20):
    student.train()
    teacher.eval()
    for xb, yb in train_loader:
        with torch.no_grad():
            t_logit = teacher(xb)
        s_logit = student(xb)
        loss = distill_loss(s_logit, t_logit, yb, T, alpha)
        opt.zero_grad()
        loss.backward()
        opt.step()

# Save final model
torch.save(student.state_dict(), "artifacts/student_model.pt")

# Export to ONNX
dummy_input = torch.randn(1, X_train.shape[1])
torch.onnx.export(student, dummy_input, "artifacts/student_model.onnx", input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}, opset_version=11)

