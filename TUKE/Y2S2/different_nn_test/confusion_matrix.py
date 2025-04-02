import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from models.model3 import Model3  # Ensure model3.py is in your working directory

# --- Load and Preprocess Data ---
df = pd.read_csv("bank-full.csv", sep=";")
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# Features and target
X = df.drop("y", axis=1)
y = df["y"]

# One-hot encode categorical features WITHOUT drop_first
X = pd.get_dummies(X, drop_first=False)

# Ensure all data is numeric and no NaNs remain
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)

print(f"Number of features after encoding: {X.shape[1]}")  # should be 51

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert test set to tensor
X_test = X_test.astype('float32')
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

# --- Load the Trained Model ---
input_dim = X.shape[1]  # 51 features now
num_classes = 2
model = Model3(input_dim, num_classes)

try:
    model.load_state_dict(torch.load("final_model.pth", map_location=torch.device('cpu')))
except RuntimeError as e:
    print("Model structure does not match the loaded weights!")
    print(e)
    exit()

model.eval()

# --- Model Inference ---
with torch.no_grad():
    outputs = model(X_test_tensor)

_, predicted = torch.max(outputs, 1)
y_pred = predicted.numpy()
y_test_np = y_test.values

# --- Confusion Matrix ---
cm = confusion_matrix(y_test_np, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")
plt.show()
