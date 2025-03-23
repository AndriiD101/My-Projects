import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from models.model1 import Model1  # Ensure model1.py is in your working directory

# --- Load and Preprocess Data ---
# The bank-full.csv file from the Bank Marketing dataset is usually semicolon-separated
df = pd.read_csv("bank-full.csv", sep=";")

# Convert target variable 'y' to binary: mapping "yes" to 1 and "no" to 0
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# Separate features and target
X = df.drop("y", axis=1)
y = df["y"]

# Convert categorical features into numerical variables using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Split the dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the test set to float32 to ensure compatibility with PyTorch
X_test = X_test.astype('float32')
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

# --- Load the Trained Model ---
input_dim = X.shape[1]
num_classes = 2
model = Model1(input_dim, num_classes)
model.load_state_dict(torch.load("final_best_model.pth", map_location=torch.device('cuda')))
model.eval()

# --- Model Inference ---
with torch.no_grad():
    outputs = model(X_test_tensor)

_, predicted = torch.max(outputs, 1)
y_pred = predicted.numpy()
y_test_np = y_test.values

# --- Create and Plot the Confusion Matrix ---
cm = confusion_matrix(y_test_np, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")

# Save the plot as an image file
plt.savefig("confusion_matrix.png")
plt.show()