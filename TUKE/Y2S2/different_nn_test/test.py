import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# Reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Load models
from models.model1 import Model1
from models.model2 import Model2
from models.model3 import Model3
from models.model4 import Model4
from models.model5 import Model5

# Load dataset
df = pd.read_csv("bank-full.csv", sep=';')

# Preprocessing
X_raw = df.drop(columns=["y"])
y_raw = df["y"].map({"no": 0, "yes": 1}).values

categorical_cols = X_raw.select_dtypes(include="object").columns.tolist()
numeric_cols = X_raw.select_dtypes(exclude="object").columns.tolist()

encoder = OneHotEncoder(sparse_output=False)
X_cat = encoder.fit_transform(X_raw[categorical_cols])

scaler = StandardScaler()
X_num = scaler.fit_transform(X_raw[numeric_cols])

X_full = np.hstack([X_num, X_cat])

X_train, X_test, y_train, y_test = train_test_split(X_full, y_raw, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

y_train_oh = torch.nn.functional.one_hot(y_train, num_classes=2).float()
y_test_oh = torch.nn.functional.one_hot(y_test, num_classes=2).float()

BATCH_SIZE = 64
train_ds = TensorDataset(X_train, y_train_oh)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

criterion = nn.CrossEntropyLoss()

# 1. Nude training (manual gradient update)
def train_model(model, model_name):
    model.train()
    best_acc = 0.0
    for epoch in range(1, 11):
        total_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            out = model(xb)
            loss = criterion(out, yb)

            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param -= param.grad

            total_loss += loss.item() * xb.size(0)
            preds = torch.argmax(out, dim=1)
            true_labels = torch.argmax(yb, dim=1)
            correct += (preds == true_labels).sum().item()
            total += xb.size(0)

        acc = correct / total
        avg_loss = total_loss / total
        print(f"{model_name} | Epoch {epoch}: accuracy: {acc:.4f}, loss: {avg_loss:.4f}")
        if acc > best_acc:
            best_acc = acc
    return best_acc

INPUT_SIZE = X_train.shape[1]
OUTPUT_SIZE = 2

models = [
    (Model1(INPUT_SIZE, OUTPUT_SIZE), "Model1"),
    (Model2(INPUT_SIZE, OUTPUT_SIZE), "Model2"),
    (Model3(INPUT_SIZE, OUTPUT_SIZE), "Model3"),
    (Model4(INPUT_SIZE, OUTPUT_SIZE), "Model4"),
    (Model5(INPUT_SIZE, OUTPUT_SIZE), "Model5"),
]

best_model = None
best_name = None
best_accuracy = 0.0

for model, name in models:
    print(f"\nTraining {name}")
    acc = train_model(model, name)
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_name = name

print(f"\nBest model: {best_name} with training accuracy: {best_accuracy:.4f}")

# 2. Optimizer training
optimizers = {
    "SGD": lambda m: torch.optim.SGD(m.parameters(), lr=0.01),
    "Adam": lambda m: torch.optim.Adam(m.parameters(), lr=0.001),
    "RMSprop": lambda m: torch.optim.RMSprop(m.parameters(), lr=0.001),
    "AdamW": lambda m: torch.optim.AdamW(m.parameters(), lr=0.001)
}

train_ds_ce = TensorDataset(X_train, y_train)
train_loader_ce = DataLoader(train_ds_ce, batch_size=BATCH_SIZE, shuffle=True)

print(f"\nTraining best model {best_name} with optimizers")

best_optimizer_name = None
best_optimizer_acc = 0.0

for opt_name, opt_fn in optimizers.items():
    model = eval(best_name)(INPUT_SIZE, OUTPUT_SIZE)
    optimizer = opt_fn(model)
    print(f"\n{opt_name} optimizer")
    for epoch in range(1, 11):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_loader_ce:
            out = model(xb)
            loss = criterion(out, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            preds = torch.argmax(out, dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        acc = correct / total
        avg_loss = total_loss / total
        print(f"{opt_name} | Epoch {epoch}: accuracy: {acc:.4f}, loss: {avg_loss:.4f}")
    if acc > best_optimizer_acc:
        best_optimizer_acc = acc
        best_optimizer_name = opt_name

print(f"\nBest optimizer: {best_optimizer_name} with accuracy: {best_optimizer_acc:.4f}")

# 3. Learning Rate Testing
learning_rates = [0.0001, 0.001, 0.01, 0.05, 0.1]
lr_results = []
lr_losses = []

print(f"\nLearning rate testing with {best_name} + {best_optimizer_name}")

for lr in learning_rates:
    model = eval(best_name)(INPUT_SIZE, OUTPUT_SIZE)
    optimizer_cls = getattr(torch.optim, best_optimizer_name)
    optimizer = optimizer_cls(model.parameters(), lr=lr)

    for epoch in range(1, 11):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_loader_ce:
            out = model(xb)
            loss = criterion(out, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            preds = torch.argmax(out, dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        acc = correct / total
        avg_loss = total_loss / total
        print(f"LR={lr} | Epoch {epoch}: accuracy: {acc:.4f}, loss: {avg_loss:.4f}")

    lr_results.append((lr, acc))
    lr_losses.append((lr, avg_loss))

# Save best learning rate
best_lr = sorted(lr_results, key=lambda x: x[1], reverse=True)[0][0]

# 4. Plot accuracy vs learning rate
lrs, accs = zip(*lr_results)
plt.figure(figsize=(8, 5))
plt.plot(lrs, accs, marker='o')
plt.xscale('log')
plt.xlabel('Learning Rate (log scale)')
plt.ylabel('Accuracy')
plt.title(f'{best_name} + {best_optimizer_name}: Learning Rate vs Accuracy')
plt.grid(True)
plt.tight_layout()
plt.savefig('lr_vs_accuracy.png')
plt.show()

# 5. Plot loss vs learning rate
lrs_l, losses = zip(*lr_losses)
plt.figure(figsize=(8, 5))
plt.plot(lrs_l, losses, marker='o', color='red')
plt.xscale('log')
plt.xlabel('Learning Rate (log scale)')
plt.ylabel('Loss')
plt.title(f'{best_name} + {best_optimizer_name}: Learning Rate vs Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig('lr_vs_loss.png')
plt.show()

# 6. Activation Functions Testing
activation_functions = {
    "ReLU": torch.nn.ReLU(),
    "Sigmoid": torch.nn.Sigmoid(),
    "Tanh": torch.nn.Tanh(),
    "LeakyReLU": torch.nn.LeakyReLU(),
    "ELU": torch.nn.ELU(),
}

activation_results = []
print(f"\nTesting activation functions on {best_name} with {best_optimizer_name} and best LR")

for act_name, act_fn in activation_functions.items():
    class ModifiedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            if best_name == "Model1":
                self.model = nn.Sequential(
                    nn.Linear(INPUT_SIZE, 16),
                    act_fn,
                    nn.Linear(16, OUTPUT_SIZE)
                )
            elif best_name == "Model2":
                self.model = nn.Sequential(
                    nn.Linear(INPUT_SIZE, 32),
                    act_fn,
                    nn.Linear(32, 16),
                    act_fn,
                    nn.Linear(16, OUTPUT_SIZE)
                )
            elif best_name == "Model3":
                self.model = nn.Sequential(
                    nn.Linear(INPUT_SIZE, 64),
                    act_fn,
                    nn.Linear(64, 32),
                    act_fn,
                    nn.Linear(32, 16),
                    act_fn,
                    nn.Linear(16, OUTPUT_SIZE)
                )
            elif best_name == "Model4":
                self.model = nn.Sequential(
                    nn.Linear(INPUT_SIZE, 64),
                    act_fn,
                    nn.Dropout(0.3),
                    nn.Linear(64, 32),
                    act_fn,
                    nn.Dropout(0.3),
                    nn.Linear(32, OUTPUT_SIZE)
                )
            elif best_name == "Model5":
                self.model = nn.Sequential(
                    nn.Linear(INPUT_SIZE, 64),
                    nn.BatchNorm1d(64),
                    act_fn,
                    nn.Linear(64, 32),
                    nn.BatchNorm1d(32),
                    act_fn,
                    nn.Linear(32, OUTPUT_SIZE)
                )

        def forward(self, x):
            return self.model(x)

    model = ModifiedModel()
    optimizer_cls = getattr(torch.optim, best_optimizer_name)
    optimizer = optimizer_cls(model.parameters(), lr=best_lr)

    for epoch in range(1, 11):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_loader_ce:
            out = model(xb)
            loss = criterion(out, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            preds = torch.argmax(out, dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        acc = correct / total
        avg_loss = total_loss / total
        print(f"{act_name} | Epoch {epoch}: accuracy: {acc:.4f}, loss: {avg_loss:.4f}")

    activation_results.append((act_name, acc))

# Plot activation vs accuracy
acts, accs = zip(*activation_results)
plt.figure(figsize=(8, 5))
plt.bar(acts, accs)
plt.xlabel("Activation Function")
plt.ylabel("Accuracy")
plt.title(f'{best_name} + {best_optimizer_name} @ LR={best_lr}: Activation Function vs Accuracy')
plt.grid(True)
plt.tight_layout()
plt.savefig('activation_vs_accuracy.png')
plt.show()

# 7. Save Final Model
final_model = eval(best_name)(INPUT_SIZE, OUTPUT_SIZE)
final_optimizer_cls = getattr(torch.optim, best_optimizer_name)
final_optimizer = final_optimizer_cls(final_model.parameters(), lr=best_lr)

# Final train loop
for epoch in range(1, 11):
    final_model.train()
    for xb, yb in train_loader_ce:
        out = final_model(xb)
        loss = criterion(out, yb)

        final_optimizer.zero_grad()
        loss.backward()
        final_optimizer.step()

# Save
torch.save(final_model.state_dict(), "final_model.pth")
print("\nFinal model saved to final_model.pth")
