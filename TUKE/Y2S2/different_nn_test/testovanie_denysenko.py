import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# -------------------------------
# Global settings and reproducibility
# -------------------------------
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# -------------------------------
# STEP 0: DATA LOADING & PREPROCESSING
# -------------------------------
def load_data(batch_size=64):
    # Make sure "bank-full.csv" is in your working directory.
    df = pd.read_csv("bank-full.csv", sep=';')
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    X = df.drop('y', axis=1)
    y = df['y']
    # One-hot encode categorical variables (drop_first to avoid dummy variable trap)
    X = pd.get_dummies(X, drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y.values, test_size=0.2, random_state=seed, stratify=y
    )
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    input_dim = X_train_tensor.shape[1]
    num_classes = 2  # Binary classification
    return train_loader, test_loader, input_dim, num_classes

train_loader, test_loader, input_dim, num_classes = load_data()

# -------------------------------
# Define Model Architectures
# -------------------------------
# Model1: One hidden layer with 32 neurons
class Model1(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Model1, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Model2: One hidden layer with 64 neurons
class Model2(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Model2, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Model3: Two hidden layers (32 and 16 neurons)
class Model3(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Model3, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Model4: Two hidden layers (64 and 32 neurons)
class Model4(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Model4, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Model5: Three hidden layers (64, 32, 16 neurons)
class Model5(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Model5, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 16)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(16, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

# -------------------------------
# Common training & evaluation functions
# -------------------------------
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        # Uncomment the next line to see the loss per epoch:
        # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader.dataset):.4f}")
    return model

def evaluate_model(model, test_loader, criterion=None):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    avg_loss = (total_loss / total) if criterion is not None else None
    return accuracy, avg_loss

# -------------------------------
# STEP 3: Test 5 Different Network Topologies
# -------------------------------
topology_models = {
    "Model1": Model1(input_dim, num_classes),
    "Model2": Model2(input_dim, num_classes),
    "Model3": Model3(input_dim, num_classes),
    "Model4": Model4(input_dim, num_classes),
    "Model5": Model5(input_dim, num_classes)
}

num_epochs = 20
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()

topology_results = {}
best_topo_accuracy = 0
best_topology_name = None
best_topology_class = None

print("Step 3: Testing 5 Different Topologies")
for name, model in topology_models.items():
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(f"Training {name}...")
    model = train_model(model, train_loader, criterion, optimizer, num_epochs)
    accuracy, _ = evaluate_model(model, test_loader, criterion)
    topology_results[name] = accuracy
    print(f"{name} Accuracy: {accuracy*100:.2f}%")
    if accuracy > best_topo_accuracy:
        best_topo_accuracy = accuracy
        best_topology_name = name
        if name == "Model1":
            best_topology_class = Model1
        elif name == "Model2":
            best_topology_class = Model2
        elif name == "Model3":
            best_topology_class = Model3
        elif name == "Model4":
            best_topology_class = Model4
        elif name == "Model5":
            best_topology_class = Model5

print("\n--- Topology Comparison Results ---")
for name, acc in topology_results.items():
    print(f"{name}: {acc*100:.2f}%")
print(f"\nBest Topology: {best_topology_name} with accuracy {best_topo_accuracy*100:.2f}%\n")

# -------------------------------
# STEP 4: Optimizer Comparison (using best topology)
# -------------------------------
optimizers_to_test = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "RMSprop": optim.RMSprop
}

optimizer_results = {}
best_opt_accuracy = 0
best_optimizer_name = None

print("Step 4: Optimizer Comparison on best topology")
for opt_name, opt_class in optimizers_to_test.items():
    model = best_topology_class(input_dim, num_classes)  # fresh instance
    optimizer = opt_class(model.parameters(), lr=learning_rate)
    print(f"Training best topology with {opt_name} optimizer...")
    model = train_model(model, train_loader, criterion, optimizer, num_epochs)
    accuracy, _ = evaluate_model(model, test_loader, criterion)
    optimizer_results[opt_name] = accuracy
    print(f"{opt_name} Accuracy: {accuracy*100:.2f}%")
    if accuracy > best_opt_accuracy:
        best_opt_accuracy = accuracy
        best_optimizer_name = opt_name
        best_model_after_opt = model  # save the best model state

print("\n--- Optimizer Comparison Results ---")
for name, acc in optimizer_results.items():
    print(f"{name}: {acc*100:.2f}%")
print(f"\nBest Optimizer: {best_optimizer_name} with accuracy {best_opt_accuracy*100:.2f}%\n")

# Save best model from optimizer comparison (for later use)
torch.save(best_model_after_opt.state_dict(), f"best_model_with_{best_optimizer_name}.pth")

# -------------------------------
# STEP 5: Learning Rate Testing (using best topology and best optimizer)
# -------------------------------
learning_rates = [0.0001, 0.001, 0.01, 0.05, 0.1]
lr_results = {}
best_lr_accuracy = 0
best_lr = None

print("Step 5: Learning Rate Testing")
for lr in learning_rates:
    model = best_topology_class(input_dim, num_classes)
    # Using Adam here assuming best optimizer is Adam; adjust if needed.
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(f"Training with learning rate: {lr}")
    model = train_model(model, train_loader, criterion, optimizer, num_epochs)
    accuracy, avg_loss = evaluate_model(model, test_loader, criterion)
    lr_results[lr] = {"accuracy": accuracy, "loss": avg_loss}
    print(f"LR: {lr} | Accuracy: {accuracy*100:.2f}% | Loss: {avg_loss:.4f}")
    if accuracy > best_lr_accuracy:
        best_lr_accuracy = accuracy
        best_lr = lr
        best_model_after_lr = model

# Plot Learning Rate vs Accuracy and Loss
lr_list = list(lr_results.keys())
accuracy_list = [lr_results[lr]["accuracy"]*100 for lr in lr_list]
loss_list = [lr_results[lr]["loss"] for lr in lr_list]

plt.figure(figsize=(8,6))
plt.plot(lr_list, accuracy_list, marker='o')
plt.xscale('log')
plt.xlabel("Learning Rate (log scale)")
plt.ylabel("Accuracy (%)")
plt.title("Learning Rate vs Accuracy")
plt.grid(True)
plt.savefig("learning_rate_vs_accuracy.png")
plt.show()

plt.figure(figsize=(8,6))
plt.plot(lr_list, loss_list, marker='o', color='red')
plt.xscale('log')
plt.xlabel("Learning Rate (log scale)")
plt.ylabel("Loss")
plt.title("Learning Rate vs Loss")
plt.grid(True)
plt.savefig("learning_rate_vs_loss.png")
plt.show()

print(f"\nBest Learning Rate: {best_lr} with Accuracy: {best_lr_accuracy*100:.2f}%\n")

# -------------------------------
# STEP 6: Activation Function Testing (using best topology, best optimizer, and best LR)
# -------------------------------
# Here, we create a subclass of Model1 that allows us to swap the activation function.
# Instead of trying to modify an attribute that doesn't exist, we override the forward method.
class Model1Activation(Model1):
    def __init__(self, input_dim, num_classes, activation):
        super(Model1Activation, self).__init__(input_dim, num_classes)
        self.activation = activation  # store the new activation function

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)  # use the provided activation function
        x = self.fc2(x)
        return x

activations = {
    "ReLU": nn.ReLU(),
    "Sigmoid": nn.Sigmoid(),
    "Tanh": nn.Tanh(),
    "LeakyReLU": nn.LeakyReLU(),
    "ELU": nn.ELU()
}

activation_results = {}
best_act_accuracy = 0
best_activation_name = None

print("Step 6: Activation Function Testing")
for act_name, act_func in activations.items():
    model = Model1Activation(input_dim, num_classes, act_func)
    optimizer = optim.Adam(model.parameters(), lr=best_lr)  # use best LR from previous step
    print(f"Training with activation function: {act_name}")
    model = train_model(model, train_loader, criterion, optimizer, num_epochs)
    accuracy, avg_loss = evaluate_model(model, test_loader, criterion)
    activation_results[act_name] = {"accuracy": accuracy, "loss": avg_loss}
    print(f"{act_name}: Accuracy = {accuracy*100:.2f}%, Loss = {avg_loss:.4f}")
    if accuracy > best_act_accuracy:
        best_act_accuracy = accuracy
        best_activation_name = act_name
        best_model_after_act = model

# Plot Activation Function vs Accuracy and Loss
act_names = list(activation_results.keys())
accuracy_vals = [activation_results[act]["accuracy"]*100 for act in act_names]
loss_vals = [activation_results[act]["loss"] for act in act_names]

plt.figure(figsize=(8,6))
plt.bar(act_names, accuracy_vals, color='skyblue')
plt.xlabel("Activation Function")
plt.ylabel("Accuracy (%)")
plt.title("Activation Function vs Accuracy")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("activation_vs_accuracy.png")
plt.show()

plt.figure(figsize=(8,6))
plt.bar(act_names, loss_vals, color='salmon')
plt.xlabel("Activation Function")
plt.ylabel("Loss")
plt.title("Activation Function vs Loss")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("activation_vs_loss.png")
plt.show()

print("\n--- Activation Function Comparison Results ---")
for act, res in activation_results.items():
    print(f"{act}: Accuracy = {res['accuracy']*100:.2f}%, Loss = {res['loss']:.4f}")
print(f"\nBest Activation Function: {best_activation_name} with Accuracy: {best_act_accuracy*100:.2f}%\n")

# -------------------------------
# STEP 7: FINAL MODEL
# -------------------------------
# Now, using the best configuration from steps 3-6, we retrain the model on the full training set and save it.
print("Step 7: Training Final Model with Best Configuration")
final_model = Model1Activation(input_dim, num_classes, activations[best_activation_name])
final_optimizer = optim.Adam(final_model.parameters(), lr=best_lr)
final_model = train_model(final_model, train_loader, criterion, final_optimizer, num_epochs)
final_accuracy, final_loss = evaluate_model(final_model, test_loader, criterion)
print(f"Final Model Accuracy: {final_accuracy*100:.2f}%, Loss: {final_loss:.4f}")

# Save the final model in two formats
final_model_path = "final_best_model.pth"
torch.save(final_model.state_dict(), final_model_path)
print(f"Final model saved as {final_model_path}")

final_model_path = "final_best_model.pt"
torch.save(final_model.state_dict(), final_model_path)
print(f"Final model saved as {final_model_path}")
