import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

# Define a simple CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers: Input channel=1 (grayscale), output channels=32 then 64
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Max pooling layer to reduce spatial dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layers:
        # After two poolings, 28x28 images become 7x7 feature maps.
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 output classes for FashionMNIST
        self.relu = nn.ReLU()

    def forward(self, x):
        # Conv1 block: Convolution -> ReLU -> Pooling
        x = self.pool(self.relu(self.conv1(x)))
        # Conv2 block: Convolution -> ReLU -> Pooling
        x = self.pool(self.relu(self.conv2(x)))
        # Flatten
        x = x.view(x.size(0), -1)
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 5

# Data transformation: convert images to tensors and normalize.
# For FashionMNIST, a common normalization uses mean=0.5 and std=0.5.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download the FashionMNIST training and test datasets
full_train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

# Split the training dataset: 50,000 for training and 10,000 for validation.
train_size = 50000
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Create DataLoaders for training, validation, and test sets.
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Set up the device (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ------------------------------
# Training loop with validation
# ------------------------------
print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    avg_train_loss = running_loss / len(train_loader)
    
    # Evaluate on the validation set
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

# Final evaluation on the test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy of the model on the 10,000 test images: {test_accuracy:.2f}%")

# --------------------------------------------------
# Visualization: Capture and display feature maps
# --------------------------------------------------

# Dictionary to store activations
activations = {}

# Define hook functions to capture the outputs after each convolution block
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Register hooks on the convolutional layers
hook1 = model.conv1.register_forward_hook(get_activation('conv1'))
hook2 = model.conv2.register_forward_hook(get_activation('conv2'))

# Choose a sample from the test dataset (e.g., the first image)
sample_data, sample_target = test_dataset[0]
sample_data = sample_data.unsqueeze(0).to(device)  # add batch dimension

# Forward pass to capture activations
_ = model(sample_data)

# Function to visualize feature maps
def visualize_feature_maps(activation, title):
    # Remove the batch dimension: shape becomes (channels, height, width)
    activation = activation.squeeze(0)
    n_channels = activation.shape[0]
    grid_size = int(np.ceil(np.sqrt(n_channels)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    for i in range(n_channels):
        # Convert tensor to numpy array and display the feature map
        axes[i].imshow(activation[i].cpu().numpy(), cmap='viridis')
        axes[i].axis('off')
    # Turn off any remaining axes in the grid
    for i in range(n_channels, len(axes)):
        axes[i].axis('off')
    plt.suptitle(title)
    plt.show()

# Visualize the feature maps after the first convolution block (conv1 + pooling)
visualize_feature_maps(activations['conv1'], 'Feature Maps after Conv1 (and pooling)')

# Visualize the feature maps after the second convolution block (conv2 + pooling)
visualize_feature_maps(activations['conv2'], 'Feature Maps after Conv2 (and pooling)')

# Optionally, remove hooks if no longer needed
hook1.remove()
hook2.remove()
