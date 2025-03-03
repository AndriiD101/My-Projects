import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image

# -------------------------------
# 1. Define the Adjusted CNN Architecture
# -------------------------------
class AdjustedCNN(nn.Module):
    def __init__(self, num_classes):
        super(AdjustedCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1: 224x224 -> 112x112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),     # Conv layer 1
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),     # Conv layer 2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.05),  # Reduced dropout

            # Block 2: 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.05),  # Reduced dropout

            # Block 3: 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),  # Reduced dropout

            # Block 4: 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1)   # Reduced dropout
        )
        # Adaptive pooling to reduce spatial dimensions to 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # Slightly reduced dropout for classifier
            nn.Linear(512, num_classes)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)         # (batch_size, 256, 1, 1)
        x = torch.flatten(x, 1)     # (batch_size, 256)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # Kaiming (He) initialization for convolution and linear layers.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

# -------------------------------
# 2. Training Loop with OneCycleLR and Early Stopping
# -------------------------------
def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=200, early_stopping_patience=15):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0

    # Set up the OneCycleLR scheduler to adjust the learning rate every iteration.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=optimizer.param_groups[0]['lr'],
        steps_per_epoch=len(dataloaders["train"]),
        epochs=num_epochs
    )

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        val_epoch_acc = None

        # Training and Validation phases
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                # Forward pass; track gradients only in train phase.
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()  # Update the learning rate every iteration

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Monitor validation accuracy for early stopping.
            if phase == 'val':
                val_epoch_acc = epoch_acc
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0  # Reset if improvement
                else:
                    epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Val Acc: {best_acc:.4f}")

    # Load best model weights.
    model.load_state_dict(best_model_wts)
    return model

# -------------------------------
# 3. Inference: Predicting Dog Breed Probabilities
# -------------------------------
def predict_dog_breed(model, image_path, transform, class_names, device):
    """
    Processes an image and outputs percentage probabilities for each class.
    """
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
        percentages = probabilities * 100              # Convert to percentages

    percentages = percentages.cpu().numpy().squeeze()
    sorted_indices = percentages.argsort()[::-1]
    print("Prediction probabilities:")
    for idx in sorted_indices:
        print(f"{class_names[idx]}: {percentages[idx]:.2f}%")

# -------------------------------
# 4. Main Function: Data Loading, Training, and Inference
# -------------------------------
def main():
    # Configuration & Hyperparameters
    data_dir = r"C:\Users\denys\Desktop\My-Projects\combined_datasets\combined_datasets"  # Should contain 'train' and 'val' subdirectories.
    num_classes = 120
    batch_size = 32
    num_epochs = 200
    learning_rate = 0.001
    early_stopping_patience = 15

    # Device configuration: use GPU if available.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transformations for training (with augmentation) and validation.
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    # Create datasets for training and validation.
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ["train", "val"]
    }

    # Create dataloaders.
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                        shuffle=True, num_workers=4)
        for x in ["train", "val"]
    }

    # Get the list of class names.
    class_names = image_datasets["train"].classes
    print(f"Detected {len(class_names)} classes.")
    if len(class_names) != num_classes:
        print(f"Warning: Expected {num_classes} classes, but found {len(class_names)} classes.")

    # Initialize the AdjustedCNN model.
    model = AdjustedCNN(num_classes=num_classes).to(device)

    # Define loss function and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model with dynamic learning rate and early stopping.
    model = train_model(model, dataloaders, criterion, optimizer, device,
                        num_epochs=num_epochs, early_stopping_patience=early_stopping_patience)

    # Save the trained model.
    torch.save(model.state_dict(), "cnn_dogbreeed_model.pt") 
    print("Model saved as 'cnn_dogbreeed_model.pt'.")

    # -------------------------------
    # Inference Example:
    # -------------------------------
    test_image_path = r"C:\Users\denys\Desktop\My-Projects\TUKE\Y2S2\CNN from scratch\images.jpg"  
    print("\nPerforming inference on test image:")

    # Use the same transform as validation for inference.
    inference_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    predict_dog_breed(model, test_image_path, inference_transform, class_names, device)

if __name__ == "__main__":
    main()
