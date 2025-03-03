import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import os
import pandas as pd
import CNN  # Import the module where AdjustedCNN is defined
import __main__  # Import the __main__ module

# Register AdjustedCNN in __main__ so that pickle can find it
__main__.AdjustedCNN = CNN.AdjustedCNN

# Load configuration
with open(r'C:\Users\denys\Desktop\My-Projects\Breedify\telegram bot\config.json', 'r') as config_file:
    config=json.load(config_file)

MODEL_PATH = config['MODEL_PATH']
LABEL_PATH = config['LABEL_PATH']

labels_df = pd.read_csv(LABEL_PATH)
dog_breeds = sorted(labels_df['Breed'].str.lower().str.replace(" ", "_"))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_labels(csv_path):
    """Load class labels from a CSV file."""
    df = pd.read_csv(csv_path)
    class_names = df['label'].tolist()  # Adjust the column name if necessary
    return class_names  # Keep the order as in the CSV

# Load class names from CSV
num_classes = len(dog_breeds)

def load_model(model_path):
    """Load the trained model."""
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    return model

# Load the trained model
model = load_model(MODEL_PATH)

# Define the transform used for inference
inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    """
    Takes an image path, processes it, and returns the predicted breed with confidence.
    """
    image = Image.open(image_path).convert("RGB")
    image = inference_transform(image)
    image = image.unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
        confidence, predicted_idx = torch.max(probabilities, 1)  # Get max confidence

    predicted_breed = dog_breeds[predicted_idx.item()]
    
    return {
        "Predicted Breed": predicted_breed,
        "Confidence": confidence.item()
    }

# Example usage:
# result = predict_image("path_to_image.jpg")
# print(result)
