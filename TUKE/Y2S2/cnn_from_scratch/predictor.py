import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import os
import pandas as pd
import CNN
import __main__ 

__main__.AdjustedCNN = CNN.AdjustedCNN

# Load configuration
with open(r'C:\Users\denys\Desktop\My-Projects\TUKE\Y2S2\cnn_from_scratch\config.json', 'r') as config_file:
    config = json.load(config_file)

MODEL_PATH = config['MODEL_PATH']
LABEL_PATH = config['LABEL_PATH']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

labels_df = pd.read_csv(LABEL_PATH)
dog_breeds = sorted(labels_df['Breed'].str.lower().str.replace(" ", "_"))

def load_model(model_path):
    model = CNN.AdjustedCNN(num_classes=len(dog_breeds)) 
    model.load_state_dict(torch.load(model_path, map_location=device))  
    model.to(device)
    model.eval()
    return model

model = load_model(MODEL_PATH)

inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    """
    Takes an image path, processes it, and returns the predicted breed with confidence.
    """
    image = Image.open(image_path).convert("RGB")
    image = inference_transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1) 
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_breed = dog_breeds[predicted_idx.item()]
    
    return {
        "Predicted Breed": predicted_breed,
        "Confidence": confidence.item()
    }
