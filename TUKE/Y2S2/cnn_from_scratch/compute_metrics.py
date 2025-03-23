import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import os
import pandas as pd
import CNN 
import __main__     

__main__.AdjustedCNN = CNN.AdjustedCNN

with open(r'C:\Users\denys\Desktop\My-Projects\TUKE\Y2S2\CNN from scratch\config.json', 'r') as config_file:
    config = json.load(config_file)

MODEL_PATH = config['MODEL_PATH']
LABEL_PATH = config['LABEL_PATH']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

labels_df = pd.read_csv(LABEL_PATH)
dog_breeds = sorted(labels_df['Breed'].str.lower().str.replace(" ", "_"))

def load_model(model_path):
    model = CNN.AdjustedCNN(num_classes=len(dog_breeds)) 
    model.load_state_dict(torch.load(model_path, map_location=device, ))
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
    Given an image path, this function processes the image and returns the predicted breed along with its confidence.
    """
    image = Image.open(image_path).convert("RGB")
    image = inference_transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_breed = dog_breeds[predicted_idx.item()]
    return predicted_breed, confidence.item()

def compute_metrics(true_labels, predicted_labels, classes):
    """
    Manually compute accuracy, precision, recall and F1 score.
    
    Parameters:
        true_labels (list): Ground truth labels.
        predicted_labels (list): Labels predicted by the model.
        classes (list): List of all class names.
        
    Returns:
        A tuple with (accuracy, macro_precision, macro_recall, macro_f1)
    """
    class_metrics = {cls: {"TP": 0, "FP": 0, "FN": 0} for cls in classes}
    
    correct = 0
    total = len(true_labels)
    
    for true, pred in zip(true_labels, predicted_labels):
        if true == pred:
            correct += 1
        for cls in classes:
            if pred == cls:
                if true == cls:
                    class_metrics[cls]["TP"] += 1
                else:
                    class_metrics[cls]["FP"] += 1
            elif true == cls:
                class_metrics[cls]["FN"] += 1

    accuracy = correct / total if total > 0 else 0.0

    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0

    for cls in classes:
        TP = class_metrics[cls]["TP"]
        FP = class_metrics[cls]["FP"]
        FN = class_metrics[cls]["FN"]
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
        
        print(f"Class: {cls}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    num_classes = len(classes)
    macro_precision = precision_sum / num_classes
    macro_recall = recall_sum / num_classes
    macro_f1 = f1_sum / num_classes
    
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    
    return accuracy, macro_precision, macro_recall, macro_f1

VAL_PATH = r'C:\Users\denys\Desktop\My-Projects\combined_datasets\combined_datasets\val'

true_labels = []
predicted_labels = []

for breed_folder in os.listdir(VAL_PATH):
    breed_path = os.path.join(VAL_PATH, breed_folder)
    if os.path.isdir(breed_path):
        true_breed = breed_folder.lower().replace(" ", "_")
        for file in os.listdir(breed_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_file_path = os.path.join(breed_path, file)
                predicted_breed, confidence = predict_image(image_file_path)
                true_labels.append(true_breed)
                predicted_labels.append(predicted_breed)
                print(f"Image: {file}, True: {true_breed}, Predicted: {predicted_breed}, Confidence: {confidence:.4f}")

compute_metrics(true_labels, predicted_labels, dog_breeds)

