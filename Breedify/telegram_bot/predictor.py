import os
import numpy as np
import json
import pandas as pd
from keras.models import load_model
from keras.utils import load_img, img_to_array
from keras.applications.inception_v3 import preprocess_input
from keras.metrics import Metric
from keras import backend as K

# Define the custom metric (F1 Score)
class F1Score(Metric):
    def __init__(self, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = K.round(y_pred)
        y_true = K.cast(y_true, "float32")
        y_pred = K.cast(y_pred, "float32")
        
        tp = K.sum(y_true * y_pred)
        fp = K.sum((1 - y_true) * y_pred)
        fn = K.sum(y_true * (1 - y_pred))
        
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + K.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + K.epsilon())
        return 2 * (precision * recall) / (precision + recall + K.epsilon())

    def reset_states(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)

custom_objects = {"F1Score": F1Score}

with open(r'C:\Users\denys\Desktop\My-Projects\Breedify\telegram_bot\config.json', 'r') as config_file:
    config=json.load(config_file)

MODEL_PATH = config['MODEL_PATH']
LABEL_PATH = config['LABEL_PATH']

labels_df = pd.read_csv(LABEL_PATH)
dog_breeds = sorted(labels_df['Breed'].str.lower().str.replace(" ", "_"))
    
def predict_image(image_path, confidence_threshold=0.7):
    """
    Predict the breed of a single dog image using a trained model.

    Args:
        image_path (str): Path to the image file.
        model_path (str): Path to the trained model file.
        label_path (str): Path to the CSV file containing dog breed labels.
        confidence_threshold (float): Threshold to determine if prediction is confident.

    Returns:
        dict: Prediction details (breed, confidence, certainty).
    """
    try:
        model = load_model(MODEL_PATH, custom_objects=custom_objects)

        img = load_img(image_path, target_size=(512, 512))  
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  
        img_array = preprocess_input(img_array)  

        # Predict the dog breed
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)  
        confidence = predictions[0][predicted_class]  
        predicted_breed = dog_breeds[predicted_class]

        certainty = "Certain" if confidence >= confidence_threshold else "Uncertain"

        return {
            "Predicted Breed": predicted_breed,
            "Confidence": confidence,
            "Certainty": certainty
        }

    except Exception as e:
        return {"Error": str(e)}

# Example usage
# model_path = r"C:\Users\denys\Desktop\UI\telegram bot\best_model_unfreeze30_test.h5"
# image_path = r"C:\Users\denys\Desktop\UI\1Dog-rough-collie-portrait.jpg"
# label_path = r"C:\Users\denys\Desktop\UI\test_images"

# result = predict_single_image(image_path, model_path, label_path)
# print(result)
