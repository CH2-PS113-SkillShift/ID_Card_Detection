import os
import cv2
import random
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


# Load the trained model
model = load_model("image_classification_model.h5")
annotations_path = "generated_dataset/annotations.csv"
df_annotations = pd.read_csv(annotations_path)


# Function to preprocess the input image for inference
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (40, 40))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# Function to predict the label of an image
def predict_image_label(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)
    confidence = prediction[0][predicted_label]
    return predicted_label, confidence


# Map predicted label to characters
characters = "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
label_to_char = {i: char for i, char in enumerate(characters)}

# Predict labels for 10 random images
random_indices = random.sample(range(len(df_annotations)), 10)
for idx in random_indices:
    image_name = df_annotations.iloc[idx]["Image_Name"]
    image_path = os.path.join("generated_dataset", image_name)

    true_label = df_annotations.iloc[idx]["Label"]

    predicted_label, confidence = predict_image_label(image_path)
    predicted_character = label_to_char[predicted_label]

    print(f"Image: {image_path}")
    print(f"Ground Truth Label: {true_label}")
    print(f"Predicted Label: {predicted_character}, Confidence: {confidence:.2f}\n")
