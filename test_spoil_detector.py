import tensorflow as tf
import numpy as np
import cv2
import os

# Load the trained model
model = tf.keras.models.load_model("models/fruit_spoil_detector.h5")

# Define class labels (same as used in training)
class_labels =["freshapples","freshbanana","freshcucumber","freshokra", "freshoranges","freshpotato","freshtomato","rottenapples","rottenbanana","rottencucumber","rottenokra","rottenoranges","rottenpotato","rottentomato"]

# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)  # Load image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (128, 128))  # Resize to match model input
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

print("Class Labels:", class_labels)

# Function to predict freshness/spoilage
def predict_fruit_spoilage(image_path):
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction[0])  # Get highest probability index
    print(f"Predicted class index: {predicted_class}, Total classes: {len(class_labels)}")
    return class_labels[predicted_class]


# ** Test the model **
if __name__ == "__main__":
    image_path = r"C:\SavEat\dataset\test_images\sample.png"  # Change to your test image path
    result = predict_fruit_spoilage(image_path)
    print(f"Prediction: {result}")
