from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Define dataset path
dataset_path = "dataset/fresh_rotten_veg_fruits"

# Image Data Generator (Augmentation)
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=30,   # Rotate images
    width_shift_range=0.2,  # Shift images horizontally
    height_shift_range=0.2, # Shift images vertically
    shear_range=0.2,  # Shear transformation
    zoom_range=0.2,   # Zoom in images
    horizontal_flip=True,  # Flip images
    validation_split=0.2
)


# Load Train & Validation Data
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# Build CNN Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze the base model
base_model.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dense(len(train_data.class_indices), activation='softmax')(x)  # Match number of classes

# Create Model
model = Model(inputs=base_model.input, outputs=x)

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_data, validation_data=val_data, epochs=10)

# Save model
model.save("models/fruit_spoil_detector.h5")