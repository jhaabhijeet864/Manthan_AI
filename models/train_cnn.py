import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Paths
BASE_DIR = Path("models/data/raw_data/images")
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"

# Parameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

# Data generators with augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)
val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Simple CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(*IMG_SIZE, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen
)

# Save model in both HDF5 and Keras formats
model.save("models/trained_models/fish_cnn.h5")
model.save("models/trained_models/fish_cnn.keras")
print("✅ Model saved to models/trained_models/fish_cnn.h5 and models/trained_models/fish_cnn.keras")

# Optional: Plot training history
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training History')
plt.savefig("models/trained_models/cnn_training_history.png")
plt.show()

# Command to run the training script
os.system("python models/train_cnn.py")