import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import pandas as pd
import cv2
import os

def load_data(csv_path, image_dir, image_size=(64, 64)):
    df = pd.read_csv(csv_path)
    X, y = [], []
    for _, row in df.iterrows():
        filename = row["file_name"]  # use the 'file_name' column
        label = row["label"]         # use the 'label' column
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            image = cv2.resize(image, image_size)
            X.append(image)
            y.append(label)
    X = np.array(X).astype(np.float32) / 255.0
    X = X.reshape(X.shape[0], image_size[0], image_size[1], 1)
    y = np.array(y).astype(np.float32)
    return X, y

csv_path = 'archive/train.csv'
image_dir = 'archive/'
X_train, y_train = load_data(csv_path, image_dir)
X_test, y_test = load_data(csv_path, image_dir)

def create_model(input_shape):
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

model = create_model((64, 64, 1))
history = model.fit(X_train, y_train, epochs=20, batch_size=64, 
                    validation_data=(X_test, y_test), verbose=1)

model.save("model_params.h5")

train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
print(f'Training Accuracy: {train_accuracy:.4f}')

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Accuracy: {test_accuracy:.4f}')
