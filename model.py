import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import pandas as pd
import cv2
import os

# Enable memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print("Using GPU:", gpus[0])
    except RuntimeError as e:
        print(e)

# Image data generator
class ImageDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, csv_path, image_dir, batch_size=32, image_size=(512, 512), shuffle=True):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.df.iloc[indices]

        X, y = self.__data_generation(batch_df)
        return X, y

    def __data_generation(self, batch_df):
        X, y = [], []

        for _, row in batch_df.iterrows():
            image_path = os.path.join(self.image_dir, row.file_name)
            label = row.label

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image = cv2.resize(image, self.image_size)
                X.append(image)
                y.append(label)

        X = np.array(X).astype(np.float32) / 255.0
        X = X.reshape(X.shape[0], self.image_size[0], self.image_size[1], 1)
        y = np.array(y).astype(np.float32)

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# Load Data Using Generator
csv_path = '/kaggle/input/ai-vs-human-generated-dataset/train.csv'
image_dir = '/kaggle/input/ai-vs-human-generated-dataset/'
batch_size = 32

train_generator = ImageDataGenerator(csv_path, image_dir, batch_size=batch_size)

# Model Definition
def create_model(input_shape):
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation=None, input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation=None, kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation=None, kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(256, (3, 3), activation=None, kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(256, activation=None, kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Dropout(0.4),

        layers.Dense(128, activation=None, kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Dropout(0.4),

        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    return model

# Initialize Model
model = create_model((512, 512, 1))

# Train Model Using Generator
model.fit(train_generator, epochs=10, verbose=1)

# Save Model
model.save("model_params.h5")

print("\nTraining Completed!")