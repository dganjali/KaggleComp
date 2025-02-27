import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import regularizers
import numpy as np
import pandas as pd
import cv2
import os

# temp code to make sure PC uses gpu
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print("Using GPU:", gpus[0])
    except RuntimeError as e:
        print(e)

def load_data(csv_path, image_dir, image_size=(256, 256), is_test=False):
    print(f"Loading data from {csv_path}")
    print(f"Images will be loaded from {image_dir}")

    df = pd.read_csv(csv_path)
    print(f"Successfully loaded CSV with {len(df)} rows")

    X = []
    y = [] if not is_test else None

    for i, row in enumerate(df.itertuples()):
        if i % 100 == 0:  # Print progress every 100 images
            print(f"Processing image {i}/{len(df)}")

        # For test data, use id as filename
        filename = row.id if is_test else row.file_name
        if not is_test:
            label = row.label

        image_path = os.path.join(image_dir, filename)
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image = cv2.resize(image, image_size)
                X.append(image)
                if not is_test:
                    y.append(label)
            else:
                print(f"Error: Could not read image at {image_path}")
        except Exception as e:
            print(f"Error processing image {filename}: {e}")
            continue

    X = np.array(X).astype(np.float32) / 255.0
    print(f"Shape of X before reshaping: {X.shape}")
    X = X.reshape(X.shape[0], image_size[0], image_size[1], 1)

    if not is_test:
        y = np.array(y).astype(np.float32)
        print(f"Final dataset shape: X={X.shape}, y={y.shape}")
        return X, y
    else:
        print(f"Final test data shape: X={X.shape}")
        return X, None

image_dir = 'archive/'

print("Loading training data...")
csv_path = 'archive/train.csv'
X_train, y_train = load_data(csv_path, image_dir)
print("Training data loaded successfully")

print("Loading test data...")
csv_path = 'archive/test.csv'
X_test, _ = load_data(csv_path, image_dir, is_test=True)
print("Test data loaded successfully")

def create_model(input_shape):
    print(f"Creating model with input shape: {input_shape}")
    model = keras.Sequential([
        layers.Conv2D(128, (3, 3), activation='relu', input_shape=input_shape,
        kernel_regularizer=regularizers.l2(0.001)),
        #layers.Dropout(0.1),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation='relu', 
        kernel_regularizer=regularizers.l2(0.001)),
        #layers.Dropout(0.1),
        layers.BatchNormalization(),
        
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu', 
        kernel_regularizer=regularizers.l2(0.001)),
        #layers.Dropout(0.1),
        layers.BatchNormalization(),

        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    print("Model created and compiled successfully")
    model.summary()  # Print model architecture
    return model

def submission(model, X_test, output_path='submission.csv'):

    print("Making predictions...")
    predictions = model.predict(X_test)
    predictions = (predictions >= 0.5).astype(int)

    submission_df = pd.DataFrame({
        'id': range(1, len(predictions) + 1),
        'label': predictions.flatten()
    })

    submission_df.to_csv(output_path, index=False)
    print(f"Submission file saved at {output_path}")


print("\nInitializing model...")
model = create_model((256, 256, 1))

print("\nStarting model training...")
history = model.fit(X_train, y_train, epochs=5, batch_size=64, 
                    validation_split=0.2, verbose=1)
print("Model training completed")

print("\nSaving model...")
model.save("model_params.h5")
print("Model saved successfully")

'''
print("Loading model...")
model = keras.models.load_model("model_params.h5")
print("Model loaded")
'''

print("\nEvaluating model on training data...")
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=1)
print(f'Training Accuracy: {train_accuracy:.4f}')

print("\nEvaluating model on test data...")
test_loss, test_accuracy = model.evaluate(X_test, verbose=1)
print(f'Test Accuracy: {test_accuracy:.4f}')

#submission(model, X_test)