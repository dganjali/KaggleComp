import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import pandas as pd
import cv2
import os

def load_data(csv_path, image_dir, target_size=(500, 500), is_test=False):
    print(f"Loading data from {csv_path}")
    print(f"Images will be loaded from {image_dir}")
    
    df = pd.read_csv(csv_path)
    print(f"Successfully loaded CSV with {len(df)} rows")
    
    X = []
    y = [] if not is_test else None
    
    # Get image sizes first
    image_sizes = []
    for i, row in enumerate(df.itertuples()):
        filename = row.id if is_test else row.file_name
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            image_sizes.append(image.shape)
    
    print(f"Image size distribution:")
    print(f"Min height: {min(h for h,w in image_sizes)}, Max height: {max(h for h,w in image_sizes)}")
    print(f"Min width: {min(w for h,w in image_sizes)}, Max width: {max(w for h,w in image_sizes)}")
    
    # Process images
    for i, row in enumerate(df.itertuples()):
        if i % 100 == 0:
            print(f"Processing image {i}/{len(df)}")
            
        filename = row.id if is_test else row.file_name
        if not is_test:
            label = row.label
            
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            # Preserve aspect ratio while resizing
            h, w = image.shape
            ratio = min(target_size[0]/h, target_size[1]/w)
            new_size = (int(w*ratio), int(h*ratio))
            
            # Resize maintaining aspect ratio
            image = cv2.resize(image, new_size)
            
            # Create blank canvas of target size
            final_image = np.zeros(target_size, dtype=np.uint8)
            
            # Center the image on canvas
            y_offset = (target_size[0] - new_size[1]) // 2
            x_offset = (target_size[1] - new_size[0]) // 2
            final_image[y_offset:y_offset+new_size[1], x_offset:x_offset+new_size[0]] = image
            
            # Add preprocessing
            final_image = cv2.equalizeHist(final_image)  # Enhance contrast
            
            X.append(final_image)
            if not is_test:
                y.append(label)
    
    X = np.array(X).astype(np.float32) / 255.0
    X = X.reshape(X.shape[0], target_size[0], target_size[1], 1)
    
    if not is_test:
        y = np.array(y).astype(np.float32)
        print(f"Final dataset shape: X={X.shape}, y={y.shape}")
        return X, y
    else:
        print(f"Final test data shape: X={X.shape}")
        return X, None

print("Loading training data...")
csv_path = 'archive/train.csv'
image_dir = 'archive/'
X_train, y_train = load_data(csv_path, image_dir)
print("Training data loaded successfully")

print("Loading test data...")
csv_path = 'archive/test.csv'
X_test, _ = load_data(csv_path, image_dir, is_test=True)
print("Test data loaded successfully")

def create_model(input_shape):
    print(f"Creating model with input shape: {input_shape}")
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
    print("Model created and compiled successfully")
    model.summary()  # Print model architecture
    return model

print("\nInitializing model...")
model = create_model((64, 64, 1))

print("\nStarting model training...")
history = model.fit(X_train, y_train, epochs=20, batch_size=64, 
                    validation_split=0.2, verbose=1)
print("Model training completed")

print("\nSaving model...")
model.save("model_params.h5")
print("Model saved successfully")

print("\nEvaluating model on training data...")
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=1)
print(f'Training Accuracy: {train_accuracy:.4f}')

print("\nEvaluating model on test data...")
test_loss, test_accuracy = model.evaluate(X_test, verbose=1)
print(f'Test Accuracy: {test_accuracy:.4f}')
