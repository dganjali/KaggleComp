import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import cv2
import os

model = keras.models.load_model('model_params.h5')

image_size = (256, 256)

def load_test_data(csv_path, image_dir, image_size=(256, 256)):
    print(f"Loading test data from {csv_path}")
    print(f"Images will be loaded from {image_dir}")

    df = pd.read_csv(csv_path)
    print(f"Successfully loaded CSV with {len(df)} rows")

    X_test = []
    ids = []

    for i, row in enumerate(df.itertuples()):
        if i % 100 == 0:
            print(f"Processing image {i}/{len(df)}")

        filename = str(row.id)
        ids.append(row.id)

        image_path = os.path.join(image_dir, filename)
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image = cv2.resize(image, image_size)
                X_test.append(image)
            else:
                print(f"Error: Could not read image at {image_path}")
                X_test.append(np.zeros(image_size, dtype=np.uint8))
                # ids.pop()
                continue

        except Exception as e:
            print(f"Error processing image {filename}: {e}")
            X_test.append(np.zeros(image_size, dtype=np.uint8))
            # ids.pop()
            continue

    X_test = np.array(X_test).astype(np.float32) / 255.0
    X_test = X_test.reshape(X_test.shape[0], image_size[0], image_size[1], 1)

    return X_test, ids

csv_path = 'archive/test.csv'
image_dir = 'archive/'
X_test, ids = load_test_data(csv_path, image_dir, image_size)

predictions = model.predict(X_test)
predictions = (predictions > 0.5).astype(int)

submission = pd.DataFrame({'id': ids, 'label': predictions.flatten()})

submission_file = 'submission.csv'
submission.to_csv(submission_file, index=False)

print(f"Submission file saved to {submission_file}")