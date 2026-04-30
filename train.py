import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout

# Settings
FRAME_SIZE = 64
MAX_FRAMES = 20

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
        frame = frame / 255.0
        frames.append(frame)

    cap.release()

    # ❗ FIX: handle empty video
    if len(frames) == 0:
        return None

    # padding
    while len(frames) < MAX_FRAMES:
        frames.append(frames[-1])

    return np.array(frames)


def load_dataset(base_path):
    X = []
    y = []

    for label, category in enumerate(['normal', 'cp']):
        path = os.path.join(base_path, category)

        for root, dirs, files in os.walk(path):
            for file in files:

                if not file.endswith(('.mp4', '.avi', '.mov')):
                    continue

                video_path = os.path.join(root, file)

                data = process_video(video_path)

                if data is None:
                    continue

                data = data.reshape(MAX_FRAMES, -1)

                X.append(data)
                y.append(label)

    return np.array(X), np.array(y)

# Load data
X, y = load_dataset("dataset")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Dataset loaded:", X.shape)

# Model
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(20, 64*64*3)),
    Dropout(0.3),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 🔥 ADD THIS (just before training)
from sklearn.utils import class_weight

weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y),
    y=y
)

class_weights = {0: weights[0], 1: weights[1]}

print("Class weights:", class_weights)

# 🔥 MODIFY TRAINING
model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=4,
    validation_data=(X_test, y_test),
    class_weight=class_weights
)

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

# Save
model.save("model.h5")

print("Model saved as model.h5")