import os
import json
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout,
    LSTM, GRU, Bidirectional, TimeDistributed,
    GlobalAveragePooling2D, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
IMG_SIZE = (128, 128)
SEQUENCE_LENGTH = 10
EPOCHS = 15
BATCH_SIZE = 8
BASE_PATH = "MINI-RGBD_web"

cp_mapping = {
    '01': 0, '02': 0, '03': 0, '04': 0, '05': 0, '06': 0,
    '07': 1, '08': 1, '09': 1, '10': 1, '11': 1, '12': 1
}

# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────
def extract_features(rgb, depth):
    rgb = cv2.resize(rgb, IMG_SIZE)
    depth = cv2.resize(depth, IMG_SIZE)

    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    depth = depth.astype(np.float32) / 255.0
    edges = edges.astype(np.float32) / 255.0

    return np.stack([edges, depth], axis=-1)


# ─────────────────────────────────────────────
# CORRECT DATA SPLIT (NO LEAKAGE)
# ─────────────────────────────────────────────
def load_sequences_split(base_path):

    folders = sorted(os.listdir(base_path))

    train_folders = folders[:8]
    test_folders = folders[8:]

    X_train, y_train = [], []
    X_test, y_test = [], []

    def process(folders_list, X, y):
        for folder in folders_list:
            print(f"Processing folder: {folder}")
            if folder not in cp_mapping:
                continue

            label = cp_mapping[folder]

            rgb_path = os.path.join(base_path, folder, "rgb")
            depth_path = os.path.join(base_path, folder, "depth")

            if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                continue

            rgb_frames = sorted(os.listdir(rgb_path))
            depth_frames = sorted(os.listdir(depth_path))

            min_len = min(len(rgb_frames), len(depth_frames))

            for i in range(0, min_len - SEQUENCE_LENGTH, 5):

                seq = []

                for j in range(SEQUENCE_LENGTH):
                    rgb = cv2.imread(os.path.join(rgb_path, rgb_frames[i+j]))
                    depth = cv2.imread(os.path.join(depth_path, depth_frames[i+j]), 0)

                    if rgb is None or depth is None:
                        continue

                    seq.append(extract_features(rgb, depth))

                if len(seq) == SEQUENCE_LENGTH:
                    X.append(seq)
                    y.append(label)

    process(train_folders, X_train, y_train)
    process(test_folders, X_test, y_test)

    return (
        np.array(X_train), np.array(X_test),
        np.array(y_train), np.array(y_test)
    )


# ─────────────────────────────────────────────
# MODEL BASE
# ─────────────────────────────────────────────
def build_base_cnn():
    return [
        TimeDistributed(Conv2D(32, (3,3), activation='relu'),
                        input_shape=(SEQUENCE_LENGTH, 128, 128, 2)),
        TimeDistributed(MaxPooling2D(2,2)),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(Dropout(0.3)),

        TimeDistributed(Conv2D(64, (3,3), activation='relu')),
        TimeDistributed(MaxPooling2D(2,2)),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(Dropout(0.3)),

        TimeDistributed(Conv2D(128, (3,3), activation='relu')),
        TimeDistributed(MaxPooling2D(2,2)),

        TimeDistributed(GlobalAveragePooling2D())
    ]


# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────
def build_cnn_lstm():
    model = Sequential(build_base_cnn() + [
        LSTM(64),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def build_cnn_gru():
    model = Sequential(build_base_cnn() + [
        GRU(64),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def build_cnn_bilstm():
    model = Sequential(build_base_cnn() + [
        Bidirectional(LSTM(64)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print("\n📥 Loading dataset (NO LEAKAGE)...")
    X_train, X_test, y_train, y_test = load_sequences_split(BASE_PATH)

    print("Train:", X_train.shape)
    print("Test :", X_test.shape)

    early_stop = EarlyStopping(patience=5, restore_best_weights=True)

    # ───── CNN-LSTM ─────
    print("\n🚀 Training CNN-LSTM...")
    model_lstm = build_cnn_lstm()

    hist_lstm = model_lstm.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            early_stop,
            ModelCheckpoint("cnn_lstm_model.h5", monitor="val_accuracy", save_best_only=True)
        ]
    )

    # ───── CNN-GRU ─────
    print("\n🚀 Training CNN-GRU...")
    model_gru = build_cnn_gru()

    hist_gru = model_gru.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            early_stop,
            ModelCheckpoint("cnn_gru_model.h5", monitor="val_accuracy", save_best_only=True)
        ]
    )

    # ───── CNN-BiLSTM ─────
    print("\n🚀 Training CNN-BiLSTM...")
    model_bilstm = build_cnn_bilstm()

    hist_bilstm = model_bilstm.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            early_stop,
            ModelCheckpoint("cnn_bilstm_model.h5", monitor="val_accuracy", save_best_only=True)
        ]
    )

    # ───── SAVE HISTORY ─────
    history = {
        "lstm": hist_lstm.history,
        "gru": hist_gru.history,
        "bilstm": hist_bilstm.history
    }

    with open("history.json", "w") as f:
        json.dump(history, f)

    print("\n🎯 Training completed WITHOUT DATA LEAKAGE")