"""
Aha yes my hard work.

How to use:
- Press `r` to toggle recording ON/OFF.
- While recording:
    1 -> Happy
    2 -> Sad
    3 -> Angry
    4 -> Surprise
    5 -> Neutral
    6 -> Fear
- Press `t` to train (only works if CSV has sufficient samples).
"""

import os
import threading
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageTk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models

import tkinter as tk

# ---- Config ----
CSV_PATH = "landmarks_dataset.csv"
MODEL_PATH = "emotion_model.h5"
SCALER_MEAN = "scaler_mean.npy"
SCALER_SCALE = "scaler_scale.npy"

EMOTION_CLASSES = ["Happy", "Sad", "Angry", "Surprise", "Neutral", "Fear"] # Emotion Classes
LANDMARKS_TO_USE = None

# GUI Config
WINDOW_TITLE = "Face Emotion - Live (press r to record, 1..5 to label, t to train)"
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
POINT_RADIUS = 1
LINE_THICKNESS = 1

# ---- MediaPipe setup ----
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=False,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# ---- Utility functions ----

def landmarks_to_vector(landmarks, w, h, indices=None):
    """
    MediaPipe landmarks that has been flattened into 2D values(x, y)
    """
    if landmarks is None:
        return None
    pts = []
    if indices is None:
        indices = range(len(landmarks))
    for i in indices:
        lm = landmarks[i]
        x = min(max(int(lm.x * w), 0), w - 1)
        y = min(max(int(lm.y * h), 0), h - 1)
        pts.extend([x, y])
    return np.array(pts, dtype=np.float32)


def ensure_csv_exists():
    """Ensure CSV file exists; if not, it's time to exist."""
    if not os.path.exists(CSV_PATH):
        open(CSV_PATH, "w").close()


# ---- Recorder ----

class Recorder:
    """
    Append landmark vectors + label to CSV.
    """
    def __init__(self, csv_path=CSV_PATH):
        self.csv_path = csv_path
        ensure_csv_exists() # Ensures csv exists.
        self.n_features = None
        if os.path.getsize(self.csv_path) > 0:
            try:
                df = pd.read_csv(self.csv_path, nrows=1)
                cols = [c for c in df.columns if c != "label"]
                self.n_features = len(cols)
                print(f"[Recorder] Existing CSV detected with {self.n_features} features.")
            except Exception:
                # malformed header; treat as empty, overwrite it on first append.
                self.n_features = None

    def append(self, vector: np.ndarray, label: int):
        """
        Append a single sample. If CSV is empty because there is a chance that CSV is empty incase if you did not know, create a header based on vector length.
        If CSV is not empty, require same vector length as header.
        """
        if vector is None:
            print("[Recorder] No vector to save (no face detected).")
            return False

        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)

        vec_len = vector.size
        if self.n_features is None:
            # create header now
            self.n_features = vec_len
            columns = [f"x{i}" for i in range(self.n_features)] + ["label"]
            df = pd.DataFrame([list(vector) + [int(label)]], columns=columns)
            df.to_csv(self.csv_path, index=False)
            print(f"[Recorder] Created CSV header with {self.n_features} features and saved label={label}.")
            return True
        else:
            if vec_len != self.n_features:
                print(f"[Recorder] ERROR: vector length {vec_len} doesn't match CSV feature length {self.n_features}.")
                print("[Recorder] This happens if LANDMARKS_TO_USE changed between runs. Delete or rename CSV to start fresh.")
                return False
            # append
            row = pd.DataFrame([list(vector) + [int(label)]])
            row.to_csv(self.csv_path, mode="a", header=False, index=False)
            print(f"[Recorder] Appended sample label={label}.")
            return True


# ---- Model functions ----

def build_model(input_dim, n_classes): # int
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.35))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(n_classes, activation="softmax"))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def train_model_from_csv(csv_path=CSV_PATH, model_path=MODEL_PATH, epochs=30, batch_size=32):
    """
    Load landmarks CSV and train the model.
    """
    if not os.path.exists(csv_path):
        print("[Trainer] No dataset CSV found. Record data first.")
        return None

    if os.path.getsize(csv_path) == 0:
        print("[Trainer] Dataset CSV is empty. Record samples before training.")
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print("[Trainer] Unable to read CSV – it may be malformed. Error:", e)
        return None

    if "label" not in df.columns:
        print("[Trainer] CSV missing 'label' column. Delete CSV and re-record.") 
        return None

    feature_cols = [c for c in df.columns if c != "label"]
    if len(feature_cols) == 0:
        print("[Trainer] No feature columns found in CSV.")
        return None

    # ensure all feature columns numeric or something
    try:
        X_raw = df[feature_cols].astype(np.float32).values
        y = df["label"].astype(np.int32).values
    except Exception as e:
        print("[Trainer] CSV contains non-numeric values. Error:", e)
        return None

    # check consistent feature vector length
    expected_len = X_raw.shape[1]
    if expected_len % 2 != 0:
        print("[Trainer] Warning: feature length is odd (expected pairs of x,y). Continuing anyway.")

    if df.shape[0] < 10:
        print("[Trainer] Not enough samples to train. Collect more (aim for dozens per class).")
        return None

    # scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # stratity because yes.
    stratify = y if len(np.unique(y)) > 1 and min(np.bincount(y)) > 1 else None
    try:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=stratify)
    except Exception:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

    model = build_model(X.shape[1], len(EMOTION_CLASSES))

    print("[Trainer] Starting training...")
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size)

    # save model and scaler arrays
    model.save(model_path)
    np.save(SCALER_MEAN, scaler.mean_)
    np.save(SCALER_SCALE, scaler.scale_)
    print(f"[Trainer] Model saved to {model_path}. Scaler saved to {SCALER_MEAN}/{SCALER_SCALE}.")
    return model

# load thy model
def load_trained_model(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        return None, None
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print("[Loader] Failed to load model:", e)
        return None, None

    if os.path.exists(SCALER_MEAN) and os.path.exists(SCALER_SCALE):
        mean = np.load(SCALER_MEAN)
        scale = np.load(SCALER_SCALE)
        scaler = (mean, scale)
    else:
        scaler = None
    return model, scaler


# ---- GUI & Camera Thread ----
class FaceEmotionApp:
    def __init__(self, window, video_source=0):
        print("[App] Initializing GUI...")
        self.window = window
        self.window.title(WINDOW_TITLE)
        self.video_source = video_source
        self.vid = cv2.VideoCapture(video_source)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)

        self.canvas = tk.Canvas(window, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT)
        self.canvas.pack()

        # status frame
        btn_frame = tk.Frame(window)
        btn_frame.pack(fill=tk.X, expand=False)
        self.rec_label_var = tk.StringVar(value="Recording: OFF")
        self.pred_var = tk.StringVar(value="Model: none")
        tk.Label(btn_frame, textvariable=self.rec_label_var).pack(side=tk.LEFT, padx=6)
        tk.Label(btn_frame, textvariable=self.pred_var).pack(side=tk.RIGHT, padx=6)

        # state
        self.is_running = True
        self.recording = False
        self.recorder = Recorder()
        self.model, self.scaler_tuple = load_trained_model()
        if self.model is not None:
            self.pred_var.set("Model: loaded")
            print("[App] Loaded trained model.")
        else:
            self.pred_var.set("Model: none")

        # prediction smoothing
        self.recent_preds = deque(maxlen=8)

        # keyboard bindings
        window.bind("<Key>", self.on_key)

        # thread for camera
        self.frame_image = None
        self.thread = threading.Thread(target=self._video_loop, daemon=True)
        self.thread.start()
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

    def on_key(self, event):
        key = event.char.lower()
        if key == "r":
            self.recording = not self.recording
            self.rec_label_var.set(f"Recording: {'ON' if self.recording else 'OFF'}")
            print("[App] Recording toggled:", self.recording)
        elif key == "t":
            print("[App] Starting training thread...")
            t = threading.Thread(target=self._train_thread, daemon=True)
            t.start()
        elif key in ("1", "2", "3", "4", "5", "6"):
            if self.recording:
                label_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5}
                self.pending_label = label_map[key]
                self._save_pending_label = True
                print(f"[App] Queued label: {key}")
            else:
                print("[App] Toggle recording (r) before saving samples.")
        elif key == "q":
            self._on_close()

    def _train_thread(self):
        trained = train_model_from_csv()
        if trained is not None:
            self.model, scaler_tuple = load_trained_model()
            if self.model is not None:
                self.pred_var.set("Model: loaded")
                print("[App] Training complete and model loaded.")

    def _video_loop(self):
        self._save_pending_label = False
        self.pending_label = None
        while self.is_running:
            ret, frame = self.vid.read()
            if not ret:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            overlay = frame.copy()
            pts_vector = None

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                indices = LANDMARKS_TO_USE
                pts_vector = landmarks_to_vector(landmarks, w, h, indices)
                
                # -- Useless Stuff I added because I thought it would be cool --
                for i, lm in enumerate(landmarks):
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    if x > w // 2:
                        # right side -> nodes only
                        cv2.circle(overlay, (x, y), POINT_RADIUS + 1, (255, 255, 255), -1)
                    else:
                        # left side -> denser small dots
                        cv2.circle(overlay, (x, y), POINT_RADIUS, (255, 255, 255), -1)

                # draw simple connector graph on right side
                right_pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks if int(lm.x * w) > w // 2]
                if len(right_pts) >= 5:
                    sample = right_pts[::max(1, len(right_pts) // 6)][:6]
                    for a in range(len(sample) - 1):
                        cv2.line(overlay, sample[a], sample[a + 1], (255, 255, 255), LINE_THICKNESS)
                    for p in sample:
                        cv2.circle(overlay, p, POINT_RADIUS + 2, (255, 255, 255), 2)

            # handle recording save
            if self.recording and getattr(self, "_save_pending_label", False) and pts_vector is not None:
                ok = self.recorder.append(pts_vector, self.pending_label)
                self._save_pending_label = False
                if not ok:
                    print("[App] Failed to save sample (vector shape mismatch).")

            # prediction
            if pts_vector is not None and self.model is not None:
                vec = pts_vector.astype(np.float32).reshape(1, -1)
                if self.scaler_tuple is None:
                    # try load scaler files
                    if os.path.exists(SCALER_MEAN) and os.path.exists(SCALER_SCALE):
                        mean = np.load(SCALER_MEAN)
                        scale = np.load(SCALER_SCALE)
                        self.scaler_tuple = (mean, scale)
                if self.scaler_tuple is not None:
                    mean, scale = self.scaler_tuple
                    # guard shape
                    if vec.shape[1] == mean.shape[0]:
                        vec = (vec - mean.reshape(1, -1)) / scale.reshape(1, -1)
                    else:
                        # scaler mismatch -> skip prediction
                        cv2.putText(overlay, "Scaler/model mismatch", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        vec = None
                if vec is not None:
                    try:
                        preds = self.model.predict(vec, verbose=0)[0]
                        top_idx = int(np.argmax(preds))
                        conf = float(preds[top_idx])
                        self.recent_preds.append((top_idx, conf))
                        # aggregate by sum of confidences
                        agg = {}
                        for idx, c in self.recent_preds:
                            agg[idx] = agg.get(idx, 0.0) + c
                        best = max(agg.items(), key=lambda x: x[1])[0]
                        text = f"{EMOTION_CLASSES[best]} {conf:.2f}"
                        cv2.putText(overlay, text, (10, DISPLAY_HEIGHT - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    except Exception as e:
                        cv2.putText(overlay, "Prediction error", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        print("[App] Prediction error:", e)
                        
                        # disgusting

            # convert and draw to Tkinter canvas
            overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            display = cv2.resize(overlay, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            self.frame_image = Image.fromarray(display)
            self.photo = ImageTk.PhotoImage(image=self.frame_image)
            # use create_image with anchor to (0,0)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.vid.release()

    def _on_close(self):
        print("[App] Closing...")
        self.is_running = False
        try:
            self.vid.release()
        except Exception:
            pass
        self.window.quit()
        self.window.destroy()


def main():
    ensure_csv_exists()
    print("[Main] Starting FaceEmotionApp")
    model, scaler_tuple = load_trained_model()
    if scaler_tuple is not None:
        scaler = (np.load(SCALER_MEAN), np.load(SCALER_SCALE))
    else:
        scaler = None

    root = tk.Tk()
    app = FaceEmotionApp(root)
    if scaler is not None:
        app.scaler_tuple = scaler
    root.mainloop() # 6777


if __name__ == "__main__":
    main()
