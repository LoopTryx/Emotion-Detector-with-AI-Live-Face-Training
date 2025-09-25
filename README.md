# 🎭 Face Emotion Detector

A modern **Face Recognition + Emotion Detection system** built with **Python, Tkinter, OpenCV, MediaPipe, and TensorFlow**.  
This project uses your webcam to detect face landmarks, record datasets, train a neural network (MLP), and predict emotions in real-time.

Oh! And I should also tell you, **MediaPipe**, an open-source framework made by google, is only available on Python versions **<= 3.11**. I used **pyenv** to install Python **3.11.8**. You may encounter some difficulties but do not be disheartened!

---

## ✨ Features

- 🖥️ **Sleek Tkinter GUI**
  - Live webcam feed displayed in-app.
  - Status bar showing recording state and model status.

- 👤 **Face Landmark Detection**
  - Powered by **MediaPipe FaceMesh** (468 face landmarks).
  - Plots points on the face in real-time.

- 📝 **Dataset Recording**
  - Press keys to label emotions while recording.
  - Saves samples to `landmarks_dataset.csv`.

- 🧠 **Neural Network Training**
  - Small **MLP (Multi-Layer Perceptron)** built with TensorFlow/Keras.
  - Can train directly from your recorded CSV dataset.
  - Saves trained model to `emotion_model.h5`.

- 😀 **Real-Time Emotion Prediction**
  - Once trained, detects emotions live from webcam input.
  - Supported classes:
    - `Happy`
    - `Sad`
    - `Angry`
    - `Surprise`
    - `Neutral`

---

## 🎮 Controls

- `r` → Toggle recording ON/OFF  
- `1` → Record sample as **Happy**  
- `2` → Record sample as **Sad**  
- `3` → Record sample as **Angry**  
- `4` → Record sample as **Surprise**  
- `5` → Record sample as **Neutral**  
- `t` → Train model on collected samples  
- `q` → Quit  

---

## 📂 Files

- `Face Emotion Detector.py` → Main Tkinter + webcam application  
- `landmarks_dataset.csv` → Recorded dataset of landmarks and labels  
- `emotion_model.h5` → Trained neural network model (saved automatically)  
- `scaler_mean.npy` / `scaler_scale.npy` → Normalization data  
- `requirements.txt` → Dependencies for easy installation  

---

## ⚙️ Installation

1. Clone or download the repo.  
2. Create a Python virtual environment (recommended).  
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows

    Install dependencies:

python -m pip install -r requirements.txt

Run the program:

    python "Face Emotion Detector.py"

🧮 How It Works

    Face Landmark Extraction
    MediaPipe detects 468 landmarks on your face → converted into vectors.

    Recording Samples
    Landmarks saved to landmarks_dataset.csv with a label (0–4).

    Training the Model
    A Multi-Layer Perceptron (MLP) learns patterns in landmarks.

        Dense layers with ReLU activation.

        Output uses Softmax for classification.

    Prediction
    Live landmarks normalized → fed into trained model → outputs emotion class.

📊 Model Architecture

    Input: Flattened face landmarks (x, y pairs).

    Dense (256) → ReLU + Dropout

    Dense (128) → ReLU + Dropout

    Dense (64) → ReLU

    Dense (5) → Softmax (for 5 emotions).

❓ FAQ

Q: Why is my model inaccurate?

    You need dozens of samples per class (record more).

    Ensure good lighting and camera angle.

Q: What if my CSV is broken?

    Delete landmarks_dataset.csv and record again.


🙏 Thanks

    MediaPipe FaceMesh OpenCV TensorFlow Tkinter
    
    
This project was really tiring and really interesting! Months of work for understanding machine language took place for this moment.
Special thanks to you for exploring this project! 🚀

These projects take a lot of effort so I would love it if you could donate to my wallet!

Ethereum: 0x4cE9a921682eeF218F84EC141bF9cd1443C58E15

Bitcoin: bc1p7nkmpr52qp8l57l9hm6dj28e4rwgrqkpucuuau0s0favu6cxk4hs462g4k

Solana: 8LPEcUKwMoweSjTV3k4RGq4mF3eNF9jYW87Tq54JN917

Big Thanks!
