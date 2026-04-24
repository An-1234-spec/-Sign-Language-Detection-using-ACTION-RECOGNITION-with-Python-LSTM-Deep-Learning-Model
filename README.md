# 🤟 Sign Language Detection using Action Recognition

> **Real-time Sign Language Detection** using LSTM Deep Learning Model with MediaPipe Holistic and TensorFlow/Keras.

---

## 📋 Overview

This project implements a **real-time sign language gesture recognition system** that detects and classifies sign language actions using body keypoint sequences. It leverages **MediaPipe Holistic** for extracting pose, face, and hand landmarks, and an **LSTM (Long Short-Term Memory) neural network** to classify temporal sequences of these keypoints into sign language gestures.

### Supported Gestures
| Gesture | Description |
|---------|-------------|
| 👋 `hello` | Waving hand gesture |
| 🙏 `thanks` | Thank you gesture |
| 🤟 `iloveyou` | I love you sign |

---

## 🏗️ Architecture

```
Input (Webcam Feed)
        │
        ▼
┌─────────────────────────┐
│   MediaPipe Holistic    │  → Extracts 1662 keypoints per frame
│   (Pose + Face + Hands) │     • Pose: 33×4 = 132
│                         │     • Face: 468×3 = 1404
│                         │     • Left Hand: 21×3 = 63
│                         │     • Right Hand: 21×3 = 63
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   Sequence Collection   │  → 30 frames per sequence
│   (Sliding Window)      │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   LSTM Neural Network   │  → Sequential model
│   • LSTM(64)            │     Input: (30, 1662)
│   • LSTM(128)           │     Trained for 2000 epochs
│   • LSTM(64)            │
│   • Dense(64)           │
│   • Dense(32)           │
│   • Dense(3, softmax)   │  → Output: gesture probabilities
└─────────────────────────┘
        │
        ▼
   Predicted Gesture
```

---

## 📁 Project Structure

```
├── RESULTS.ipynb          # Main Jupyter notebook (training + inference)
├── action.h5              # Trained LSTM model weights
├── requirements.txt       # Python dependencies
├── MP_Data/               # Collected keypoint data (numpy arrays)
│   ├── hello/             # 30 sequences × 30 frames each
│   ├── thanks/            # 30 sequences × 30 frames each
│   ├── iloveyou/          # 30 sequences × 30 frames each
│   └── OK/                # Additional gesture data
├── Logs/                  # TensorBoard training logs
│   └── train/
└── 0.npy                  # Sample extracted keypoint array
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Webcam (for real-time detection and data collection)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/An-1234-spec/-Sign-Language-Detection-using-ACTION-RECOGNITION-with-Python-LSTM-Deep-Learning-Model.git
   cd -Sign-Language-Detection-using-ACTION-RECOGNITION-with-Python-LSTM-Deep-Learning-Model
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/Mac
   venv\Scripts\activate       # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the notebook**
   ```bash
   jupyter notebook RESULTS.ipynb
   ```

---

## 🔬 Pipeline (Notebook Sections)

The `RESULTS.ipynb` notebook walks through the full pipeline:

| # | Section | Description |
|---|---------|-------------|
| 1 | **Install & Import** | Install packages and import libraries |
| 2 | **Keypoints using MP Holistic** | Set up MediaPipe Holistic model and drawing utilities |
| 3 | **Extract Keypoint Values** | Extract pose, face, and hand landmarks as numpy arrays |
| 4 | **Setup Folders for Collection** | Create directory structure for storing training data |
| 5 | **Collect Keypoint Values** | Record keypoint sequences from webcam for each gesture |
| 6 | **Preprocess Data** | Create feature arrays (X) and categorical labels (y), train/test split |
| 7 | **Build & Train LSTM** | Build Sequential LSTM model and train for 2000 epochs |
| 8 | **Make Predictions** | Run inference on test data |
| 9 | **Save Weights** | Save/load trained model weights (`action.h5`) |
| 10 | **Evaluation** | Confusion matrix and accuracy score |
| 11 | **Test in Real Time** | Live webcam detection with probability visualization |

---

## 📊 Model Details

- **Input Shape:** `(30, 1662)` — 30 frames of 1662 keypoint features
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Metric:** Categorical Accuracy
- **Training Epochs:** 2000
- **Test Split:** 5%

---

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| **TensorFlow / Keras** | LSTM model building and training |
| **MediaPipe** | Real-time pose, face, and hand landmark detection |
| **OpenCV** | Webcam capture and image processing |
| **NumPy** | Keypoint data storage and array operations |
| **scikit-learn** | Train/test splitting, evaluation metrics |
| **Matplotlib** | Data visualization |
| **SciPy** | Statistical analysis for predictions |

---

## 📈 Training Visualization

Monitor training progress with TensorBoard:
```bash
tensorboard --logdir=Logs
```

---

## 🤝 Acknowledgements

Inspired by [Nicholas Renotte's](https://www.youtube.com/nicolasrenotte) Sign Language Detection tutorial using Action Recognition with Python.

---

## 📄 License

This project is for educational purposes. Feel free to use and modify.
