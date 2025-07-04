# 🎥 Real-Time Suspicious Activity Detection using YOLOv8 & LSTMs

A deep learning project that identifies and classifies **suspicious behaviors like shoplifting** from video streams in **real-time**.
This system moves beyond simple classification by **tracking individuals** and analyzing their **actions over time**.

---

## 🚀 Project Overview

This project implements a **multi-stage pipeline** to analyze video footage and **flag potential shoplifting activities**.

Instead of classifying entire videos, it focuses on identifying specific "**atomic actions**" like:

* 🪹 Concealing
* 🚪 Loitering
* ✅ Normal Behavior

The system:

* Processes a video feed
* Tracks each person
* Labels their actions in **real-time**

---

## 🛠️ Tech Stack

* 🤖 **Language:** Python
* 📚 **Libraries:** TensorFlow, Keras, OpenCV, Ultralytics, NumPy, Scikit-learn
* 🧠 **Models:**

  * Object Detection & Tracking: **YOLOv8n**
  * Feature Extraction: **MobileNetV2**
  * Temporal Action Classification: **LSTM**
* 🎮 **Video Processing:** FFmpeg

---

## 📂 Project Structure

```bash
. (root directory)
├── anti_shoplift_model.ipynb     # Jupyter Notebook for training the LSTM model
├── inference.py                  # Full pipeline: tracking + classification
├── track_video.py                # YOLOv8 + tracking only
├── action_recognition_model.h5   # Saved LSTM model
├── action_classes.txt            # Action labels
├── yolov8n.pt                    # YOLOv8n model weights
├── requirements.txt              # Dependencies list
├── tracked_output.mp4            # YOLO tracking output
└── inference_output.mp4          # Final output with action labels
```

---

## ⚙️ How to Run

Follow these steps to set up and run the project on your local machine.

### 🗁 Step 1: Clone the Repository

```bash
git clone https://github.com/adityav-123/Real-Time-Suspicious-Activity-Detection-using-YOLOv8-LSTMs.git
cd Real-Time-Suspicious-Activity-Detection-using-YOLOv8-LSTMs
```

---

### 🔧 Step 2: Set Up the Environment

It is recommended to use a **virtual environment** for isolated package management.

```bash
# 🔧 Create a virtual environment
python -m venv venv

# 🚀 Activate it
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# 📦 Install the required dependencies
pip install -r requirements.txt
```

---

### 🎮 Step 3: Prepare the Data

Download the dataset from Kaggle:
🔗 [Shoplifting Video Dataset on Kaggle](https://www.kaggle.com/datasets/kipshidze/shoplifting-video-dataset)

After downloading:

* Extract full-length videos from the dataset.
* Use FFmpeg to extract short video clips of individual actions.

#### 💡 Example FFmpeg Command:

```bash
ffmpeg -i input.mp4 -ss 00:00:10 -t 5 -c copy concealing/clip1.mp4
```

---

### 🧠 Step 4: Train the Action Recognition Model

Run the notebook to train the **CNN+LSTM action classifier**:

```bash
anti_shoplift_model.ipynb
```

This will generate the following model artifacts:

* 🧠 `action_recognition_model.h5`
* 🏛️ `action_classes.txt`

---

### 🤖 Step 5: Run the Full Inference Pipeline

Run the script that combines tracking + action recognition:

```bash
python inference.py
```

📌 **Important:**
Edit the `VIDEO_PATH_IN` variable inside `inference.py` to point to your test video path.

This script will:

* 🔍 Detect & track individuals
* 🧠 Classify their actions
* 📀 Save the output as `inference_output.mp4`

---

## 🌊 Workflow

The system operates in **four major stages**:

### 🧕️ Stage 1: Person Detection & Tracking

* Utilizes a **YOLOv8** object detection model to detect people frame-by-frame.
* Assigns a **unique, persistent ID** to each detected person.

---

### 🎬 Stage 2: Data Curation

* Review source videos and extract labeled clips using FFmpeg:

  * `concealing/`, `loitering/`, and `normal/`

---

### 🧠 Stage 3: Action Model Training

* **MobileNetV2 (CNN)** extracts spatial features.
* **LSTM** captures temporal context across frame sequences.
* Trained via `anti_shoplift_model.ipynb`.

---

### 📊 Stage 4: Inference

* `inference.py` performs real-time prediction:

  * Detects and tracks people
  * Maintains frame history per person
  * Predicts action
  * Annotates the video with:

    * 🔑 Track ID
    * 🏷️ Action Label
    * 📊 Confidence Score

* Output is saved as `inference_output.mp4`

---

## 🔮 Future Improvements

### 📊 Suspicion Scoring System

* Build a rule-based score system for individuals
* Factor in:

  * Action sequence
  * Duration
  * Repetition

---

### 🌐 Deploy as a Web App

* Use **Docker** to package the pipeline
* Host on:

  * Hugging Face Spaces
  * Streamlit
  * AWS EC2 / Lambda

---

### 🎯 Improve Model Accuracy

* Expand and diversify dataset
* Apply augmentations:

  * Flip, brightness, zoom
* Try other models:

  * 3D CNNs, GRUs, Transformers

---

---

> 🧠 *This project showcases real-time human behavior understanding through deep learning.*
