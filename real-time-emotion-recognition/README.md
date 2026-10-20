# Real-Time Facial Emotion Recognition Using Deep Learning

An educational, portfolio-grade computer vision and web application that captures live webcam frames, performs facial expression classification in near real-time using deep learning, and displays dynamic confidence scores on an interactive web dashboard.

Built with **Python**, **Flask**, **DeepFace**, **OpenCV**, and vanilla **HTML5 / CSS3 / JavaScript**.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [System Architecture & Pipeline](#system-architecture--pipeline)
4. [Technology Stack](#technology-stack)
5. [Project Folder Structure](#project-folder-structure)
6. [System Requirements](#system-requirements)
7. [Installation & Setup (Windows / PowerShell)](#installation--setup-windows--powershell)
8. [Running the Application](#running-the-application)
9. [How It Works](#how-it-works)
10. [Testing Scenarios](#testing-scenarios)
11. [Troubleshooting & Common Issues](#troubleshooting--common-issues)
12. [Privacy & Educational Disclaimer](#privacy--educational-disclaimer)
13. [Future Enhancements](#future-enhancements)
14. [Licenses & Third-Party Attributions](#licenses--third-party-attributions)

---

## Project Overview

In computer vision and human-computer interaction (HCI), facial expression analysis is a foundational task. This project demonstrates an end-to-end client-server architecture:
- **Client (Browser)**: Interfaces with the user's camera using the modern `MediaDevices.getUserMedia()` API, captures frame snapshots onto an offscreen HTML5 `<canvas>`, and asynchronously sends frames to the server via `fetch()`.
- **Server (Flask & DeepFace)**: Decodes frames strictly in-memory (RAM) using OpenCV, runs a convolutional neural network (CNN) model using DeepFace to predict facial expression categories, and returns JSON-formatted confidence scores.
- **Dashboard**: Dynamically updates the dominant predicted expression, custom emoji representation, round-trip latency, and animated progress bars for seven key emotional expressions.

> **Scientific Note:** This project performs **facial expression classification** based on surface facial muscle movements. It does **not** claim to measure or reveal a human being's true internal psychological emotional state.

---

## Key Features

- **Real-Time Webcam Analysis**: Captures and processes frames at periodic intervals (1.8s) to deliver smooth feedback without exhausting CPU resources.
- **In-Memory Frame Processing**: Preserves user privacy—no webcam images or video streams are ever saved to disk or transmitted to third-party clouds.
- **Robust Error Handling**:
  - Automatically identifies when no face is visible in the frame and guides the user.
  - Rejects empty, corrupted, or missing multipart uploads with accurate HTTP status codes (400 / 500).
  - Handles camera permissions and device disconnection gracefully in the frontend.
- **Request Serialization**: Implements concurrency locking (`isAnalyzing`) in JavaScript to prevent overlapping asynchronous requests when inference latency fluctuates.
- **Safe DOM Manipulation**: Exclusively utilizes safe DOM APIs (`textContent`, `replaceChildren`, `createElement`) to prevent Cross-Site Scripting (XSS).
- **Responsive Dashboard**: Professional slate-blue theme built with CSS Flexbox and Grid, fully responsive across desktop and mobile screens.
- **Model Warmup**: Pre-caches model weights at backend startup to eliminate cold-start delays on initial user requests.

---

## System Architecture & Pipeline

```text
+---------------------------------------------------------------------------------+
|                                 CLIENT BROWSER                                  |
|                                                                                 |
|  [Webcam Stream] ---> [<video> Live Feed]                                      |
|                             |                                                   |
|                      (Every 1.8 seconds)                                        |
|                             v                                                   |
|                   [Hidden HTML5 <canvas>]                                       |
|                             | (Canvas to Blob / JPEG)                           |
|                             v                                                   |
|                    [FormData Payload]                                           |
+-----------------------------|---------------------------------------------------+
                              | HTTP POST /analyze (multipart/form-data)
                              v
+---------------------------------------------------------------------------------+
|                              FLASK BACKEND (app.py)                             |
|                                                                                 |
|  1. Request Validation (Ensure 'frame' exists, non-empty, valid format)         |
|  2. In-Memory Decoding: cv2.imdecode(np.frombuffer(bytes, np.uint8))           |
|  3. Face Detection & Preprocessing (Haar Cascade / OpenCV)                      |
|  4. DeepFace Emotion Inference (Convolutional Neural Network)                   |
|  5. Result Formatting: Dominant expression + normalized confidence %           |
+-----------------------------|---------------------------------------------------+
                              | HTTP 200 OK (application/json)
                              v
+---------------------------------------------------------------------------------+
|                               CLIENT UI UPDATE                                  |
|                                                                                 |
|  - Dominant Expression Tag & Emoji (😊, 😢, 😠, 😨, 😲, 🤢, 😐)                  |
|  - Ranked Confidence Meters (0% - 100%)                                         |
|  - Inference Latency (ms)                                                       |
|  - Real-Time Status Hints                                                       |
+---------------------------------------------------------------------------------+
```

---

## Technology Stack

- **Backend**:
  - **Python 3.13** (Recommended & fully wheel-compatible)
  - **Flask**: Lightweight WSGI web application framework
  - **DeepFace**: Lightweight facial analysis library
  - **OpenCV (`opencv-contrib-python-headless`)**: In-memory image decoding and Haar Cascade face detection
  - **NumPy**: Multidimensional array processing for image tensors
  - **ONNX Runtime**: Optimized inference engine for neural network weights
- **Frontend**:
  - **HTML5**: Semantic tags, accessible ARIA attributes
  - **CSS3**: Custom properties, Flexbox, Grid layout, mobile responsiveness
  - **JavaScript (ES6+)**: `MediaDevices` API, HTML5 Canvas, asynchronous `fetch`, DOM API

---

## Project Folder Structure

```text
real-time-emotion-recognition/
├── app.py                  # Flask server, API endpoints, DeepFace inference
├── requirements.txt        # Direct project dependencies
├── README.md               # Complete project documentation
├── .gitignore              # Files and folders excluded from Git tracking
├── templates/
│   └── index.html          # Web dashboard structure and layout
└── static/
    ├── style.css           # Modern slate-blue responsive styling
    └── script.js           # Webcam controls, frame capture, async API calls
```

---

## System Requirements

- **Operating System**: Windows 10 / 11, macOS, or Linux
- **Python**: **Python 3.10, 3.11, 3.12, or 3.13**
  > **Notice regarding Python 3.15+**: Python 3.15 is a preview/bleeding-edge release without precompiled wheels for machine learning frameworks (`tensorflow`, `onnxruntime`). Use Python 3.13 for full compatibility.
- **Hardware**: Integrated or external USB Webcam
- **Browser**: Modern Chromium-based browser (Google Chrome, Microsoft Edge, Brave) or Mozilla Firefox with camera access enabled

---

## Installation & Setup (Windows / PowerShell)

Follow these step-by-step instructions in **Visual Studio Code** or the **Windows PowerShell** terminal:

### 1. Open the Project Folder
Open Visual Studio Code in the project directory:
```powershell
cd C:\Users\rajes\OneDrive\Desktop\real-time-emotion-recognition
```

### 2. Verify Your Python Installation
Check the Python versions available on your machine:
```powershell
py -0
```
Ensure Python 3.13 is available.

### 3. Create a Virtual Environment Named `.venv`
Create the virtual environment using Python 3.13:
```powershell
py -3.13 -m venv .venv
```

### 4. Activate the Virtual Environment
Activate the environment in PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

> **PowerShell Script Execution Policy Note**: If you encounter an error stating `running scripts is disabled on this system`, run this command in your PowerShell session:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
> ```
> Then re-run `.\.venv\Scripts\Activate.ps1`. When activated, your terminal prompt will be prefixed with `(.venv)`.

### 5. Install Dependencies
Install all required packages from `requirements.txt`:
```powershell
pip install -r requirements.txt
```

---

## Running the Application

### 1. Start the Flask Server
Make sure your virtual environment is active, then execute:
```powershell
python app.py
```

You will see console output indicating that the model is warmed up and ready:
```text
[INFO] emotion_app: Pre-warming DeepFace emotion classification model...
[INFO] emotion_app: Emotion classification model is warm and ready.
[INFO] emotion_app: Starting Real-Time Facial Emotion Recognition server at http://127.0.0.1:5000
 * Serving Flask app 'app'
 * Running on http://127.0.0.1:5000
```

### 2. Open the Web Application
Open your web browser and navigate to:
```text
http://127.0.0.1:5000
```

### 3. Grant Camera Permission & Test
1. Click **Start Camera**.
2. When prompted by your browser, click **Allow** to permit webcam access.
3. Align your face within the camera feed.
4. Observe the live dominant expression badge, emoji, confidence score meters, and latency counter.
5. Click **Stop Camera** at any time to release your webcam hardware.

---

## How It Works

### Expression Categories
The DeepFace emotion model classifies faces into **7 standardized categories**:
- **Happy** (😊)
- **Sad** (😢)
- **Angry** (😠)
- **Fear** (😨)
- **Surprise** (😲)
- **Disgust** (🤢)
- **Neutral** (😐)

### Client-Side Processing (`static/script.js`)
1. **Camera Stream**: Calls `navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false })`.
2. **Snapshot Extraction**: Draws the current frame from `<video>` onto an offscreen `<canvas>` with dimensions matched to the active video resolution.
3. **Blob Conversion**: Encodes the canvas image as a JPEG binary blob (`canvas.toBlob(...)`).
4. **Multipart Upload**: Uses `fetch('/analyze')` with `FormData` to send the blob as `frame.jpg`.
5. **DOM Updates**: On receiving JSON, the DOM elements are updated via `textContent` and style properties. The dominant category is highlighted in bold blue.

### Server-Side Processing (`app.py`)
1. **Validation**: Confirms `frame` exists, is non-empty, and parses into a NumPy array.
2. **Decoding**: Converts raw bytes to BGR image array via `cv2.imdecode(..., cv2.IMREAD_COLOR)` directly in RAM.
3. **Inference**: Invokes `DeepFace.analyze(img_path=frame, actions=['emotion'], detector_backend='opencv', enforce_detection=True)`.
4. **No-Face Detection**: If no face is detected, DeepFace raises a `ValueError`. The backend intercepts this exception and cleanly responds with `{"status": "no_face", "message": "No face detected in the frame. Please face the camera."}` and HTTP 200.
5. **Output**: Returns JSON with the dominant classification and rounded scores.

---

## Testing Scenarios

| Test Case | User Action | Expected Frontend Reaction | Expected Backend Behavior |
| :--- | :--- | :--- | :--- |
| **1. Baseline Neutral** | Look directly at camera with a calm, relaxed face | Displays **Neutral** (😐) with highest confidence bar | Returns HTTP 200 with `dominant_emotion: "neutral"` |
| **2. Smile / Joy** | Smile widely showing teeth | Displays **Happy** (😊) with dominant score | Returns HTTP 200 with `dominant_emotion: "happy"` |
| **3. Surprise** | Raise eyebrows and open mouth wide | Displays **Surprise** (😲) | Returns HTTP 200 with `dominant_emotion: "surprise"` |
| **4. Face Obscured / Absent** | Cover camera lens or step out of frame | Displays **No Face Detected** (🤔) with amber warning | Returns HTTP 200 with `status: "no_face"` |
| **5. Stop Camera** | Click "Stop Camera" button | Webcam turns off, LED turns off, dashboard resets | Loop terminated, tracks stopped via `track.stop()` |
| **6. Corrupt / Missing Frame** | Send empty POST to `/analyze` | Displays error alert banner | Returns HTTP 400 with `status: "error"` |

---

## Troubleshooting & Common Issues

### 1. `ResolutionImpossible` or DeepFace installation fails on Python 3.15
- **Cause**: Python 3.15 is an early release. Machine learning wheels (`onnxruntime`, `tensorflow`) do not yet build wheels for 3.15.
- **Solution**: Install and use Python 3.13:
  ```powershell
  py -3.13 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```

### 2. `Activate.ps1 cannot be loaded because running scripts is disabled`
- **Cause**: Windows PowerShell security policy restricts unsigned script execution by default.
- **Solution**: Run this command once in your current PowerShell session:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
  ```

### 3. Camera LED stays on after clicking "Stop Camera"
- **Cause**: MediaStream track was not properly closed.
- **Solution**: Our script executes `track.stop()` across all active media tracks, releasing hardware access immediately. Refresh the page if another application is holding the camera.

### 4. `Port 5000 is already in use`
- **Cause**: Another Flask or background process is running on port 5000.
- **Solution**: Terminate the previous python process or change `port=5000` to `port=5050` in `app.py`.

---

## Privacy & Educational Disclaimer

- **Privacy Guarantee**: No webcam images, video streams, or biometric templates are stored on disk, logged to databases, or transmitted across external networks. All computations happen in volatile system RAM on your local machine.
- **Educational Scope**: Facial expression recognition models map muscle contractions and facial geometry to nominal classification labels (Ekman categories). These classifications reflect surface appearance only and do not provide scientific ground truth regarding a person's inner psychological feelings, thoughts, or intentions.

---

## Future Enhancements

- [ ] **Multi-Face Analysis**: Detect and display bounding boxes and expressions for multiple people simultaneously.
- [ ] **Temporal Smoothing**: Implement an Exponential Moving Average (EMA) or Kalman Filter across sequential frames to reduce prediction jitter.
- [ ] **Emotion Trend Analytics**: Add a Chart.js timeline showing expression transitions over a session.
- [ ] **Custom Lighting Normalization**: Apply Histogram Equalization (CLAHE) to improve detection in dim lighting conditions.

---

## Licenses & Third-Party Attributions

- **Flask**: Pallets Projects (BSD-3-Clause)
- **DeepFace**: Sefik Ilkin Serengil (MIT License)
- **OpenCV**: OpenCV Team (Apache 2.0 License)
- **NumPy**: NumPy Developers (BSD License)
- **Inter Font**: Rasmus Andersson (SIL Open Font License)
