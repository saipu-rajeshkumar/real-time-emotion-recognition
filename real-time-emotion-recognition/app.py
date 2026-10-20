"""
Real-Time Facial Emotion Recognition Backend
Powered by Flask, OpenCV, and DeepFace
"""

import logging
import os
import sys

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from deepface import DeepFace

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("emotion_app")

# Initialize Flask application
app = Flask(__name__)

# Maximum allowed payload size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


@app.route("/")
def home():
    """
    Renders the web dashboard.
    """
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Receives an uploaded frame from the webcam, decodes it into an OpenCV image,
    and performs facial expression recognition using DeepFace.
    
    Returns:
        JSON response with the dominant expression, emotion scores, and status.
    """
    # 1. Validate that the frame file is present in request
    if "frame" not in request.files:
        logger.warning("Received /analyze request without 'frame' field.")
        return jsonify({
            "status": "error",
            "error": "Missing image file. Please provide a 'frame' in the multipart form data."
        }), 400

    file = request.files["frame"]

    # 2. Check for empty filename or empty file content
    if file.filename == "":
        logger.warning("Received empty filename in /analyze request.")
        return jsonify({
            "status": "error",
            "error": "The uploaded frame filename is empty."
        }), 400

    image_bytes = file.read()
    if not image_bytes:
        logger.warning("Received 0 bytes in /analyze request.")
        return jsonify({
            "status": "error",
            "error": "The uploaded image is empty."
        }), 400

    # 3. Decode image bytes into an OpenCV BGR NumPy array in memory
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as exc:
        logger.error("Failed to parse image buffer: %s", exc)
        return jsonify({
            "status": "error",
            "error": "Failed to decode image buffer."
        }), 400

    if frame is None or frame.size == 0:
        logger.warning("cv2.imdecode returned None or empty array.")
        return jsonify({
            "status": "error",
            "error": "Invalid image data. Could not decode into a valid picture."
        }), 400

    # 4. Perform Facial Expression Analysis using DeepFace
    try:
        # We use OpenCV Haar Cascade detector backend for fast CPU performance
        # enforce_detection=True ensures that frames without a clear face raise ValueError
        results = DeepFace.analyze(
            img_path=frame,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=True,
            silent=True
        )

        if not results:
            return jsonify({
                "status": "no_face",
                "message": "No face detected in the frame. Please face the camera."
            }), 200

        result = results[0]
        raw_emotions = result.get("emotion", {})
        dominant_emotion = result.get("dominant_emotion", "neutral")

        # Format expression scores rounded to two decimal places
        formatted_scores = {
            category.lower(): round(float(score), 2)
            for category, score in raw_emotions.items()
        }

        logger.info("Detected expression: %s (confidence: %s%%)",
                    dominant_emotion,
                    formatted_scores.get(dominant_emotion.lower(), "--"))

        return jsonify({
            "status": "success",
            "dominant_emotion": dominant_emotion.lower(),
            "emotion": dominant_emotion.capitalize(),
            "scores": formatted_scores
        }), 200

    except ValueError as ve:
        # DeepFace raises ValueError when enforce_detection=True and no face is located
        logger.info("Face detection: No face detected in frame.")
        return jsonify({
            "status": "no_face",
            "message": "No face detected. Please align your face with the camera."
        }), 200

    except Exception as exc:
        # Catch unexpected errors, log securely, and return helpful message
        logger.exception("Unexpected error during emotion analysis: %s", exc)
        return jsonify({
            "status": "error",
            "error": "An error occurred while analyzing the facial expression. Please try again."
        }), 500


# Warm up the emotion model at application startup to eliminate cold-start latency
def preload_model():
    try:
        from deepface.modules import modeling
        logger.info("Pre-warming DeepFace emotion classification model...")
        modeling.build_model(task="facial_attribute", model_name="Emotion")
        logger.info("Emotion classification model is warm and ready.")
    except Exception as e:
        logger.warning("Model warm-up note: %s. Model will initialize upon first request.", e)


if __name__ == "__main__":
    preload_model()
    # Development server bound strictly to localhost (127.0.0.1) for secure local execution
    logger.info("Starting Real-Time Facial Emotion Recognition server at http://127.0.0.1:5000")
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
