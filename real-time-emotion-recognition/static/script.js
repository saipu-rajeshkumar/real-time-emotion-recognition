/**
 * Real-Time Facial Emotion Recognition Client
 * 
 * Features:
 * - Webcam stream acquisition via getUserMedia
 * - Canvas frame extraction & asynchronous multipart upload to Flask /analyze
 * - Concurrency locking (prevents overlapping fetch requests)
 * - Safe DOM manipulation (strict use of textContent and DOM APIs)
 * - Graceful handling of "no-face", permission denied, network failures
 * - Complete resource cleanup on camera stop
 */

(() => {
    'use strict';

    // DOM Elements
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('captureCanvas');
    const videoPlaceholder = document.getElementById('videoPlaceholder');
    const scanningIndicator = document.getElementById('scanningIndicator');
    
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    
    const cameraStatusBadge = document.getElementById('cameraStatusBadge');
    const cameraStatusText = document.getElementById('cameraStatusText');
    const analysisStatus = document.getElementById('analysisStatus');
    
    const expressionEmoji = document.getElementById('expressionEmoji');
    const dominantExpression = document.getElementById('dominantExpression');
    const dominantConfidence = document.getElementById('dominantConfidence');
    const feedbackMessage = document.getElementById('feedbackMessage');
    const scoresContainer = document.getElementById('scoresContainer');
    const inferenceLatency = document.getElementById('inferenceLatency');

    // Runtime state variables
    let mediaStream = null;
    let analysisIntervalId = null;
    let isAnalyzing = false;
    const ANALYSIS_INTERVAL_MS = 1800; // Sample every 1.8 seconds to balance real-time feel and CPU usage

    // Standard Emotion Emoji & Metadata mapping (using unicode escape literals to prevent encoding issues)
    const EMOTION_MAP = {
        happy:    { emoji: '\u{1F60A}', label: 'Happy' },       // 😊
        sad:      { emoji: '\u{1F622}', label: 'Sad' },         // 😢
        angry:    { emoji: '\u{1F620}', label: 'Angry' },       // 😠
        fear:     { emoji: '\u{1F628}', label: 'Fear' },        // 😨
        surprise: { emoji: '\u{1F632}', label: 'Surprise' },    // 😲
        disgust:  { emoji: '\u{1F922}', label: 'Disgust' },     // 🤢
        neutral:  { emoji: '\u{1F610}', label: 'Neutral' }      // 😐
    };

    /**
     * Start the user webcam feed
     */
    async function startCamera() {
        resetUI();
        updateCameraStatus('starting', 'Starting Camera...');
        updateFeedback('Requesting camera permissions. Please allow access in your browser prompt.', 'info');

        try {
            // Request user-facing video track only (audio is strictly disabled)
            mediaStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'user',
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                },
                audio: false
            });

            video.srcObject = mediaStream;
            await video.play();

            // Update UI state for running camera
            videoPlaceholder.classList.add('hidden');
            startBtn.disabled = true;
            stopBtn.disabled = false;
            
            updateCameraStatus('active', 'Camera Active');
            updateFeedback('Camera live. Keep your face well-lit and facing the camera.', 'info');
            analysisStatus.textContent = 'Analyzing facial expressions periodically...';

            // Schedule recurring frame capture
            analysisIntervalId = setInterval(captureAndAnalyzeFrame, ANALYSIS_INTERVAL_MS);

        } catch (error) {
            console.error('Camera initialization failed:', error);
            handleCameraError(error);
        }
    }

    /**
     * Handle camera permission or device access errors
     */
    function handleCameraError(error) {
        stopCamera();
        
        let message = 'Could not access the webcam.';
        if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
            message = 'Camera permission was denied. Please allow camera access in your browser settings and try again.';
        } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
            message = 'No camera device found on this system. Please connect a webcam.';
        } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
            message = 'Camera is currently in use by another application. Please close it and retry.';
        }

        updateCameraStatus('stopped', 'Camera Error');
        updateFeedback(message, 'danger');
        analysisStatus.textContent = 'Camera initialization failed.';
    }

    /**
     * Capture one video frame, send it to the backend, and update the dashboard
     */
    async function captureAndAnalyzeFrame() {
        // Prevent concurrent requests if previous analysis is still ongoing
        if (isAnalyzing) {
            return;
        }

        // Validate that video is playing and dimensions are ready
        if (video.paused || video.ended || video.readyState < 2 || !video.videoWidth || !video.videoHeight) {
            return;
        }

        isAnalyzing = true;
        scanningIndicator.classList.remove('hidden');
        updateCameraStatus('analyzing', 'Analyzing...');

        const startTime = performance.now();

        try {
            // Draw current video frame onto hidden canvas
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            
            // Draw without horizontal flip so coordinates are standard orientation
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            // Convert canvas snapshot to JPEG Blob
            const imageBlob = await new Promise((resolve, reject) => {
                canvas.toBlob((blob) => {
                    if (blob) {
                        resolve(blob);
                    } else {
                        reject(new Error('Canvas frame encoding failed.'));
                    }
                }, 'image/jpeg', 0.85);
            });

            // Package into multipart/form-data payload
            const formData = new FormData();
            formData.append('frame', imageBlob, 'frame.jpg');

            // Send to Flask backend
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            const latency = Math.round(performance.now() - startTime);
            displayLatency(latency);

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || `Server returned error (${response.status})`);
            }

            // Handle detection outcome
            handleAnalysisResult(data);

        } catch (error) {
            console.error('Frame analysis error:', error);
            updateFeedback(error.message || 'Network error occurred during analysis.', 'danger');
            analysisStatus.textContent = 'Analysis connection interrupted.';
        } finally {
            isAnalyzing = false;
            scanningIndicator.classList.add('hidden');
            if (mediaStream && !startBtn.disabled) {
                // If camera was stopped during fetch
                updateCameraStatus('stopped', 'Camera Off');
            } else if (mediaStream) {
                updateCameraStatus('active', 'Camera Active');
            }
        }
    }

    /**
     * Process successful JSON response from /analyze
     */
    function handleAnalysisResult(data) {
        if (data.status === 'no_face' || (data.error && !data.scores)) {
            expressionEmoji.textContent = '\u{1F914}'; // 🤔
            dominantExpression.textContent = 'No Face Detected';
            dominantConfidence.textContent = 'Confidence: --';
            updateFeedback(data.message || data.error || 'No face visible. Please look directly into the camera.', 'warning');
            renderEmptyScores('Position your face in good lighting.');
            analysisStatus.textContent = 'Searching for face in frame...';
            return;
        }

        const dominantKey = (data.dominant_emotion || data.emotion || '').toLowerCase();
        const emotionMeta = EMOTION_MAP[dominantKey] || { emoji: '\u{1F642}', label: dominantKey || 'Unknown' };

        // Calculate confidence percentage of dominant emotion if scores present
        let dominantScoreText = '--';
        if (data.scores && dominantKey in data.scores) {
            dominantScoreText = `${data.scores[dominantKey].toFixed(1)}%`;
        }

        // Update Dominant Expression Box
        expressionEmoji.textContent = emotionMeta.emoji;
        dominantExpression.textContent = emotionMeta.label;
        dominantConfidence.textContent = `Confidence: ${dominantScoreText}`;

        updateFeedback(`Classified expression: ${emotionMeta.label}. Educational prediction based on facial geometry.`, 'success');
        analysisStatus.textContent = 'Analysis updated successfully.';

        // Update Scores visualization
        if (data.scores && Object.keys(data.scores).length > 0) {
            renderScores(data.scores, dominantKey);
        }
    }

    /**
     * Safely render score progress bars using DOM manipulation (no innerHTML)
     */
    function renderScores(scores, dominantKey) {
        scoresContainer.replaceChildren();

        // Sort descending by score for optimal visual clarity
        const entries = Object.entries(scores).sort((a, b) => b[1] - a[1]);

        for (const [category, score] of entries) {
            const numScore = typeof score === 'number' ? score : parseFloat(score) || 0;
            const clampedScore = Math.max(0, Math.min(100, numScore));
            const isDominant = category.toLowerCase() === dominantKey;

            const row = document.createElement('div');
            row.className = `score-row ${isDominant ? 'dominant' : ''}`;

            const labelRow = document.createElement('div');
            labelRow.className = 'score-label-row';

            const categorySpan = document.createElement('span');
            categorySpan.className = 'score-category';
            const meta = EMOTION_MAP[category.toLowerCase()];
            categorySpan.textContent = meta ? `${meta.emoji} ${meta.label}` : category;

            const valueSpan = document.createElement('span');
            valueSpan.className = 'score-value';
            valueSpan.textContent = `${clampedScore.toFixed(1)}%`;

            labelRow.append(categorySpan, valueSpan);

            const track = document.createElement('div');
            track.className = 'progress-track';

            const fill = document.createElement('div');
            fill.className = 'progress-fill';
            fill.style.width = `${clampedScore}%`;

            track.appendChild(fill);
            row.append(labelRow, track);
            scoresContainer.appendChild(row);
        }
    }

    /**
     * Render empty / placeholder state in scores container
     */
    function renderEmptyScores(message) {
        scoresContainer.replaceChildren();
        const placeholder = document.createElement('div');
        placeholder.className = 'scores-empty-state';
        const text = document.createElement('p');
        text.textContent = message;
        placeholder.appendChild(text);
        scoresContainer.appendChild(placeholder);
    }

    /**
     * Stop the camera and release media tracks
     */
    function stopCamera() {
        // Clear periodic analysis timer
        if (analysisIntervalId !== null) {
            clearInterval(analysisIntervalId);
            analysisIntervalId = null;
        }

        // Release hardware camera tracks
        if (mediaStream) {
            mediaStream.getTracks().forEach((track) => {
                track.stop();
            });
            mediaStream = null;
        }

        video.srcObject = null;
        isAnalyzing = false;

        // Reset UI buttons & elements
        videoPlaceholder.classList.remove('hidden');
        scanningIndicator.classList.add('hidden');
        inferenceLatency.classList.add('hidden');

        startBtn.disabled = false;
        stopBtn.disabled = true;

        updateCameraStatus('stopped', 'Camera Off');
        analysisStatus.textContent = 'Camera is stopped.';
        updateFeedback('Camera stopped. Press "Start Camera" to resume.', 'info');
        resetUI();
    }

    /**
     * Reset prediction results to initial default state
     */
    function resetUI() {
        expressionEmoji.textContent = '\u{1F610}'; // 😐
        dominantExpression.textContent = 'Awaiting Input...';
        dominantConfidence.textContent = 'Confidence: --%';
        renderEmptyScores('No analysis scores available yet.');
        inferenceLatency.classList.add('hidden');
    }

    /**
     * Update camera status badge
     */
    function updateCameraStatus(stateClass, label) {
        cameraStatusBadge.className = `status-badge status-${stateClass}`;
        cameraStatusText.textContent = label;
    }

    /**
     * Update dynamic feedback alert banner
     */
    function updateFeedback(message, type) {
        feedbackMessage.className = `feedback-banner ${type}`;
        feedbackMessage.textContent = message;
    }

    /**
     * Display round-trip inference latency
     */
    function displayLatency(ms) {
        inferenceLatency.textContent = `${ms} ms`;
        inferenceLatency.classList.remove('hidden');
    }

    // Attach Event Listeners
    startBtn.addEventListener('click', startCamera);
    stopBtn.addEventListener('click', stopCamera);

    // Initial state setup on page load
    resetUI();

    // Clean up media tracks if the window/tab is closed
    window.addEventListener('beforeunload', () => {
        if (mediaStream) {
            mediaStream.getTracks().forEach(t => t.stop());
        }
    });
})();
