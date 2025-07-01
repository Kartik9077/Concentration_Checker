🎯 Concentration Tracker

A real-time attention monitoring system using MediaPipe and OpenCV. This tool evaluates user concentration based on eye blinks, gaze direction, and head pose, making it ideal for study monitoring, remote learning, and productivity applications.

🔍 Features

👁️ Eye Blink Detection
Uses Eye Aspect Ratio (EAR) to detect blinks and prolonged eye closures.
👀 Gaze Detection
Determines if the user is looking straight or away using iris landmarks.
🧠 Head Pose Estimation
Estimates head orientation based on the nose position relative to the screen center.
📊 Concentration Score
Calculates a weighted attention score:
Score = 0.4 * Gaze + 0.4 * Head Pose + 0.2 * (Not Blinking)
🔴 Live Visual Feedback
Real-time webcam overlay with:
Concentration % bar
Blink alerts
Distraction counter
ACTIVE / DISTRACTED indicator
FPS display
⚠️ Distraction Tracking
Counts frames with lost focus and issues warnings on repeated distractions.
🧪 Sample Output

The output video stream includes:

📉 A real-time concentration meter
⚠️ Blink and distraction alerts
🔵 Status: ACTIVE or DISTRACTED
⏱️ FPS Counter for performance feedback
🛠️ Tech Stack

Python 3.x
OpenCV
MediaPipe (FaceMesh)
NumPy
⚙️ How It Works

Facial landmarks are detected using MediaPipe FaceMesh.
EAR (Eye Aspect Ratio) tracks blinks.
Iris landmarks detect gaze direction.
Nose position estimates head pose.
A composite score is computed to reflect real-time focus level.
Visual overlays provide instant feedback to the user.
