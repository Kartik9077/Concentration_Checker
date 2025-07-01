# Concentration Tracker

A real-time concentration tracking system built using **MediaPipe** and **OpenCV**. This tool evaluates a user's attentiveness based on eye blinks, gaze direction, and head pose — ideal for applications like study monitoring, e-learning, or productivity enhancement.

## Features

- **Eye Blink Detection**  
  Calculates Eye Aspect Ratio (EAR) to detect blinks and periods of eye closure.

- **Gaze Detection**  
  Estimates if the user is looking straight or away using iris landmarks.

- **Head Pose Estimation**  
  Evaluates user orientation based on nose position relative to screen center.

- **Concentration Score**  
  Computes a weighted score combining gaze, head pose, and blinking behavior.

- **Live Visual Feedback**  
  Real-time UI overlay on webcam feed showing concentration level, blink status, and distraction counter.

- **Distraction Tracking**  
  Counts how many frames the user is not paying attention and issues warnings if needed.

## Sample Output

The video feed displays:
- A concentration percentage bar
- Blink detection alerts
- Distraction count
- ACTIVE / DISTRACTED indicator
- FPS counter

## Tech Stack

- Python 3.x
- OpenCV
- MediaPipe (FaceMesh)
- NumPy

## How It Works

1. **Face landmarks** are detected using MediaPipe FaceMesh.
2. **EAR (Eye Aspect Ratio)** is used to detect blinks.
3. **Iris position** is used to assess gaze direction.
4. **Nose position** is used to infer head pose.
5. A **composite concentration score** is calculated as: score = 0.4 * gaze + 0.4 * head_pose + 0.2 * (not blinking)
6. A **visual feedback system** shows user concentration in real time.

## Run the Project

```bash
git clone https://github.com/asutoshp10/concentration_tracker.git
cd concentration_tracker
pip install -r requirements.txt
python concentration_tracker.py
