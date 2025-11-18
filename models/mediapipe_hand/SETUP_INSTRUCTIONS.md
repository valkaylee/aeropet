# Setup Instructions for Hand Gesture Recognition

This guide will help you set up the hand gesture recognition system after cloning the repository.

## Prerequisites

- **Python 3.10, 3.11, or 3.12** (Python 3.11 recommended for best TensorFlow compatibility)
- **pip** (Python package manager)

## Installation

### macOS (Recommended - Prevents TensorFlow Hanging)

```bash
# Fix protobuf version first (prevents TensorFlow hanging on macOS)
pip install protobuf==5.28.3

# Install TensorFlow
pip install tensorflow

# Install other dependencies
pip install qai-hub-models numpy pillow opencv-python
```

### Linux/Windows

```bash
# Install TensorFlow
pip install tensorflow

# Install other dependencies
pip install qai-hub-models numpy pillow opencv-python
```

## Verify Installation

Test that everything works:

```bash
cd models/mediapipe_hand
python3 test_tensorflow.py
```

You should see all tests passing. If you see errors, check the troubleshooting section below.

## Run the Gesture Demo

### With Camera

```bash
python3 models/mediapipe_hand/gesture_demo.py --camera 0
```

**Options:**
- `--camera <number>`: Camera device ID (default: 0)
- `--skip-gesture-classifier`: Skip gesture classification if TensorFlow has issues
- `--score-threshold <float>`: Detection threshold (default: 0.95, lower = more detections)

### Without Gesture Classification

If TensorFlow is causing issues, you can still use hand detection:

```bash
python3 models/mediapipe_hand/gesture_demo.py --camera 0 --skip-gesture-classifier
```

## Troubleshooting

### TensorFlow Installation Issues

**Problem:** `ModuleNotFoundError: No module named 'tensorflow'`

**Solution:**
```bash
pip install tensorflow
```

**Problem:** TensorFlow hangs or crashes on macOS

**Solution:**
```bash
# Uninstall existing versions
pip uninstall protobuf tensorflow tensorflow-metal -y

# Install compatible versions
pip install protobuf==5.28.3
pip install tensorflow
```

**Alternative:** Use Python 3.11 instead of 3.12:
```bash
# Install Python 3.11 via Homebrew
brew install python@3.11
python3.11 -m pip install protobuf==5.28.3 tensorflow qai-hub-models numpy pillow opencv-python
```

### Camera Not Opening

**Solutions:**
1. Check camera permissions (macOS: System Settings > Privacy & Security > Camera)
2. Try a different camera index: `--camera 1`
3. Check if another application is using the camera

### Poor Gesture Recognition

**Solutions:**
1. Ensure good lighting
2. Keep hand fully visible in frame
3. Lower the score threshold: `--score-threshold 0.5`
4. Make clear, distinct gestures (Open, Close, Pointer, OK)

## Available Gestures

The classifier recognizes:
- **Open**: Open hand (all fingers extended)
- **Close**: Closed fist
- **Pointer**: Pointing gesture (index finger extended)
- **OK**: OK sign (thumb and index finger forming a circle)

## Project Structure

```
aeropet/
├── models/
│   └── mediapipe_hand/
│       ├── gesture_demo.py          # Main demo script
│       ├── model/
│       │   └── keypoint_classifier/
│       │       ├── keypoint_classifier.tflite    # Gesture model
│       │       ├── keypoint_classifier_label.csv # Gesture labels
│       │       ├── keypoint_classifier.py         # Model loader
│       │       └── preprocess.py                  # Preprocessing utilities
│       └── SETUP_INSTRUCTIONS.md    # This file
└── README.md
```

## Additional Resources

- [Qualcomm AI Hub MediaPipe Hand](https://aihub.qualcomm.com/models/mediapipe_hand)
- [Kinivi Hand Gesture Recognition](https://github.com/kinivi/hand-gesture-recognition-mediapipe)
