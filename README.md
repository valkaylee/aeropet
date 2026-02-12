# aeropet

Hand gesture recognition system using Qualcomm AI Hub's MediaPipe Hand with kinivi's gesture classification.

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd aeropet
   ```

2. **Install Python dependencies (see [SETUP_INSTRUCTIONS.md](models/mediapipe_hand/SETUP_INSTRUCTIONS.md) for macOS compatibility fixes):**
   ```bash
   # macOS (recommended)
   pip install protobuf==5.28.3 tensorflow qai-hub-models numpy pillow opencv-python
   
   # Linux/Windows
   pip install tensorflow qai-hub-models numpy pillow opencv-python
   ```

3. **Run the gesture demo:**
   ```bash
   python3 gesture_demo.py --camera 0
   ```

## Documentation

- **[Setup Instructions](models/mediapipe_hand/SETUP_INSTRUCTIONS.md)** - Complete setup guide for new users
- **[Gesture Classification Integration](models/mediapipe_hand/GESTURE_CLASSIFICATION_INTEGRATION.md)** - Technical details about the integration
- **[TensorFlow Troubleshooting](models/mediapipe_hand/TENSORFLOW_FIX.md)** - Solutions for TensorFlow issues

## Features

- ✅ Real-time hand detection using Qualcomm AI Hub's MediaPipe Hand
- ✅ Gesture classification (Open, Close, Pointer, OK) using kinivi's pre-trained model
- ✅ Compatible with DJI Tello drone (uses same MediaPipe implementation)
- ✅ Works with webcam or image input

## Requirements

- Python 3.10-3.13
- qai-hub-models
- TensorFlow or tflite-runtime
- OpenCV
- NumPy, Pillow

See [SETUP_INSTRUCTIONS.md](models/mediapipe_hand/SETUP_INSTRUCTIONS.md) for detailed requirements and installation steps.


## start commands

1. Add your new gesture labels to
    models/mediapipe_hand/model/keypoint_classifier/keypoint_classifier_label.csv:
    
2. Collect training data:
    python3 models/mediapipe_hand/collect_training_data.py --camera 0
    
- Press k to enter logging mode
- Make your gesture, press the number key (e.g., 4 for ThumbsUp)
- Collect 50-100 samples per gesture, varying angle/distance
- Press q when done
1. Train the model:
    
    python3 models/mediapipe_hand/train_gesture_classifier.py --epochs 150
    
    This overwrites keypoint_classifier.tflite with your new model.
    
2. Test it:
    python3 models/mediapipe_hand/gesture_demo.py --camera 0
    
Key Detail

The collection script uses the same preprocess_landmark() function that's used during inference, so the
training data format exactly matches what the model sees at runtime. This is the critical part that

makes it compatible with kinivi's workflow.


## Project Overview

Aeropet is a hand gesture recognition system that controls DJI Tello drones. It combines:
- **Qualcomm AI Hub's MediaPipe Hand** for 21-point hand landmark detection
- **Kinivi's gesture classifier** (TFLite MLP model) for classifying gestures: Open, Close, Pointer, OK

## Commands

### Run the gesture demo (webcam)
```bash
python3 models/mediapipe_hand/gesture_demo.py --camera 0
```
Options: `--score-threshold 0.5` (lower = more detections), `--skip-gesture-classifier`

### Run Tello drone control
```bash
python3 test_gestures.py
```
Requires connection to Tello WiFi network.

### Test TensorFlow/TFLite installation
```bash
python3 models/mediapipe_hand/test_tensorflow.py
```

### Install dependencies
```bash
# macOS (protobuf version matters)
pip install protobuf==5.28.3 tensorflow qai-hub-models numpy pillow opencv-python

# Linux/Windows
pip install tensorflow qai-hub-models numpy pillow opencv-python
```

## Architecture

### Pipeline Flow
1. **MediaPipeHandApp** (`models/mediapipe_hand/app.py`) detects hands and returns 21 landmarks (x, y, z) per hand
2. **preprocess_landmark** (`models/mediapipe_hand/model/keypoint_classifier/preprocess.py`) converts landmarks to relative coordinates (wrist-centered), normalizes to [-1,1], and flattens to 42 features
3. **KeyPointClassifier** (`models/mediapipe_hand/model/keypoint_classifier/keypoint_classifier.py`) runs TFLite inference to classify gesture

### Key Data Structures
- Raw landmarks from MediaPipe: `np.ndarray` shape `(21, 3)` with (x, y, z) pixel coordinates
- Preprocessed landmarks: `np.ndarray` shape `(42,)` - normalized relative x,y coordinates
- `app.predict_landmarks_from_image(frame, raw_output=True)` returns 5 values: boxes, keypoints, roi_4corners, **landmarks (index 3)**, is_right_hand

### Gesture Labels
Located in `models/mediapipe_hand/model/keypoint_classifier/keypoint_classifier_label.csv`:
- 0: Open (open hand)
- 1: Close (fist)
- 2: Pointer (index finger extended)
- 3: OK (thumb+index circle)

## Known Issues

### macOS TensorFlow Hanging
If TensorFlow hangs during model loading, the protobuf version conflict is usually the cause:
```bash
pip uninstall protobuf tensorflow tensorflow-metal -y
pip install protobuf==5.28.3 tensorflow
```

### Landmark Shape Debugging
If gesture classification fails, check that `landmarks.shape == (21, 3)`. The raw_output from MediaPipe returns landmarks at index 3, not index 2.
