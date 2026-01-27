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