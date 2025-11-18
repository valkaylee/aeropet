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