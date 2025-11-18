# Gesture Classification Integration Guide

This document explains how kinivi's hand gesture recognition has been integrated into the Qualcomm AI Hub MediaPipe Hand module.

## Overview

The integration combines:
- **Qualcomm AI Hub's MediaPipe Hand**: For hand landmark detection (compatible with DJI Tello drone)
- **Kinivi's Gesture Classifier**: For classifying hand gestures using a pre-trained MLP model

## Files Added/Modified

### New Files Created

1. **`model/keypoint_classifier/keypoint_classifier.py`**
   - TFLite model loader and inference module
   - Adapted from kinivi's repository to work with Qualcomm AI Hub

2. **`model/keypoint_classifier/preprocess.py`**
   - Preprocessing utilities to convert Qualcomm landmarks to kinivi's format
   - Normalizes landmarks relative to wrist and scales them

3. **`model/keypoint_classifier/keypoint_classifier.tflite`**
   - Pre-trained gesture classification model (downloaded from kinivi repo)
   - Classifies gestures: Open, Close, Pointer, OK

4. **`model/keypoint_classifier/keypoint_classifier_label.csv`**
   - Gesture labels mapping class IDs to gesture names

5. **`model/__init__.py`** and **`model/keypoint_classifier/__init__.py`**
   - Python package initialization files

### Modified Files

1. **`gesture_demo.py`**
   - Updated to use kinivi's gesture classifier instead of simple rule-based classification
   - Automatically loads the TFLite model and labels on startup
   - Classifies gestures: Open, Close, Pointer, OK (instead of just STOP/UP)

## How It Works

1. **Landmark Detection**: Qualcomm AI Hub's MediaPipe Hand detects 21 hand landmarks
2. **Preprocessing**: Landmarks are normalized relative to the wrist and scaled
3. **Classification**: Preprocessed landmarks are fed to the TFLite model
4. **Output**: Gesture label is returned (Open, Close, Pointer, OK, or UNKNOWN)

## Usage

Run the gesture demo as before:

```bash
python models/mediapipe_hand/gesture_demo.py --camera 0
```

Or with an image:

```bash
python models/mediapipe_hand/gesture_demo.py --image path/to/image.jpg
```

The demo will:
- Load the Qualcomm AI Hub MediaPipe Hand model
- Load the kinivi gesture classifier
- Detect hands and classify gestures in real-time
- Print detected gestures to the console

## Gesture Labels

The classifier recognizes the following gestures (from `keypoint_classifier_label.csv`):
- **Open**: Open hand
- **Close**: Closed fist
- **Pointer**: Pointing gesture (index finger extended)
- **OK**: OK sign (thumb and index finger forming a circle)

## Preprocessing Details

The preprocessing function (`preprocess_landmark`) performs:
1. **Relative Coordinates**: Converts all landmarks to be relative to the wrist (landmark 0)
2. **Normalization**: Scales coordinates by the maximum absolute value to normalize to [-1, 1] range
3. **Flattening**: Converts the 21x3 landmark array to a 63-element 1D array

This preprocessing ensures the classifier works regardless of:
- Hand size
- Distance from camera
- Image resolution

## Compatibility

- ✅ Maintains full compatibility with Qualcomm AI Hub's MediaPipe Hand
- ✅ Works with DJI Tello drone (uses same MediaPipe implementation)
- ✅ Uses TensorFlow Lite for efficient inference
- ✅ Gracefully handles missing model files (falls back to UNKNOWN)

## Troubleshooting

### Model Not Found
If you see "Warning: Gesture classifier model not found", ensure:
- `keypoint_classifier.tflite` exists in `model/keypoint_classifier/`
- The file path is correct relative to `gesture_demo.py`

### Import Errors
If you encounter import errors:
- Ensure all `__init__.py` files are present
- **Install TensorFlow** (required for gesture classification):
  ```bash
  pip install tensorflow
  ```
  Or use the lighter alternative:
  ```bash
  pip install tflite-runtime
  ```

### Poor Classification Accuracy
If gesture classification is inaccurate:
- Ensure good lighting and clear hand visibility
- Make sure the hand is fully visible in the frame
- The preprocessing may need adjustment for your specific use case

## Future Enhancements

Potential improvements:
1. Add point history classifier for dynamic gestures (swipes, circles, etc.)
2. Add confidence scores for gesture predictions
3. Add gesture filtering/smoothing to reduce false positives
4. Custom training data collection for drone-specific gestures

## References

- [Kinivi's Hand Gesture Recognition Repository](https://github.com/kinivi/hand-gesture-recognition-mediapipe)
- [Qualcomm AI Hub MediaPipe Hand](https://aihub.qualcomm.com/models/mediapipe_hand)

