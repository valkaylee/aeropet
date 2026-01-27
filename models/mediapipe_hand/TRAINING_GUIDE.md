# Hand Gesture Model Training Guide

This guide explains how to collect training data and train a custom gesture classifier model, compatible with kinivi's workflow.

## Overview

The training workflow consists of two steps:
1. **Data Collection**: Collect hand gesture samples using your webcam
2. **Model Training**: Train a TensorFlow Lite model from the collected data

## Prerequisites

Install required dependencies:
```bash
pip install tensorflow scikit-learn opencv-python numpy
```

## Step 1: Collect Training Data

Run the data collection script:

```bash
python models/mediapipe_hand/collect_training_data.py --camera 0
```

### Controls:
- **Press 'k'**: Enter/exit logging mode
- **Press '0'-'9'**: Label the current gesture (only works in logging mode)
- **Press 'q'**: Quit and save data

### How to Collect Data:

1. **Prepare your gesture labels**: 
   - Edit `model/keypoint_classifier/keypoint_classifier_label.csv`
   - Add one gesture name per line (e.g., Open, Close, Pointer, OK, ThumbsUp, etc.)
   - The line number corresponds to the label ID (0, 1, 2, ...)

2. **Start the collection script**:
   ```bash
   python models/mediapipe_hand/collect_training_data.py --camera 0
   ```

3. **Collect samples**:
   - Press 'k' to enter logging mode
   - Make the gesture you want to record
   - Press the number key (0-9) corresponding to that gesture
   - Repeat for multiple samples of each gesture
   - Press 'k' again to exit logging mode when done

4. **Tips for good data**:
   - Collect at least 50-100 samples per gesture
   - Vary hand position, rotation, and distance from camera
   - Collect samples in different lighting conditions
   - Make sure the hand is fully visible in frame

### Output:
- Data is saved to `model/keypoint_classifier/keypoint.csv`
- Format: `label, x0, y0, x1, y1, ..., x20, y20` (42 features per sample)
- The CSV file is appended to, so you can collect data in multiple sessions

## Step 2: Train the Model

Once you have collected sufficient data, train the model:

```bash
python models/mediapipe_hand/train_gesture_classifier.py
```

### Training Options:

```bash
# Basic training (uses defaults)
python models/mediapipe_hand/train_gesture_classifier.py

# Custom training parameters
python models/mediapipe_hand/train_gesture_classifier.py \
    --epochs 200 \
    --batch-size 64 \
    --test-size 0.2

# Custom data/output paths
python models/mediapipe_hand/train_gesture_classifier.py \
    --data path/to/keypoint.csv \
    --output path/to/keypoint_classifier.tflite
```

### Training Parameters:
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size for training (default: 32)
- `--test-size`: Fraction of data for validation (default: 0.2)
- `--data`: Path to training CSV file
- `--output`: Path to save trained TFLite model

### Output:
- Trained model saved to `model/keypoint_classifier/keypoint_classifier.tflite`
- Training metrics (accuracy, loss, classification report)

## Step 3: Test the Trained Model

Test your trained model using the gesture demo:

```bash
python models/mediapipe_hand/gesture_demo.py --camera 0
```

The demo will automatically load your trained model from `model/keypoint_classifier/keypoint_classifier.tflite`.

## Compatibility with Kinivi's Workflow

✅ **Fully Compatible**: This training workflow is compatible with kinivi's approach:

1. **Data Format**: 
   - Uses the same CSV format (label + 42 preprocessed features)
   - Preprocessing matches kinivi's implementation (relative coordinates, normalized)

2. **Model Format**:
   - Outputs TensorFlow Lite (.tflite) format
   - Compatible with the existing `KeyPointClassifier` class
   - Same input/output format (42 features → class probabilities)

3. **Training Process**:
   - Uses similar MLP architecture (Multi-Layer Perceptron)
   - Same preprocessing pipeline
   - Can be used with kinivi's training notebook if preferred

### Using Kinivi's Training Notebook (Alternative)

If you prefer to use kinivi's Jupyter notebook:

1. **Collect data** using `collect_training_data.py` (creates `keypoint.csv`)
2. **Download kinivi's notebook**: `keypoint_classification.ipynb` from [kinivi's repository](https://github.com/kinivi/hand-gesture-recognition-mediapipe)
3. **Set `NUM_CLASSES`** to match your number of gestures
4. **Run the notebook** - it will work with your collected CSV data

The data format is identical, so the notebook will work seamlessly.

## Troubleshooting

### Not enough training data
- **Error**: "Some classes have very few samples"
- **Solution**: Collect more samples (aim for 50-100+ per gesture)

### Poor model accuracy
- **Solutions**:
  - Collect more diverse samples (different angles, distances, lighting)
  - Increase training epochs
  - Check that gestures are distinct and clearly visible
  - Ensure balanced class distribution

### Model not loading
- **Check**: Model file exists at `model/keypoint_classifier/keypoint_classifier.tflite`
- **Check**: TensorFlow/TFLite is installed correctly
- **Verify**: Model was trained successfully (check training output)

### Camera not opening
- **Check**: Camera permissions (macOS: System Settings > Privacy & Security > Camera)
- **Try**: Different camera index (`--camera 1`)
- **Check**: No other application is using the camera

## File Structure

```
models/mediapipe_hand/
├── collect_training_data.py          # Data collection script
├── train_gesture_classifier.py       # Training script
├── gesture_demo.py                   # Demo with trained model
└── model/
    └── keypoint_classifier/
        ├── keypoint.csv               # Training data (collected)
        ├── keypoint_classifier.tflite # Trained model (output)
        └── keypoint_classifier_label.csv # Gesture labels
```

## Example Workflow

```bash
# 1. Collect training data
python models/mediapipe_hand/collect_training_data.py --camera 0
# Press 'k', make gesture, press '0', repeat for all gestures...

# 2. Train the model
python models/mediapipe_hand/train_gesture_classifier.py --epochs 150

# 3. Test the model
python models/mediapipe_hand/gesture_demo.py --camera 0
```

## References

- [Kinivi's Hand Gesture Recognition Repository](https://github.com/kinivi/hand-gesture-recognition-mediapipe)
- [Qualcomm AI Hub MediaPipe Hand](https://aihub.qualcomm.com/models/mediapipe_hand)

