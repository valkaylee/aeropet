# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import argparse
import os
from typing import cast

import numpy as np
from PIL import Image

from qai_hub_models.models.mediapipe_hand.app import MediaPipeHandApp
from qai_hub_models.models.mediapipe_hand.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    MediaPipeHand,
)
from qai_hub_models.utils.args import add_output_dir_arg
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.camera_capture import capture_and_display_processed_frames
from qai_hub_models.utils.display import display_or_save_image

# Import kinivi gesture classifier (lazy import to avoid TensorFlow loading issues)
# These will be imported only when needed, not at module load time

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "hand.jpeg"
)

# Global classifier and labels (initialized in main)
keypoint_classifier = None
gesture_labels = []


def classify_gesture(landmarks: np.ndarray) -> str:
    """
    Classify hand gesture from landmarks using kinivi's gesture classifier.
    
    Args:
        landmarks: numpy array of shape (21, 3) with (x, y, z) coordinates
                   from Qualcomm AI Hub MediaPipe Hand
        
    Returns:
        Gesture label string (e.g., "Open", "Close", "Pointer", "OK") or "UNKNOWN"
    """
    global keypoint_classifier, gesture_labels
    
    if landmarks.shape[0] < 21:
        return "UNKNOWN"
    
    if keypoint_classifier is None:
        return "UNKNOWN"
    
    try:
        # Lazy import to avoid TensorFlow loading at module import time
        from model.keypoint_classifier.preprocess import preprocess_landmark
        
        # Validate landmarks shape
        if landmarks.shape != (21, 3):
            print(f"    ⚠️  Unexpected landmark shape: {landmarks.shape}, expected (21, 3)")
            return "UNKNOWN"
        
        # Preprocess landmarks for the classifier
        preprocessed_landmarks = preprocess_landmark(landmarks)
        
        # Validate preprocessed landmarks (model expects 42 features: 21 landmarks * 2 coords)
        if preprocessed_landmarks.shape[0] != 42:
            print(f"    ⚠️  Unexpected preprocessed shape: {preprocessed_landmarks.shape}, expected (42,)")
            return "UNKNOWN"
        
        # Classify gesture
        gesture_id = keypoint_classifier(preprocessed_landmarks)
        
        # Debug: Print the raw gesture ID
        print(f"    → Raw gesture_id: {gesture_id}, labels available: {len(gesture_labels)}")
        
        # Get gesture label
        if 0 <= gesture_id < len(gesture_labels):
            label = gesture_labels[gesture_id]
            print(f"    → Mapped to label: '{label}'")
            return label
        else:
            print(f"    ⚠️  Invalid gesture_id {gesture_id}, expected 0-{len(gesture_labels)-1}")
            return "UNKNOWN"
    except Exception as e:
        print(f"    ✗ Error classifying gesture: {e}")
        import traceback
        traceback.print_exc()
        return "UNKNOWN"


# Run Mediapipe Hand landmark detection with gesture recognition
def main(is_test: bool = False):
    # Demo parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        required=False,
        help="image file path or URL",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera Input ID",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.95,  # Lowered from 0.95 to allow more landmarks through
        help="Score threshold for NonMaximumSuppression (lower = more detections)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="Intersection over Union (IoU) threshold for NonMaximumSuppression",
    )
    parser.add_argument(
        "--skip-gesture-classifier",
        action="store_true",
        help="Skip loading gesture classifier (useful if TensorFlow is causing issues)",
    )
    add_output_dir_arg(parser)

    print(
        "Note: This demo is running through torch, and not meant to be real-time without dedicated ML hardware."
    )
    print("Use Ctrl+C in your terminal to exit.")

    print("Parsing arguments...")
    args = parser.parse_args([] if is_test else None)
    print(f"Arguments parsed. skip_gesture_classifier={args.skip_gesture_classifier}")
    if is_test:
        args.image = INPUT_IMAGE_ADDRESS

    # Load app
    print("Loading MediaPipe Hand model...")
    print("  Creating MediaPipeHand model...")
    model = MediaPipeHand.from_pretrained()
    print("  MediaPipeHand model created")
    print("  Creating MediaPipeHandApp...")
    app = MediaPipeHandApp.from_pretrained(model)
    # Lower the score threshold to get more landmark detections
    app.min_detector_box_score = args.score_threshold
    print(f"  MediaPipeHandApp created (score threshold: {app.min_detector_box_score})")
    print("✓ Model and App Loaded")
    
    # Initialize gesture classifier (non-blocking - camera will work even if this fails)
    global keypoint_classifier, gesture_labels
    keypoint_classifier = None
    gesture_labels = []
    
    if args.skip_gesture_classifier:
        print("Skipping gesture classifier (--skip-gesture-classifier flag set)")
    else:
        print("Attempting to load gesture classifier (optional)...")
        try:
            # Lazy import to avoid TensorFlow loading at module import time
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
            from model.keypoint_classifier.preprocess import load_gesture_labels
            
            # Get the path to the model directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(current_dir, 'model', 'keypoint_classifier')
            model_path = os.path.join(model_dir, 'keypoint_classifier.tflite')
            label_path = os.path.join(model_dir, 'keypoint_classifier_label.csv')
            
            print(f"  Checking for model at: {model_path}")
            if os.path.exists(model_path):
                print(f"  Model file found ({os.path.getsize(model_path)} bytes)")
                print("  Initializing TFLite interpreter...")
                keypoint_classifier = KeyPointClassifier(model_path=model_path)
                print("  TFLite interpreter initialized")
                print("  Loading labels...")
                gesture_labels = load_gesture_labels(label_path)
                print(f"✓ Gesture Classifier Loaded. Available gestures: {', '.join(gesture_labels)}")
            else:
                print(f"  Model file not found at {model_path}")
                print("  Gesture classification will be disabled.")
        except ImportError as e:
            print(f"  TensorFlow/tflite-runtime not installed: {e}")
            print("  Install with: pip install tensorflow (or pip install tflite-runtime)")
            print("  Or use --skip-gesture-classifier flag to skip gesture classification")
            print("  Gesture classification will be disabled.")
        except Exception as e:
            print(f"  Warning: Failed to load gesture classifier: {e}")
            print("  This might be due to TensorFlow/tensorflow-metal conflicts on macOS")
            print("  Try: --skip-gesture-classifier flag to skip gesture classification")
            import traceback
            traceback.print_exc()
            print("  Gesture classification will be disabled.")
    
    print("Initialization complete. Starting camera...")

    if args.image:
        image = load_image(args.image)
        # Get raw landmarks for gesture classification
        raw_result = app.predict_landmarks_from_image(image, raw_output=True)
        # Extract the values we need - there are 5 return values!
        batched_selected_boxes = raw_result[0]
        batched_selected_keypoints = raw_result[1]
        batched_roi_4corners = raw_result[2]
        batched_selected_landmarks = raw_result[3]  # Correct index!
        batched_is_right_hand = raw_result[4]  # Correct index!
        
        # Classify gestures
        print(f"Processing {len(batched_selected_landmarks)} batches of landmarks...")
        for batch_idx, landmarks in enumerate(batched_selected_landmarks):
            if landmarks.nelement() != 0:
                landmarks_np = landmarks.cpu().numpy()
                print(f"Batch {batch_idx}: Found {landmarks_np.shape[0]} hand(s)")
                for hand_idx in range(landmarks_np.shape[0]):
                    try:
                        print(f"  Classifying hand {hand_idx}...")
                        gesture = classify_gesture(landmarks_np[hand_idx])
                        print(f"  Result: {gesture}")
                        if gesture != "UNKNOWN":
                            print(f"Gesture detected: {gesture}")
                    except Exception as e:
                        print(f"Error classifying gesture: {e}")
                        import traceback
                        traceback.print_exc()
            else:
                print(f"Batch {batch_idx}: No landmarks detected")
        

        pred_image = app.predict_landmarks_from_image(image)
        assert isinstance(pred_image[0], np.ndarray)
        out_image = Image.fromarray(pred_image[0], "RGB")
        if not is_test:
            display_or_save_image(out_image, args.output_dir)
    else:
        last_gesture = None

        frame_count = 0
        
        def frame_processor(frame: np.ndarray) -> np.ndarray:
            nonlocal last_gesture, frame_count
            frame_count += 1
            
            # Get raw landmarks for gesture classification
            raw_result = app.predict_landmarks_from_image(frame, raw_output=True)
            # Extract the values we need - there are 5 return values!
            # Based on MediaPipeApp structure: boxes, keypoints, roi_4corners, landmarks, is_right_hand
            batched_selected_boxes = raw_result[0]
            batched_selected_keypoints = raw_result[1]
            batched_roi_4corners = raw_result[2]  # This was missing!
            batched_selected_landmarks = raw_result[3]  # Was accessing [2], should be [3]!
            batched_is_right_hand = raw_result[4]  # Was accessing [3], should be [4]!
            
            # CRITICAL FIX: The batched_selected_landmarks is returning (4, 2) instead of (21, 3)
            # This suggests the landmark detector output is being filtered or not accessed correctly.
            # The issue might be that landmarks are filtered by score threshold, or we need to
            # access the landmark detector output directly. Let's try a workaround by calling
            # the landmark detector directly if the landmarks are wrong shape.
            
            # Debug: Check all returned structures on first frame only
            if frame_count == 1:
                print(f"DEBUG: raw_result length: {len(raw_result)}")
                print(f"DEBUG: Return value structure:")
                print(f"  [0] = batched_selected_boxes")
                print(f"  [1] = batched_selected_keypoints") 
                print(f"  [2] = batched_roi_4corners")
                print(f"  [3] = batched_selected_landmarks (THIS IS WHAT WE NEED!)")
                print(f"  [4] = batched_is_right_hand")
                for i, item in enumerate(raw_result):
                    if isinstance(item, list) and len(item) > 0:
                        if hasattr(item[0], 'nelement'):
                            if item[0].nelement() != 0:
                                shape = item[0].shape
                                print(f"DEBUG: raw_result[{i}][0] shape: {shape}")
                            else:
                                print(f"DEBUG: raw_result[{i}][0] is empty")
                    else:
                        print(f"DEBUG: raw_result[{i}] type: {type(item)}")
                
                # Check the actual landmarks (index 3)
                if len(batched_selected_landmarks) > 0 and batched_selected_landmarks[0].nelement() != 0:
                    lm_shape = batched_selected_landmarks[0].shape
                    print(f"DEBUG: ✓ Landmarks found! Shape: {lm_shape}")
                    if len(lm_shape) == 3 and lm_shape[1] == 21 and lm_shape[2] == 3:
                        print(f"DEBUG: ✓✓✓ CORRECT! This is (num_hands, 21, 3)")
                    else:
                        print(f"DEBUG: ⚠️  Wrong shape - expected (num_hands, 21, 3)")
            
            # Classify gestures on every frame (adjust if too slow)
            classify_this_frame = True  # Classify every frame
            
            # Count hands detected
            hands_detected = 0
            for landmarks_tensor in batched_selected_landmarks:
                if landmarks_tensor.nelement() != 0:
                    landmarks_np = landmarks_tensor.cpu().numpy()
                    hands_detected += landmarks_np.shape[0]
            
            classifier_status = 'Loaded' if keypoint_classifier else 'Not loaded'
            print(f"[Frame {frame_count}] Hands: {hands_detected}, Classifier: {classifier_status}")
            
            # Classify gestures
            current_gesture = None
            if classify_this_frame and hands_detected > 0:
                for batch_idx, landmarks_tensor in enumerate(batched_selected_landmarks):
                    if landmarks_tensor.nelement() != 0:
                        landmarks_np = landmarks_tensor.cpu().numpy()
                        # landmarks_np shape should be (num_hands, 21, 3)
                        print(f"  DEBUG: Batch {batch_idx} landmarks shape: {landmarks_np.shape}")
                        
                        for hand_idx in range(landmarks_np.shape[0]):
                            hand_landmarks = landmarks_np[hand_idx]
                            print(f"  DEBUG: Hand {hand_idx} landmarks shape: {hand_landmarks.shape}")
                            
                            # Check if we have the right shape
                            if hand_landmarks.shape == (21, 3):
                                try:
                                    gesture = classify_gesture(hand_landmarks)
                                    print(f"  → Classified as: '{gesture}'")
                                    
                                    if gesture != "UNKNOWN":
                                        current_gesture = gesture
                                except Exception as e:
                                    print(f"  ✗ Classification error: {e}")
                                    import traceback
                                    traceback.print_exc()
                            else:
                                print(f"  ⚠️  Skipping hand {hand_idx}: wrong shape {hand_landmarks.shape}, expected (21, 3)")
                                # Try to see what we actually got
                                if len(hand_landmarks.shape) == 2:
                                    print(f"     This looks like {hand_landmarks.shape[0]} points with {hand_landmarks.shape[1]} coordinates")
            
            # Print when gesture changes
            if current_gesture and current_gesture != last_gesture:
                print(f"✓ Gesture changed to: {current_gesture}")
                last_gesture = current_gesture
            elif not current_gesture and last_gesture:
                print(f"✗ Gesture lost (was: {last_gesture})")
                last_gesture = None
            elif current_gesture and current_gesture == last_gesture:
                # Print every few frames to show gesture is stable
                if frame_count % 4 == 0:
                    print(f"  → Gesture: {current_gesture} (stable)")
            
            return cast(np.ndarray, app.predict_landmarks_from_image(frame)[0])

        print(f"Initializing camera (index {args.camera})...")
        print("  Calling capture_and_display_processed_frames...")
        print("  (This will open a window showing the camera feed)")
        print("  (Press 'q' or close the window to exit)")
        print("  (Gestures will be printed to this terminal)")
        print()
        try:
            capture_and_display_processed_frames(
                frame_processor, "Gesture Detection Demo", args.camera
            )
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"\nError in camera capture: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Camera session ended.")


if __name__ == "__main__":
    main()
