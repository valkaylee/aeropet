#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Collection Script for Hand Gesture Training
Replicates kinivi's training workflow using Qualcomm AI Hub MediaPipe Hand

Usage:
    python collect_training_data.py --camera 0

Controls:
    - Press 'k' to enter/exit logging mode
    - Press '0'-'9' to label the current gesture (only works in logging mode)
    - Press 'q' to quit
"""

import argparse
import csv
import os
from typing import cast

import cv2
import numpy as np

from qai_hub_models.models.mediapipe_hand.app import MediaPipeHandApp
from qai_hub_models.models.mediapipe_hand.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    MediaPipeHand,
)
from model.keypoint_classifier.preprocess import preprocess_landmark


def main():
    parser = argparse.ArgumentParser(
        description="Collect hand gesture training data"
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
        default=0.5,
        help="Score threshold for hand detection (lower = more detections)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save training data CSV (default: model/keypoint_classifier/)",
    )
    args = parser.parse_args()

    # Set up output file path
    if args.output_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, 'model', 'keypoint_classifier')
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'keypoint.csv')
    
    # Check if CSV exists and load existing data count
    existing_samples = 0
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            existing_samples = sum(1 for row in reader) - 1  # Subtract header
        print(f"Found existing CSV with {existing_samples} samples")
    
    # Load MediaPipe Hand model
    print("Loading MediaPipe Hand model...")
    model = MediaPipeHand.from_pretrained()
    app = MediaPipeHandApp.from_pretrained(model)
    app.min_detector_box_score = args.score_threshold
    print("✓ Model loaded")
    
    # Load gesture labels to show which number corresponds to which gesture
    label_path = os.path.join(output_dir, 'keypoint_classifier_label.csv')
    gesture_labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r', encoding='utf-8') as f:
            gesture_labels = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(gesture_labels)} gesture labels: {gesture_labels}")
    else:
        print(f"Warning: Label file not found at {label_path}")
        print("  Create this file with one gesture name per line (e.g., Open, Close, Pointer, OK)")
    
    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return
    
    print(f"Camera opened (index {args.camera})")
    print("\n" + "="*60)
    print("DATA COLLECTION MODE")
    print("="*60)
    print("Controls:")
    print("  Press 'k' to enter/exit logging mode")
    print("  Press '0'-'9' to label gesture (only in logging mode)")
    print("  Press 'q' to quit")
    print("="*60)
    if gesture_labels:
        print("\nGesture Labels:")
        for i, label in enumerate(gesture_labels):
            if i < 10:
                print(f"  {i}: {label}")
    print()
    
    # State variables
    logging_mode = False
    sample_count = existing_samples
    frame_count = 0

    # Inference settings - run model every N frames for better responsiveness
    INFERENCE_INTERVAL = 3  # Run inference every 3 frames

    # Cached landmarks (persists between inference frames)
    cached_landmarks = None
    hand_detected = False

    # Open CSV file for appending
    csv_file = open(csv_path, 'a', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)

    # Write header if file is new
    if existing_samples == 0:
        # Header: label, x0, y0, x1, y1, ..., x20, y20 (42 features)
        header = ['label'] + [f'x{i//2}' if i % 2 == 0 else f'y{i//2}' for i in range(42)]
        csv_writer.writerow(header)
        csv_file.flush()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame")
                break

            frame_count += 1

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Only run inference every N frames (expensive operation)
            if frame_count % INFERENCE_INTERVAL == 0:
                raw_result = app.predict_landmarks_from_image(frame, raw_output=True)
                batched_selected_landmarks = raw_result[3]

                # Process landmarks
                hand_detected = False
                cached_landmarks = None

                for landmarks_tensor in batched_selected_landmarks:
                    if landmarks_tensor.nelement() != 0:
                        landmarks_np = landmarks_tensor.cpu().numpy()
                        if landmarks_np.shape[0] > 0:
                            hand_detected = True
                            cached_landmarks = landmarks_np[0]  # Use first hand
                            break

            # Draw landmarks on frame (use cached landmarks)
            display_frame = frame.copy()
            if hand_detected and cached_landmarks is not None:
                # Draw landmarks
                for i, landmark in enumerate(cached_landmarks):
                    x, y = int(landmark[0]), int(landmark[1])
                    cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
                    if i == 0:  # Wrist
                        cv2.circle(display_frame, (x, y), 8, (255, 0, 0), 2)

            # Draw UI overlay - make logging mode VERY obvious
            if logging_mode:
                # Draw thick green border when logging
                border_thickness = 15
                cv2.rectangle(display_frame, (0, 0),
                             (display_frame.shape[1], display_frame.shape[0]),
                             (0, 255, 0), border_thickness)

                # Large green banner at top
                cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 50), (0, 200, 0), -1)
                cv2.putText(display_frame, "** LOGGING MODE - Press 0-9 to save **", (10, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                # Show available gestures
                if gesture_labels:
                    label_text = "Labels: " + ", ".join([f"{i}={g}" for i, g in enumerate(gesture_labels[:10])])
                    cv2.putText(display_frame, label_text, (10, 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                # Red text when not logging
                cv2.putText(display_frame, "NORMAL MODE - Press 'k' to start logging", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Hand detection status
            hand_status = "Hand: DETECTED" if hand_detected else "Hand: NOT DETECTED"
            hand_color = (0, 255, 0) if hand_detected else (0, 0, 255)
            cv2.putText(display_frame, hand_status, (10, display_frame.shape[0] - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)

            cv2.putText(display_frame, f"Samples collected: {sample_count}", (10, display_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show frame
            cv2.imshow('Hand Gesture Data Collection', display_frame)

            # Handle keyboard input - this now runs every frame for responsiveness
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('k'):
                logging_mode = not logging_mode
                status = "ENABLED" if logging_mode else "DISABLED"
                print(f"\n>>> Logging mode {status} <<<\n")
            elif logging_mode and key >= ord('0') and key <= ord('9'):
                label_id = key - ord('0')

                if hand_detected and cached_landmarks is not None:
                    try:
                        # Preprocess landmarks (same as used in inference)
                        preprocessed = preprocess_landmark(cached_landmarks)

                        # Write to CSV: label, x0, y0, x1, y1, ..., x20, y20
                        row = [label_id] + preprocessed.tolist()
                        csv_writer.writerow(row)
                        csv_file.flush()

                        sample_count += 1
                        label_name = gesture_labels[label_id] if label_id < len(gesture_labels) else str(label_id)
                        print(f"  ✓ Saved sample {sample_count}: Label {label_id} ({label_name})")
                    except Exception as e:
                        print(f"  ✗ Error saving sample: {e}")
                else:
                    print(f"  ⚠️  No hand detected. Cannot save sample for label {label_id}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        csv_file.close()
        cv2.destroyAllWindows()
        print(f"\nData collection complete!")
        print(f"Total samples collected: {sample_count}")
        print(f"CSV file saved to: {csv_path}")
        print(f"\nNext step: Train the model using train_gesture_classifier.py")


if __name__ == "__main__":
    main()

