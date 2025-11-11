# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import argparse
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

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "hand.jpeg"
)


def classify_gesture(landmarks: np.ndarray) -> str:
    """
    Classify hand gesture from landmarks.
    
    Landmark indices:
    - 0: Wrist
    - 4: Thumb tip
    - 8: Index tip
    - 12: Middle tip
    
    Args:
        landmarks: numpy array of shape (21, 3) with (x, y, z) coordinates
        
    Returns:
        "STOP", "UP", or "UNKNOWN"
    """
    if landmarks.shape[0] < 21:
        return "UNKNOWN"
    
    wrist = landmarks[0]  # Wrist (landmark 0)
    thumb_tip = landmarks[4]  # Thumb tip (landmark 4)
    index_tip = landmarks[8]  # Index tip (landmark 8)
    middle_tip = landmarks[12]  # Middle tip (landmark 12)
    
    # STOP gesture: thumb tip close to index tip (in pixel coordinates)
    thumb_index_distance = np.linalg.norm(thumb_tip[:2] - index_tip[:2])
    if thumb_index_distance < 30.0:  # Threshold in pixels for "close" (adjust based on image resolution)
        return "STOP"
    
    # UP gesture: index and middle tips are above wrist (lower y values)
    wrist_y = wrist[1]
    index_tip_y = index_tip[1]
    middle_tip_y = middle_tip[1]
    
    if index_tip_y < wrist_y and middle_tip_y < wrist_y:
        return "UP"
    
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
        default=0.95,
        help="Score threshold for NonMaximumSuppression",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="Intersection over Union (IoU) threshold for NonMaximumSuppression",
    )
    add_output_dir_arg(parser)

    print(
        "Note: This demo is running through torch, and not meant to be real-time without dedicated ML hardware."
    )
    print("Use Ctrl+C in your terminal to exit.")

    args = parser.parse_args([] if is_test else None)
    if is_test:
        args.image = INPUT_IMAGE_ADDRESS

    # Load app
    app = MediaPipeHandApp.from_pretrained(MediaPipeHand.from_pretrained())
    print("Model and App Loaded")

    if args.image:
        image = load_image(args.image)
        # Get raw landmarks for gesture classification
        raw_result = app.predict_landmarks_from_image(image, raw_output=True)
        # Extract the values we need (handles variable number of return values)
        batched_selected_boxes = raw_result[0]
        batched_selected_keypoints = raw_result[1]
        batched_selected_landmarks = raw_result[2]
        batched_is_right_hand = raw_result[3]
        
        # Classify gestures
        for batch_idx, landmarks in enumerate(batched_selected_landmarks):
            if landmarks.nelement() != 0:
                landmarks_np = landmarks.cpu().numpy()
                for hand_idx in range(landmarks_np.shape[0]):
                    gesture = classify_gesture(landmarks_np[hand_idx])
                    if gesture in ["STOP", "UP", "UNKNOWN"]:
                        print(f"Gesture detected: {gesture}")
        
        pred_image = app.predict_landmarks_from_image(image)
        assert isinstance(pred_image[0], np.ndarray)
        out_image = Image.fromarray(pred_image[0], "RGB")
        if not is_test:
            display_or_save_image(out_image, args.output_dir)
    else:
        last_gesture = None

        def frame_processor(frame: np.ndarray) -> np.ndarray:
            nonlocal last_gesture
            # Get raw landmarks for gesture classification
            raw_result = app.predict_landmarks_from_image(frame, raw_output=True)
            # Extract the values we need (handles variable number of return values)
            batched_selected_boxes = raw_result[0]
            batched_selected_keypoints = raw_result[1]
            batched_selected_landmarks = raw_result[2]
            batched_is_right_hand = raw_result[3]
            
            # Classify gestures and print when STOP or UP is detected
            current_gesture = None
            for batch_idx, landmarks in enumerate(batched_selected_landmarks):
                if landmarks.nelement() != 0:
                    landmarks_np = landmarks.cpu().numpy()
                    for hand_idx in range(landmarks_np.shape[0]):
                        gesture = classify_gesture(landmarks_np[hand_idx])
                        if gesture in ["STOP", "UP"]:
                            current_gesture = gesture
            
            # Print only when gesture changes
            if current_gesture and current_gesture != last_gesture:
                print(f"Gesture detected: {current_gesture}")
                last_gesture = current_gesture
            elif not current_gesture:
                last_gesture = None
            
            return cast(np.ndarray, app.predict_landmarks_from_image(frame)[0])

        capture_and_display_processed_frames(
            frame_processor, "Gesture Detection Demo", args.camera
        )


if __name__ == "__main__":
    main()
