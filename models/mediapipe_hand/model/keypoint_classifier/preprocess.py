"""
Preprocessing utilities for converting Qualcomm AI Hub MediaPipe landmarks
to the format expected by kinivi's gesture classifier.
"""
import numpy as np


def preprocess_landmark(landmark_list: np.ndarray) -> np.ndarray:
    """
    Preprocess hand landmarks for gesture classification.
    
    This function converts landmarks to relative coordinates (relative to wrist)
    and normalizes them, matching the preprocessing used in kinivi's repository.
    
    Args:
        landmark_list: numpy array of shape (21, 3) with (x, y, z) coordinates
                      from Qualcomm AI Hub MediaPipe Hand
        
    Returns:
        preprocessed_landmarks: 1D numpy array of shape (42,) containing
                                normalized relative coordinates (x, y only)
    """
    if landmark_list.shape[0] != 21:
        raise ValueError(f"Expected 21 landmarks, got {landmark_list.shape[0]}")
    
    # Convert to numpy array if needed
    temp_landmark_list = np.array(landmark_list, dtype=np.float32)
    
    # Extract only x, y coordinates (z is not used by the model)
    # The model expects 42 features (21 landmarks * 2 coordinates)
    xy_landmarks = temp_landmark_list[:, :2]  # Shape: (21, 2)
    
    # Base point is the wrist (landmark 0)
    base_x, base_y = xy_landmarks[0]
    
    # Convert to relative coordinates (relative to wrist)
    relative_landmark_list = xy_landmarks.copy()
    relative_landmark_list[:, 0] = relative_landmark_list[:, 0] - base_x
    relative_landmark_list[:, 1] = relative_landmark_list[:, 1] - base_y
    
    # Normalize by the maximum absolute value to scale to [-1, 1] range
    # This helps with gesture recognition regardless of hand size/distance
    max_value = np.max(np.abs(relative_landmark_list))
    
    if max_value > 0:
        relative_landmark_list = relative_landmark_list / max_value
    
    # Flatten to 1D array: [x0, y0, x1, y1, ..., x20, y20]
    # This gives us 42 features (21 landmarks * 2 coordinates)
    preprocessed_landmarks = relative_landmark_list.flatten()
    
    return preprocessed_landmarks


def load_gesture_labels(label_file_path: str) -> list[str]:
    """
    Load gesture labels from CSV file.
    
    Args:
        label_file_path: Path to the label CSV file
        
    Returns:
        List of gesture label strings
    """
    labels = []
    try:
        with open(label_file_path, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Warning: Label file not found at {label_file_path}")
        labels = ["Unknown"]
    
    return labels

