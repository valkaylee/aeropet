#!/usr/bin/env python3
"""
Tello Gesture Detection Diagnostic Tool
Uses Tello camera to detect and classify gestures WITHOUT moving the drone.
Shows detailed debug output to identify classification issues.
"""

import sys
import os
import time
import cv2
import numpy as np

# Import Tello
try:
    from djitellopy import Tello
    print("✓ djitellopy imported")
except Exception as e:
    print(f"ERROR importing djitellopy: {e}")
    sys.exit(1)

# Add the models directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models', 'mediapipe_hand'))
print("✓ Added models directory to path")

# Import MediaPipe Hand App for landmark detection
try:
    from qai_hub_models.models.mediapipe_hand.app import MediaPipeHandApp
    from qai_hub_models.models.mediapipe_hand.model import MediaPipeHand
    print("✓ MediaPipe Hand imports successful")
except Exception as e:
    print(f"ERROR importing MediaPipe Hand: {e}")
    sys.exit(1)

# Import gesture classifier components
try:
    from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
    from model.keypoint_classifier.preprocess import preprocess_landmark, load_gesture_labels
    print("✓ Gesture classifier imports successful")
except Exception as e:
    print(f"ERROR importing gesture classifier: {e}")
    sys.exit(1)

# ===============================
#  LOAD MODELS
# ===============================
print("\nLoading MediaPipe Hand model...")
try:
    model = MediaPipeHand.from_pretrained()
    app = MediaPipeHandApp.from_pretrained(model)
    app.min_detector_box_score = 0.3  # Lower threshold for better detection on Tello
    print("✓ MediaPipe Hand model loaded")
except Exception as e:
    print(f"ERROR loading MediaPipe Hand model: {e}")
    sys.exit(1)

# Load gesture classifier
print("Loading gesture classifier...")
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, 'models', 'mediapipe_hand', 'model', 'keypoint_classifier')
model_path = os.path.join(model_dir, 'keypoint_classifier.tflite')
label_path = os.path.join(model_dir, 'keypoint_classifier_label.csv')

try:
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        sys.exit(1)
    if not os.path.exists(label_path):
        print(f"ERROR: Label file not found at {label_path}")
        sys.exit(1)
    
    keypoint_classifier = KeyPointClassifier(model_path=model_path)
    gesture_labels = load_gesture_labels(label_path)
    
    # Normalize labels: strip whitespace + BOM (\ufeff)
    gesture_labels = [label.strip().lstrip("\ufeff") for label in gesture_labels]
    
    print(f"✓ Gesture classifier loaded.")
    print(f"  Label mapping: {dict(enumerate(gesture_labels))}")
    print(f"  Available gestures: {', '.join(gesture_labels)}")
except Exception as e:
    print(f"ERROR loading gesture classifier: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ===============================
#  GESTURE CLASSIFICATION FUNCTION
# ===============================
def classify_gesture(landmarks: np.ndarray, debug=False) -> tuple:
    """
    Classify hand gesture from landmarks using the gesture classifier.
    
    Args:
        landmarks: numpy array of shape (21, 3) with (x, y, z) coordinates
        debug: If True, print detailed debug info
        
    Returns:
        Tuple of (gesture_label, gesture_id, confidence_info)
    """
    if landmarks.shape != (21, 3):
        return ("UNKNOWN", -1, "Wrong shape")
    
    try:
        # Debug: Show landmark ranges
        if debug:
            print(f"      Landmark ranges: X=[{landmarks[:, 0].min():.3f}, {landmarks[:, 0].max():.3f}], "
                  f"Y=[{landmarks[:, 1].min():.3f}, {landmarks[:, 1].max():.3f}], "
                  f"Z=[{landmarks[:, 2].min():.3f}, {landmarks[:, 2].max():.3f}]")
        
        # Preprocess landmarks for the classifier
        preprocessed_landmarks = preprocess_landmark(landmarks)
        
        # Validate preprocessed landmarks (model expects 42 features: 21 landmarks * 2 coords)
        if preprocessed_landmarks.shape[0] != 42:
            return ("UNKNOWN", -1, f"Wrong preprocessed shape: {preprocessed_landmarks.shape}")
        
        # Debug: Show preprocessed ranges
        if debug:
            print(f"      Preprocessed range: [{preprocessed_landmarks.min():.3f}, {preprocessed_landmarks.max():.3f}]")
        
        # Classify gesture
        gesture_id = keypoint_classifier(preprocessed_landmarks)
        
        # Get gesture label
        if 0 <= gesture_id < len(gesture_labels):
            label = gesture_labels[gesture_id].strip().lstrip("\ufeff")
            return (label, gesture_id, "OK")
        else:
            return ("UNKNOWN", gesture_id, f"Invalid ID: {gesture_id}, max: {len(gesture_labels)-1}")
    except Exception as e:
        return ("UNKNOWN", -1, f"Error: {e}")

# ===============================
#  CONNECT TO TELLO (NO TAKEOFF)
# ===============================
print("\nInitializing Tello connection...")
print("⚠️  Make sure you are connected to the Tello's WiFi network!")
print("⚠️  NOTE: This script will NOT take off or move the drone!")

# Test network connectivity first
import socket
print("Testing network connectivity to Tello (192.168.10.1)...")
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2)
    sock.connect(('192.168.10.1', 8889))
    sock.close()
    print("✓ Network connectivity test passed")
except Exception as e:
    print(f"✗ Network connectivity test failed: {e}")
    print("\nTroubleshooting tips:")
    print("  1. Make sure the Tello is powered ON")
    print("  2. Verify you're connected to the Tello WiFi (check WiFi settings)")
    print("  3. Try disconnecting and reconnecting to the Tello WiFi")
    sys.exit(1)

try:
    tello = Tello()
    print("✓ Tello object created")
    
    print("Connecting to Tello...")
    tello.connect()
    battery = tello.get_battery()
    print(f"✓ Connected to Tello. Battery: {battery}%")
    
    if battery < 20:
        print("⚠️  WARNING: Battery is low!")
    
    print("Starting video stream...")
    tello.streamon()
    time.sleep(2)  # Give stream time to start
    frame_reader = tello.get_frame_read()
    print("✓ Video stream started")
    print("\n" + "="*70)
    print("DIAGNOSTIC MODE - NO DRONE MOVEMENT")
    print("="*70)
    print("Show gestures to the camera. Press 'q' to quit.")
    print("="*70 + "\n")
except Exception as e:
    print(f"ERROR connecting to Tello: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ===============================
#  GESTURE DETECTION LOOP
# ===============================
frame_count = 0
gesture_history = []
last_gesture = None
last_print_time = 0

try:
    while True:
        # Get frame from Tello
        frame = frame_reader.frame
        frame_count += 1
        
        # Skip invalid frames
        if frame is None or frame.size == 0:
            continue
        
        # Verify frame dimensions
        if len(frame.shape) != 3 or frame.shape[0] == 0 or frame.shape[1] == 0:
            continue
        
        # Debug: Print frame info on first frame
        if frame_count == 1:
            print(f"Frame info: shape={frame.shape}, dtype={frame.dtype}, "
                  f"min={frame.min()}, max={frame.max()}")
        
        try:
            # Tello sends RGB, convert to BGR for display later
            # But keep RGB for MediaPipe processing
            frame_rgb = frame.copy()  # Tello already sends RGB
            
            # Get raw landmarks for gesture classification
            raw_result = app.predict_landmarks_from_image(frame_rgb, raw_output=True)
            batched_selected_landmarks = raw_result[3]  # Landmarks are at index 3
            
        except Exception as e:
            # Skip corrupted frames silently (reduces console spam)
            if frame_count % 30 == 0:
                print(f"[Frame {frame_count}] Error processing frame: {e}")
            continue
        
        # Process detected hands
        current_gesture = None
        gesture_id = -1
        confidence_info = ""
        hands_detected = 0
        
        for landmarks_tensor in batched_selected_landmarks:
            if landmarks_tensor.nelement() != 0:
                landmarks_np = landmarks_tensor.cpu().numpy()
                hands_detected += landmarks_np.shape[0]
                
                # Process first hand only
                if landmarks_np.shape[0] > 0:
                    hand_landmarks = landmarks_np[0]
                    if hand_landmarks.shape == (21, 3):
                        # Debug mode when gesture changes or every 30 frames
                        debug_mode = (current_gesture is None or 
                                     (frame_count % 30 == 0 and current_gesture != "UNKNOWN"))
                        current_gesture, gesture_id, confidence_info = classify_gesture(
                            hand_landmarks, debug=debug_mode)
                        break  # Use first hand only
        
        # Update gesture history
        if current_gesture and current_gesture != "UNKNOWN":
            gesture_history.append((current_gesture, gesture_id))
            if len(gesture_history) > 10:  # Keep last 10 for analysis
                gesture_history.pop(0)
        else:
            if hands_detected == 0:
                gesture_history.clear()
        
        # Print detailed info every frame when gesture changes or every 2 seconds
        now = time.time()
        should_print = False
        
        if current_gesture and current_gesture != last_gesture:
            should_print = True
            last_gesture = current_gesture
        elif now - last_print_time > 2.0:  # Print status every 2 seconds
            should_print = True
            last_print_time = now
        
        if should_print:
            gesture_str = current_gesture if current_gesture else "None"
            print(f"[Frame {frame_count:4d}] Hands: {hands_detected} | "
                  f"Gesture ID: {gesture_id:2d} | Label: '{gesture_str:8s}' | "
                  f"Info: {confidence_info}")
            if gesture_history:
                recent = gesture_history[-5:]  # Show last 5
                ids = [g[1] for g in recent]
                labels = [g[0] for g in recent]
                print(f"         Recent history: IDs={ids} | Labels={labels}")
            
            # Show statistics
            if gesture_history:
                id_counts = {}
                for g, gid in gesture_history:
                    id_counts[gid] = id_counts.get(gid, 0) + 1
                print(f"         ID distribution: {id_counts}")
        
        # Display on frame (draw on RGB frame, will convert to BGR for display)
        display_lines = [
            f"Frame: {frame_count}",
            f"Hands: {hands_detected}",
        ]
        
        if current_gesture:
            display_lines.append(f"Gesture ID: {gesture_id}")
            display_lines.append(f"Label: {current_gesture}")
            if confidence_info != "OK":
                display_lines.append(f"Info: {confidence_info}")
        
        # Show recent history
        if gesture_history:
            recent_ids = [str(g[1]) for g in gesture_history[-5:]]
            display_lines.append(f"Recent IDs: {','.join(recent_ids)}")
        
        y_offset = 30
        for i, line in enumerate(display_lines):
            color = (0, 255, 0) if current_gesture and current_gesture != "UNKNOWN" else (0, 165, 255)
            cv2.putText(frame_rgb, line, (10, y_offset + i * 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Highlight expected gestures
        if gesture_id == 0:
            cv2.putText(frame_rgb, "EXPECTED: Open (ID=0)", (10, frame_rgb.shape[0] - 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif gesture_id == 1:
            cv2.putText(frame_rgb, "EXPECTED: Close (ID=1)", (10, frame_rgb.shape[0] - 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif gesture_id == 2:
            cv2.putText(frame_rgb, "EXPECTED: Pointer (ID=2)", (10, frame_rgb.shape[0] - 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif gesture_id == 3:
            cv2.putText(frame_rgb, "EXPECTED: OK (ID=3)", (10, frame_rgb.shape[0] - 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Convert RGB to BGR for OpenCV display (fixes green/purple tint)
        frame_display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        cv2.imshow("Tello Gesture Diagnostic (NO MOVEMENT)", frame_display)
        
        # Reduce lag with minimal wait time
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nExiting...")
            break

except KeyboardInterrupt:
    print("\nCtrl+C - Exiting")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()

# Cleanup
try:
    tello.streamoff()
except:
    pass
cv2.destroyAllWindows()
print(f"\n✓ Done - Processed {frame_count} frames")
print(f"  Final gesture history: {gesture_history[-10:] if gesture_history else 'None'}")