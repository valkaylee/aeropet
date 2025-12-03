#!/usr/bin/env python3
"""
Tello Drone Gesture Control
Uses MediaPipe Hand detection and gesture classification to control Tello drone.
Gestures: OK (left), Pointer (up), Open (backward), Close (land)
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
    
    print(f"✓ Gesture classifier loaded. Available gestures: {', '.join(gesture_labels)}")
except Exception as e:
    print(f"ERROR loading gesture classifier: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ===============================
#  GESTURE CLASSIFICATION FUNCTION
# ===============================
def classify_gesture(landmarks: np.ndarray) -> str:
    """
    Classify hand gesture from landmarks using the gesture classifier.
    
    Args:
        landmarks: numpy array of shape (21, 3) with (x, y, z) coordinates
        
    Returns:
        Gesture label string (e.g., "Open", "Close", "Pointer", "OK") or "UNKNOWN"
    """
    if landmarks.shape != (21, 3):
        return "UNKNOWN"
    
    try:
        # Preprocess landmarks for the classifier
        preprocessed_landmarks = preprocess_landmark(landmarks)
        
        # Validate preprocessed landmarks (model expects 42 features: 21 landmarks * 2 coords)
        if preprocessed_landmarks.shape[0] != 42:
            return "UNKNOWN"
        
        # Classify gesture
        gesture_id = keypoint_classifier(preprocessed_landmarks)
        
        # Get gesture label
        if 0 <= gesture_id < len(gesture_labels):
            label = gesture_labels[gesture_id].strip().lstrip("\ufeff")
            # Debug: Print all classifications to see what's happening
            print(f"    → gesture_id={gesture_id} → '{label}'")
            return label
        else:
            if gesture_id >= 0:  # Only log if it's a valid index issue
                print(f"    ⚠️  Invalid gesture_id {gesture_id}, expected 0-{len(gesture_labels)-1}")
            return "UNKNOWN"
    except Exception as e:
        return "UNKNOWN"

# ===============================
#  CONNECT TO TELLO
# ===============================
print("\nInitializing Tello connection...")
print("⚠️  Make sure you are connected to the Tello's WiFi network!")

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
        print("⚠️  WARNING: Battery is low! Consider charging before flight.")
    
    print("Starting video stream...")
    tello.streamon()
    time.sleep(2)  # Give stream time to start
    frame_reader = tello.get_frame_read()
    print("✓ Video stream started")
except Exception as e:
    print(f"ERROR connecting to Tello: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Take off
print("\nTaking off...")
try:
    tello.takeoff()
    time.sleep(2)  # Wait for takeoff to complete
    print("✓ Drone is airborne")
except Exception as e:
    print(f"ERROR during takeoff: {e}")
    tello.streamoff()
    sys.exit(1)

# ===============================
#  GESTURE CONTROL LOOP
# ===============================
print("\n" + "="*60)
print("Gesture Control Active")
print("="*60)
print("Gestures:")
print("  OK → Move LEFT")
print("  Pointer → Move UP")
print("  Open → Move BACKWARD")
print("  Close → LAND")
print("="*60)
print("\nShow your hand to the camera...\n")

# Parameters
COMMAND_COOLDOWN = 2.0  # seconds between commands (increased to prevent rapid repeats)
GESTURE_STABILITY_FRAMES = 5  # Need same gesture 5 times in a row (more stable)
FRAME_SKIP = 1  # Process every frame for best gesture detection

# State
gesture_history = []
frame_count = 0
last_cmd_time = 0
last_gesture = None

try:
    while True:
        frame = frame_reader.frame
        frame_count += 1
        
        if frame is None or frame.size == 0:
            continue
        
        # Skip most frames for performance (still show video)
        if frame_count % FRAME_SKIP != 0:
            cv2.imshow("Tello Gesture Control", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        # Get raw landmarks for gesture classification
        try:
            raw_result = app.predict_landmarks_from_image(frame, raw_output=True)
            batched_selected_landmarks = raw_result[3]  # Landmarks are at index 3
        except Exception:
            # Skip corrupted frames
            cv2.imshow("Tello Gesture Control", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        # Process detected hands (same logic as gesture_demo.py)
        current_gesture = None
        hands_detected = 0
        
        for landmarks_tensor in batched_selected_landmarks:
            if landmarks_tensor.nelement() != 0:
                landmarks_np = landmarks_tensor.cpu().numpy()
                # landmarks_np shape should be (num_hands, 21, 3)
                hands_detected += landmarks_np.shape[0]
                
                # Process first hand only
                if landmarks_np.shape[0] > 0:
                    hand_landmarks = landmarks_np[0]
                    # Check if we have the right shape
                    if hand_landmarks.shape == (21, 3):
                        try:
                            current_gesture = classify_gesture(hand_landmarks)
                            if current_gesture == "UNKNOWN":
                                current_gesture = None
                        except Exception as e:
                            # Skip classification errors
                            current_gesture = None
                        break  # Use first hand only
                    else:
                        # Debug: log unexpected shapes
                        if frame_count % 30 == 0:  # Log occasionally
                            print(f"  ⚠️  Unexpected landmark shape: {hand_landmarks.shape}, expected (21, 3)")
        
        # Update gesture history with filtering
        if current_gesture and current_gesture != "UNKNOWN":
            # Only add if it's different from the last gesture (filter rapid switches)
            if len(gesture_history) == 0 or gesture_history[-1] != current_gesture:
                gesture_history.append(current_gesture)
            elif len(gesture_history) > 0:
                # If same as last, add it again (helps with stability)
                gesture_history.append(current_gesture)
            
            if len(gesture_history) > GESTURE_STABILITY_FRAMES * 2:  # Keep more history
                gesture_history.pop(0)
        else:
            # Clear history on no detection to avoid stale gestures
            if len(gesture_history) > 0:
                gesture_history.clear()
        
        # Check for stable gesture (same gesture 3 times in a row)
        stable_gesture = None
        if len(gesture_history) >= GESTURE_STABILITY_FRAMES:
            last_three = gesture_history[-GESTURE_STABILITY_FRAMES:]
            if all(g == last_three[0] for g in last_three):
                stable_gesture = last_three[0]
        
        # Display on frame
        display_text = f"Frame: {frame_count} | Hands: {hands_detected}"
        if current_gesture:
            display_text += f" | Gesture: {current_gesture}"
        if stable_gesture:
            display_text += f" | STABLE: {stable_gesture}"
        
        cv2.putText(frame, display_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show gesture history
        if gesture_history:
            hist_text = f"History: {gesture_history}"
            cv2.putText(frame, hist_text, (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow("Tello Gesture Control", frame)
        
        # Print gesture status with more detail (less verbose)
        if current_gesture:
            if current_gesture != last_gesture:
                print(f"[Frame {frame_count}] Detected: {current_gesture} | History: {gesture_history[-GESTURE_STABILITY_FRAMES:]}")
                last_gesture = current_gesture
            elif stable_gesture:
                if stable_gesture != getattr(print, '_last_stable', None):
                    print(f"[Frame {frame_count}] ✓✓✓ STABLE: {stable_gesture} ✓✓✓")
                    print._last_stable = stable_gesture
        
        # Execute commands if gesture is stable and cooldown has passed
        now = time.time()
        if stable_gesture and now - last_cmd_time >= COMMAND_COOLDOWN:
            if stable_gesture == "OK":
                print(f"→ EXECUTING: {stable_gesture} - Moving LEFT")
                tello.move_left(20)
                last_cmd_time = now
                gesture_history.clear()
                last_gesture = None  # Reset to allow new gesture detection
            elif stable_gesture == "Pointer":
                print(f"→ EXECUTING: {stable_gesture} - Moving UP")
                tello.move_up(20)
                last_cmd_time = now
                gesture_history.clear()
                last_gesture = None  # Reset to allow new gesture detection
            elif stable_gesture == "Open":
                print(f"→ EXECUTING: {stable_gesture} - Moving BACKWARD")
                tello.move_back(20)
                last_cmd_time = now
                gesture_history.clear()
                last_gesture = None  # Reset to allow new gesture detection
            elif stable_gesture == "Close":
                print(f"→ EXECUTING: {stable_gesture} - LANDING")
                tello.land()
                break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Emergency stop - Landing")
            tello.land()
            break

except KeyboardInterrupt:
    print("\nCtrl+C - Landing")
    try:
        tello.land()
    except:
        pass
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    try:
        tello.land()
    except:
        pass

# Cleanup
tello.streamoff()
cv2.destroyAllWindows()
print(f"\n✓ Done - Processed {frame_count} frames")
