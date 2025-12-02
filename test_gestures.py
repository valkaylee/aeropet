print("Starting script...")
print("Importing libraries...")

try:
    from djitellopy import Tello
    print("✓ djitellopy imported")
except Exception as e:
    print(f"ERROR importing djitellopy: {e}")
    sys.exit(1)

import cv2
import time
import os
import sys
import numpy as np

print("✓ Basic libraries imported")

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

# Load MediaPipe Hand model
print("Loading MediaPipe Hand model...")
try:
    model = MediaPipeHand.from_pretrained()
    app = MediaPipeHandApp.from_pretrained(model)
    app.min_detector_box_score = 0.5  # Lower threshold for better detection
    print("✓ MediaPipe Hand model loaded")
except Exception as e:
    print(f"ERROR loading MediaPipe Hand model: {e}")
    sys.exit(1)

# Load gesture classifier
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, 'models', 'mediapipe_hand', 'model', 'keypoint_classifier')
model_path = os.path.join(model_dir, 'keypoint_classifier.tflite')
label_path = os.path.join(model_dir, 'keypoint_classifier_label.csv')

print(f"Loading gesture classifier from {model_path}...")
try:
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        sys.exit(1)
    if not os.path.exists(label_path):
        print(f"ERROR: Label file not found at {label_path}")
        sys.exit(1)
    
    classifier = KeyPointClassifier(model_path=model_path)
    labels = load_gesture_labels(label_path)
    print(f"✓ Gesture classifier loaded. Available gestures: {', '.join(labels)}")
except Exception as e:
    print(f"ERROR loading gesture classifier: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Initialize Tello
print("Initializing Tello connection...")
print("\n⚠️  IMPORTANT: Make sure you are connected to the Tello's WiFi network!")
print("   The Tello WiFi is usually named 'TELLO-XXXXXX'")
print("   Check your WiFi settings if you see connection errors.\n")

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
    print("  4. Make sure no firewall is blocking UDP port 8889")
    print("  5. Try restarting the Tello drone")
    sys.exit(1)

try:
    tello = Tello()
    print("✓ Tello object created")
    
    # Try connecting with retries
    max_retries = 3
    connected = False
    for attempt in range(1, max_retries + 1):
        print(f"Connecting to Tello (attempt {attempt}/{max_retries})...")
        print("   Attempting to connect to 192.168.10.1...")
        try:
            tello.connect()
            connected = True
            break
        except Exception as connect_error:
            if attempt < max_retries:
                print(f"   Connection attempt {attempt} failed: {connect_error}")
                print("   Retrying in 2 seconds...")
                time.sleep(2)
            else:
                raise connect_error
    
    if connected:
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
    print(f"\nERROR connecting to Tello: {e}")
    print("\nAdditional troubleshooting:")
    print("  - Is the Tello powered ON and in command mode?")
    print("  - Try unplugging and replugging the Tello's battery")
    print("  - Some Tellos need to be 'woken up' by sending a command via the Tello app first")
    print("  - Check if your firewall/antivirus is blocking UDP connections")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Take off
print("Taking off...")
try:
    tello.takeoff()
    time.sleep(2)  # Wait for takeoff to complete
    print("✓ Drone is airborne")
except Exception as e:
    print(f"ERROR during takeoff: {e}")
    import traceback
    traceback.print_exc()
    tello.streamoff()
    sys.exit(1)

last_cmd = 0
gesture_history = []  # Track recent gestures for stability
GESTURE_STABILITY_FRAMES = 15  # Require gesture to be detected 15 times in a row (~0.5 seconds at 30 FPS)
frame_count = 0

def detect_hand_landmarks(frame):
    """Detect hand landmarks from a frame using MediaPipe Hand."""
    raw_result = app.predict_landmarks_from_image(frame, raw_output=True)
    batched_selected_landmarks = raw_result[3]  # Landmarks are at index 3
    
    # Extract first hand's landmarks if available
    for landmarks_tensor in batched_selected_landmarks:
        if landmarks_tensor.nelement() != 0:
            landmarks_np = landmarks_tensor.cpu().numpy()
            if landmarks_np.shape[0] > 0:
                # Return first hand's landmarks (21, 3)
                hand_landmarks = landmarks_np[0]
                if hand_landmarks.shape == (21, 3):
                    return hand_landmarks
    return None

print("Starting gesture recognition loop.")
print("Gestures: Pointer (up), Open (backward), OK (left), Close/Closed fist (land)")
print("Note: Hold gesture steady for best results\n")
while True:
    frame = frame_reader.frame
    frame_count += 1

    landmarks = detect_hand_landmarks(frame)
    if landmarks is None:
        gesture_history = []  # Reset history if no hand detected
        continue

    # Preprocess and classify gesture
    features = preprocess_landmark(landmarks)
    gesture_id = classifier(features)
    gesture = labels[gesture_id] if 0 <= gesture_id < len(labels) else "UNKNOWN"

    # Add to gesture history (keep last N gestures)
    gesture_history.append(gesture)
    if len(gesture_history) > GESTURE_STABILITY_FRAMES:
        gesture_history.pop(0)
    
    # Check if gesture is stable (same gesture for required frames)
    stable_gesture = None
    if len(gesture_history) >= GESTURE_STABILITY_FRAMES:
        # Check if all recent gestures are the same
        if all(g == gesture_history[0] for g in gesture_history):
            stable_gesture = gesture_history[0]

    # Display detected gesture on frame
    color = (0, 255, 0) if stable_gesture else (0, 165, 255)  # Green if stable, orange if not
    cv2.putText(frame, f"Gesture: {gesture}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    if stable_gesture:
        cv2.putText(frame, f"STABLE: {stable_gesture}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        stability = len([g for g in gesture_history if g == gesture])
        cv2.putText(frame, f"Stability: {stability}/{GESTURE_STABILITY_FRAMES}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    
    cv2.putText(frame, f"Frame: {frame_count}", (10, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow("Tello", frame)

    now = time.time()

    # Only execute if gesture is stable AND enough time has passed
    if stable_gesture and now - last_cmd > 1:    
        if stable_gesture == "Pointer":
            print(f"[Frame {frame_count}] Detected: {stable_gesture} - Moving up")
            tello.move_up(20)
            last_cmd = now
            gesture_history = []  # Reset after command
        elif stable_gesture == "Open":
            print(f"[Frame {frame_count}] Detected: {stable_gesture} - Moving backward")
            tello.move_back(20)
            last_cmd = now
            gesture_history = []  # Reset after command
        elif stable_gesture == "OK":
            print(f"[Frame {frame_count}] Detected: {stable_gesture} - Moving left")
            tello.move_left(20)
            last_cmd = now
            gesture_history = []  # Reset after command
        elif stable_gesture == "Close":
            print(f"[Frame {frame_count}] Detected: {stable_gesture} - Landing")
            tello.land()
            last_cmd = now
            break  # Exit loop after landing

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Emergency stop - Landing")
        tello.land()
        break

tello.streamoff()
cv2.destroyAllWindows()
print(f"\nDone - Processed {frame_count} frames total")
