#!/usr/bin/env python3
"""
Tello Drone Gesture Control (7 Gestures) — FLIGHT VERSION (FIXED COLORS + UI)

Gestures → Actions:
Open    → backward
Close   → land
Pointer → up
Four    → forward
Peace   → down
YOLO    → right
El      → left

Fixes vs your “blue-tint” version:
- Uses the same color pipeline as your diagnostic tool:
  - Treat frame as RGB for inference/overlay
  - Convert RGB -> BGR ONLY for cv2.imshow()
- Adds the same bottom “mapping” text
- Shows current gesture + stable gesture + next command preview

Added (per your request):
- Shows the last 5 gesture IDs on screen (like cam_gestures.py)
  Example: "Recent IDs: 6 6 6 6 6" when holding "El"
- Keeps stability logic at 3 frames (STABLE_FRAMES=3) for commands
- Does NOT change anything else about flight behavior / mapping / cooldown / colors
"""

import sys
import os
import time
import cv2
import numpy as np
import socket

from djitellopy import Tello

# ===============================
#  PATH SETUP
# ===============================
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models", "mediapipe_hand"))

from qai_hub_models.models.mediapipe_hand.app import MediaPipeHandApp
from qai_hub_models.models.mediapipe_hand.model import MediaPipeHand
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from model.keypoint_classifier.preprocess import preprocess_landmark, load_gesture_labels

# ===============================
#  GESTURE → MOVEMENT MAP
# ===============================
GESTURE_ACTIONS = {
    "Open":    ("move_back", 20),
    "Pointer": ("move_up", 20),
    "Four":    ("move_forward", 20),
    "Peace":   ("move_down", 20),
    "YOLO":    ("move_right", 20),
    "El":      ("move_left", 20),
    "Close":   ("land", 0),
}

# ===============================
#  EXPECTED GESTURE SET (ID mapping like diagnostic)
#  (Used only for consistent ID notion; flight logic still uses labels.)
# ===============================
EXPECTED_GESTURES = {
    0: ("Open", "backward"),
    1: ("Close", "done"),
    2: ("Pointer", "up"),
    3: ("Four", "forward"),
    4: ("Peace", "down"),
    5: ("YOLO", "right"),
    6: ("El", "left"),
}

# ===============================
#  LOAD MODELS
# ===============================
model = MediaPipeHand.from_pretrained()
app = MediaPipeHandApp.from_pretrained(model)
app.min_detector_box_score = 0.3

current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, "models", "mediapipe_hand", "model", "keypoint_classifier")

keypoint_classifier = KeyPointClassifier(
    model_path=os.path.join(model_dir, "keypoint_classifier.tflite")
)
gesture_labels = [
    g.strip().lstrip("\ufeff")
    for g in load_gesture_labels(os.path.join(model_dir, "keypoint_classifier_label.csv"))
]

# ===============================
#  CLASSIFY GESTURE
# ===============================
def classify_gesture(landmarks: np.ndarray):
    """
    landmarks: (21,3) float array
    returns: (label string or None, gesture_id int)
    """
    if landmarks is None or landmarks.shape != (21, 3):
        return None, -1

    pre = preprocess_landmark(landmarks)

    # Model expects 42 features
    if pre is None or pre.shape[0] != 42:
        return None, -1

    gid = keypoint_classifier(pre)

    if 0 <= gid < len(gesture_labels):
        return gesture_labels[gid], gid
    return None, -1

# ===============================
#  CONNECT TO TELLO (sanity ping)
# ===============================
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(2)
sock.connect(("192.168.10.1", 8889))
sock.close()

tello = Tello()
tello.connect()
tello.streamon()
time.sleep(2)

frame_reader = tello.get_frame_read()

print("\nTAKING OFF...")
tello.takeoff()
in_air = True
time.sleep(2)

# ===============================
#  CONTROL PARAMETERS
# ===============================
STABLE_FRAMES = 3
COMMAND_COOLDOWN = 1.5
gesture_history = []
gesture_id_history = []   # <-- last 5 IDs for on-screen debug
last_cmd_time = 0.0

# ===============================
#  UI HELPERS
# ===============================
MAPPING_LINES = [
    "Open=BACK  Close=LAND  Pointer=UP  Four=FWD",
    "Peace=DOWN  YOLO=RIGHT  El=LEFT",
]

def action_preview_from_stable(stable_label: str) -> str:
    if not stable_label:
        return ""
    action = GESTURE_ACTIONS.get(stable_label)
    if not action:
        return ""
    cmd, val = action
    if cmd == "land":
        return "LAND"
    return f"{cmd.upper()}({val})"

# ===============================
#  MAIN LOOP
# ===============================
try:
    while True:
        frame = frame_reader.frame
        if frame is None or frame.size == 0:
            continue

        # Mirror like your diagnostic script
        frame = cv2.flip(frame, 1)

        # --- IMPORTANT: match your diagnostic color pipeline ---
        # Treat as RGB for inference + overlay, then convert to BGR only for display.
        frame_rgb = frame.copy()

        # Detect landmarks
        raw = app.predict_landmarks_from_image(frame_rgb, raw_output=True)
        landmarks_batch = raw[3]  # landmarks at index 3

        current_gesture = None
        gesture_id = -1

        # Classify first detected hand
        for t in landmarks_batch:
            if t.nelement() > 0:
                lm = t.cpu().numpy()[0]
                current_gesture, gesture_id = classify_gesture(lm)
                break

        # Stability logic (STABLE_FRAMES=3 for command firing)
        if current_gesture:
            gesture_history.append(current_gesture)
            gesture_id_history.append(gesture_id)

            if len(gesture_history) > STABLE_FRAMES:
                gesture_history.pop(0)

            # Keep last 5 IDs for display (like diagnostic)
            if len(gesture_id_history) > 5:
                gesture_id_history.pop(0)
        else:
            gesture_history.clear()
            gesture_id_history.clear()

        stable = None
        if len(gesture_history) == STABLE_FRAMES and all(g == gesture_history[0] for g in gesture_history):
            stable = gesture_history[0]

        # -------------------------------
        # DISPLAY (same vibe as diagnostic)
        # -------------------------------
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        next_action = action_preview_from_stable(stable)

        # Top lines: show ID + label + stability + next command
        cv2.putText(
            frame_bgr,
            f"Gesture ID: {gesture_id} | Label: {current_gesture} | Stable: {stable} | Next: {next_action}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )

        # Show last 5 IDs (e.g., 6 6 6 6 6)
        recent_ids_str = " ".join(str(i) for i in gesture_id_history)
        cv2.putText(
            frame_bgr,
            f"Recent IDs: {recent_ids_str}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        # Bottom mapping text (commands)
        base_y = frame_bgr.shape[0] - 30
        for j, line in enumerate(MAPPING_LINES[::-1]):
            cv2.putText(
                frame_bgr,
                line,
                (10, base_y - j * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

        cv2.imshow("Tello Gesture Control", frame_bgr)

        # -------------------------------
        # EXECUTE COMMAND
        # -------------------------------
        now = time.time()
        if stable and (now - last_cmd_time > COMMAND_COOLDOWN):
            action = GESTURE_ACTIONS.get(stable)
            if action:
                cmd, val = action
                print(f"EXECUTE → {stable.upper()}  =>  {cmd} {val if cmd != 'land' else ''}".strip())

                if cmd == "land":
                    tello.land()
                    in_air = False
                    break
                else:
                    # move_* commands
                    getattr(tello, cmd)(val)

                last_cmd_time = now
                gesture_history.clear()
                gesture_id_history.clear()

        # Quit key
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

except KeyboardInterrupt:
    pass

finally:
    # ===============================
    #  CLEANUP
    # ===============================
    print("\nCLEANUP...")
    try:
        if in_air:
            print("LANDING...")
            tello.land()
    except Exception:
        pass

    try:
        tello.streamoff()
    except Exception:
        pass

    cv2.destroyAllWindows()
    print("DONE.")
