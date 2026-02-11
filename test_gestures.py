#!/usr/bin/env python3
"""
Tello Drone Gesture Control (7 Gestures)

Gestures → Actions:
Open    → backward
Close   → done (hover / no move)
Pointer → up
Four    → forward
Peace   → right
YOLO    → down
El      → left
"""

import sys
import os
import time
import cv2
import numpy as np
import socket

# ===============================
#  IMPORT TELLO
# ===============================
from djitellopy import Tello

# ===============================
#  PATH SETUP
# ===============================
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models', 'mediapipe_hand'))

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
    "Peace":   ("move_right", 20),
    "YOLO":    ("move_down", 20),
    "El":      ("move_left", 20),
    "Close":   ("hover", 0),   # no movement
}

# ===============================
#  LOAD MODELS
# ===============================
model = MediaPipeHand.from_pretrained()
app = MediaPipeHandApp.from_pretrained(model)
app.min_detector_box_score = 0.3

current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, 'models', 'mediapipe_hand', 'model', 'keypoint_classifier')

keypoint_classifier = KeyPointClassifier(
    model_path=os.path.join(model_dir, 'keypoint_classifier.tflite')
)
gesture_labels = [
    g.strip().lstrip("\ufeff")
    for g in load_gesture_labels(os.path.join(model_dir, 'keypoint_classifier_label.csv'))
]

# ===============================
#  CLASSIFY GESTURE
# ===============================
def classify_gesture(landmarks):
    if landmarks.shape != (21, 3):
        return None
    pre = preprocess_landmark(landmarks)
    if pre.shape[0] != 42:
        return None
    gid = keypoint_classifier(pre)
    if 0 <= gid < len(gesture_labels):
        return gesture_labels[gid]
    return None

# ===============================
#  CONNECT TO TELLO
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
time.sleep(2)

# ===============================
#  CONTROL PARAMETERS
# ===============================
STABLE_FRAMES = 3          # ← CHANGED FROM 5 TO 3
COMMAND_COOLDOWN = 1.5
gesture_history = []
last_cmd_time = 0

# ===============================
#  MAIN LOOP
# ===============================
try:
    while True:
        frame = frame_reader.frame
        if frame is None:
            continue

        raw = app.predict_landmarks_from_image(frame, raw_output=True)
        landmarks_batch = raw[3]

        current_gesture = None

        for t in landmarks_batch:
            if t.nelement() > 0:
                lm = t.cpu().numpy()[0]
                current_gesture = classify_gesture(lm)
                break

        if current_gesture:
            gesture_history.append(current_gesture)
            if len(gesture_history) > STABLE_FRAMES:
                gesture_history.pop(0)
        else:
            gesture_history.clear()

        stable = None
        if len(gesture_history) == STABLE_FRAMES:
            if all(g == gesture_history[0] for g in gesture_history):
                stable = gesture_history[0]

        # DISPLAY
        cv2.putText(
            frame,
            f"Gesture: {current_gesture} | Stable: {stable}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Tello Gesture Control", frame)

        # EXECUTE COMMAND
        now = time.time()
        if stable and now - last_cmd_time > COMMAND_COOLDOWN:
            action = GESTURE_ACTIONS.get(stable)
            if action:
                cmd, val = action
                print(f"EXECUTE → {stable.upper()}")

                if cmd == "hover":
                    pass
                else:
                    getattr(tello, cmd)(val)

                last_cmd_time = now
                gesture_history.clear()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    pass

# ===============================
#  CLEANUP
# ===============================
print("\nLANDING...")
tello.land()
tello.streamoff()
cv2.destroyAllWindows()
