#!/usr/bin/env python3
"""
Tello Drone Gesture Control + AprilTag Authorization + BLE Buzzer Alarm

Behavior:
- Gestures only control the drone when the authorized AprilTag is visible.
- If the tag is not visible, gestures are ignored.
- If the tag stays missing past a grace period, the buzzer starts beeping:
    slow -> faster -> fastest -> continuous tone
- If the tag reappears, buzzer stops immediately.

Gesture mapping (swapped for user-facing direction):
Open    -> backward
Close   -> land
Pointer -> up
Four    -> forward
Peace   -> down
YOLO    -> left    (swapped)
El      -> right   (swapped)

Other changes:
- Movement distance increased from 20 to 60
"""

import sys
import os
import time
import cv2
import numpy as np
import socket
import asyncio
import threading

from djitellopy import Tello
from pupil_apriltags import Detector
from bleak import BleakScanner, BleakClient

# ===============================
#  PATH SETUP
# ===============================
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models", "mediapipe_hand"))

from qai_hub_models.models.mediapipe_hand.app import MediaPipeHandApp
from qai_hub_models.models.mediapipe_hand.model import MediaPipeHand
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from model.keypoint_classifier.preprocess import preprocess_landmark, load_gesture_labels

# ===============================
#  BLE BUZZER SETTINGS
# ===============================
DEVICE_NAME = "AEROPET_BLE"
CHAR_UUID = "abcdefab-1234-1234-1234-1234567890ab"

# ===============================
#  APRILTAG SETTINGS
# ===============================
TARGET_TAG_ID = 0
TAG_MISSING_GRACE = 2.0   # seconds before alarm begins

# Seatbelt-style escalation timing
# elapsed missing time after grace:
# 0-3s   -> slow beep
# 3-6s   -> medium beep
# 6-9s   -> fast beep
# 9s+    -> continuous beep
def get_alarm_stage(missing_elapsed: float) -> str:
    if missing_elapsed < TAG_MISSING_GRACE:
        return "off"

    alarm_time = missing_elapsed - TAG_MISSING_GRACE
    if alarm_time < 3.0:
        return "slow"
    elif alarm_time < 6.0:
        return "medium"
    elif alarm_time < 9.0:
        return "fast"
    else:
        return "solid"

# ===============================
#  GESTURE → MOVEMENT MAP
#  swapped left/right + bigger movement
# ===============================
MOVE_DIST = 60

GESTURE_ACTIONS = {
    "Open":    ("move_back", MOVE_DIST),
    "Pointer": ("move_up", MOVE_DIST),
    "Four":    ("move_forward", MOVE_DIST),
    "Peace":   ("move_down", MOVE_DIST),
    "YOLO":    ("move_left", MOVE_DIST),   # swapped
    "El":      ("move_right", MOVE_DIST),  # swapped
    "Close":   ("land", 0),
}

EXPECTED_GESTURES = {
    0: ("Open", "backward"),
    1: ("Close", "land"),
    2: ("Pointer", "up"),
    3: ("Four", "forward"),
    4: ("Peace", "down"),
    5: ("YOLO", "left"),   # swapped
    6: ("El", "right"),    # swapped
}

# ===============================
#  BLE BUZZER CONTROLLER
# ===============================
class BuzzerController:
    """
    Runs BLE writes from a background asyncio loop.
    Public methods are thread-safe:
      - set_stage("off"|"slow"|"medium"|"fast"|"solid")
      - stop()
    """

    def __init__(self, device_name: str, char_uuid: str):
        self.device_name = device_name
        self.char_uuid = char_uuid

        self._loop = None
        self._thread = None
        self._stage = "off"
        self._running = False
        self._lock = threading.Lock()

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()

    def _thread_main(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._run())

    async def _find_device(self):
        print(f"[BLE] Scanning for {self.device_name}...")
        device = await BleakScanner.find_device_by_name(self.device_name, timeout=10.0)
        if device is None:
            print(f"[BLE] Could not find {self.device_name}")
            return None
        print(f"[BLE] Found device: {device.address}")
        return device

    async def _write_safe(self, client, value: int):
        try:
            await client.write_gatt_char(self.char_uuid, bytes([value]), response=True)
        except Exception as e:
            print(f"[BLE] Write failed: {e}")

    async def _run(self):
        device = await self._find_device()
        if device is None:
            print("[BLE] Buzzer disabled because device was not found.")
            while self._running:
                await asyncio.sleep(0.25)
            return

        try:
            async with BleakClient(device) as client:
                print("[BLE] Connected to buzzer")
                await self._write_safe(client, 0)

                while self._running:
                    with self._lock:
                        stage = self._stage

                    if stage == "off":
                        await self._write_safe(client, 0)
                        await asyncio.sleep(0.10)

                    elif stage == "slow":
                        await self._write_safe(client, 1)
                        await asyncio.sleep(0.18)
                        await self._write_safe(client, 0)
                        await asyncio.sleep(0.70)

                    elif stage == "medium":
                        await self._write_safe(client, 1)
                        await asyncio.sleep(0.12)
                        await self._write_safe(client, 0)
                        await asyncio.sleep(0.35)

                    elif stage == "fast":
                        await self._write_safe(client, 1)
                        await asyncio.sleep(0.08)
                        await self._write_safe(client, 0)
                        await asyncio.sleep(0.14)

                    elif stage == "solid":
                        await self._write_safe(client, 1)
                        await asyncio.sleep(0.10)

                await self._write_safe(client, 0)

        except Exception as e:
            print(f"[BLE] Connection error: {e}")

    def set_stage(self, stage: str):
        with self._lock:
            self._stage = stage

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

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
#  APRILTAG DETECTOR
# ===============================
apriltag_detector = Detector(
    families="tag36h11",
    nthreads=2,
    quad_decimate=0.5,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

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

    if pre is None or pre.shape[0] != 42:
        return None, -1

    gid = keypoint_classifier(pre)

    if 0 <= gid < len(gesture_labels):
        return gesture_labels[gid], gid
    return None, -1

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
in_air = True
time.sleep(2)

# ===============================
#  START BUZZER
# ===============================
buzzer = BuzzerController(DEVICE_NAME, CHAR_UUID)
buzzer.start()

# ===============================
#  CONTROL PARAMETERS
# ===============================
STABLE_FRAMES = 3
COMMAND_COOLDOWN = 1.5

gesture_history = []
gesture_id_history = []
last_cmd_time = 0.0

last_tag_seen_time = time.time()

# ===============================
#  UI HELPERS
# ===============================
MAPPING_LINES = [
    "Open=BACK  Close=LAND  Pointer=UP  Four=FWD",
    "Peace=DOWN  YOLO=LEFT  El=RIGHT",
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

        frame = cv2.flip(frame, 1)

        # Tello frame treated as RGB for inference/overlay logic
        frame_rgb = frame.copy()

        # --------------------------------
        # APRILTAG DETECTION
        # --------------------------------
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        tags = apriltag_detector.detect(gray)

        tag_visible = False
        tag_centers = []

        for tag in tags:
            corners = tag.corners.astype(int)
            tag_centers.append(tuple(tag.center.astype(int)))

            if tag.tag_id == TARGET_TAG_ID:
                tag_visible = True

        now = time.time()
        if tag_visible:
            last_tag_seen_time = now

        missing_elapsed = now - last_tag_seen_time
        alarm_stage = get_alarm_stage(missing_elapsed)

        if tag_visible:
            buzzer.set_stage("off")
        else:
            buzzer.set_stage(alarm_stage)

        # --------------------------------
        # GESTURE DETECTION
        # Only matters if authorized tag visible
        # --------------------------------
        raw = app.predict_landmarks_from_image(frame_rgb, raw_output=True)
        landmarks_batch = raw[3]

        current_gesture = None
        gesture_id = -1

        for t in landmarks_batch:
            if t.nelement() > 0:
                lm = t.cpu().numpy()[0]
                current_gesture, gesture_id = classify_gesture(lm)
                break

        if tag_visible and current_gesture:
            gesture_history.append(current_gesture)
            gesture_id_history.append(gesture_id)

            if len(gesture_history) > STABLE_FRAMES:
                gesture_history.pop(0)

            if len(gesture_id_history) > 5:
                gesture_id_history.pop(0)
        else:
            # no tag -> no authorization -> no gesture control
            gesture_history.clear()
            gesture_id_history.clear()

        stable = None
        if len(gesture_history) == STABLE_FRAMES and all(g == gesture_history[0] for g in gesture_history):
            stable = gesture_history[0]

        # --------------------------------
        # DISPLAY
        # --------------------------------
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Draw AprilTags
        for tag in tags:
            corners = tag.corners.astype(int)
            color = (0, 255, 0) if tag.tag_id == TARGET_TAG_ID else (0, 200, 255)

            for i in range(4):
                p1 = tuple(corners[i])
                p2 = tuple(corners[(i + 1) % 4])
                cv2.line(frame_bgr, p1, p2, color, 2)

            c = tuple(tag.center.astype(int))
            label = f"ID {tag.tag_id}"
            if tag.tag_id == TARGET_TAG_ID:
                label += " (AUTHORIZED)"
            cv2.putText(
                frame_bgr,
                label,
                c,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2
            )

        next_action = action_preview_from_stable(stable)
        auth_text = "AUTHORIZED" if tag_visible else "NOT AUTHORIZED"

        cv2.putText(
            frame_bgr,
            f"Tag: {auth_text} | Missing: {missing_elapsed:.1f}s | Alarm: {alarm_stage}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0) if tag_visible else (0, 0, 255),
            2,
        )

        cv2.putText(
            frame_bgr,
            f"Gesture ID: {gesture_id} | Label: {current_gesture} | Stable: {stable} | Next: {next_action}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 255) if tag_visible else (120, 120, 120),
            2,
        )

        recent_ids_str = " ".join(str(i) for i in gesture_id_history)
        cv2.putText(
            frame_bgr,
            f"Recent IDs: {recent_ids_str}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 0) if tag_visible else (120, 120, 120),
            2,
        )

        if not tag_visible:
            cv2.putText(
                frame_bgr,
                "GESTURES LOCKED: authorized AprilTag not visible",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 0, 255),
                2,
            )

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

        cv2.imshow("Tello Gesture + AprilTag Auth", frame_bgr)

        # --------------------------------
        # EXECUTE COMMAND
        # only if tag visible
        # --------------------------------
        now = time.time()
        if tag_visible and stable and (now - last_cmd_time > COMMAND_COOLDOWN):
            action = GESTURE_ACTIONS.get(stable)
            if action:
                cmd, val = action
                print(f"EXECUTE -> {stable.upper()} => {cmd} {val if cmd != 'land' else ''}".strip())

                if cmd == "land":
                    buzzer.set_stage("off")
                    tello.land()
                    in_air = False
                    break
                else:
                    getattr(tello, cmd)(val)

                last_cmd_time = now
                gesture_history.clear()
                gesture_id_history.clear()

        # Quit key
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

except KeyboardInterrupt:
    pass

finally:
    print("\nCLEANUP...")

    try:
        buzzer.set_stage("off")
        buzzer.stop()
    except Exception:
        pass

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

    try:
        tello.end()
    except Exception:
        pass

    cv2.destroyAllWindows()
    print("DONE.")