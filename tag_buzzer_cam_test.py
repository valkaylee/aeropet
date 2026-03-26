#!/usr/bin/env python3
"""
Tello Camera-Only Gesture + AprilTag Authorization + BLE Buzzer Alarm

Behavior:
- Uses Tello camera only (NO TAKEOFF / NO FLIGHT / NO MOVEMENT)
- Gestures are only considered "authorized" when the target AprilTag is visible
- If the tag is not visible, gestures are ignored
- If the tag stays missing past a grace period, the buzzer starts beeping:
    slow -> faster -> fastest -> continuous tone
- If the tag reappears, buzzer stops immediately
- Shows what command WOULD be executed, but does not actually move the drone

Gesture mapping (user-facing direction):
Open    -> backward
Close   -> land
Pointer -> up
Four    -> forward
Peace   -> down
YOLO    -> left
El      -> right
"""

import sys
import os
import time
import cv2
import numpy as np
import asyncio
import threading

from bleak import BleakScanner, BleakClient

# ===============================
#  BLE BUZZER SETTINGS
# ===============================
DEVICE_NAME = "AEROPET_BLE"
CHAR_UUID = "abcdefab-1234-1234-1234-1234567890ab"

# ===============================
#  BLE BUZZER CONTROLLER
# ===============================
class BuzzerController:
    """
    Runs BLE writes from a background asyncio loop.
    Retries scan indefinitely until device is found.
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
        while self._running:
            print(f"[BLE] Scanning for {self.device_name}...")
            device = await BleakScanner.find_device_by_name(self.device_name, timeout=10.0)
            if device is not None:
                print(f"[BLE] Found device: {device.address}")
                return device
            print(f"[BLE] Not found, retrying...")
        return None

    async def _write_safe(self, client, value: int):
        try:
            await client.write_gatt_char(self.char_uuid, bytes([value]), response=True)
        except Exception as e:
            print(f"[BLE] Write failed: {e}")

    async def _run(self):
        while self._running:
            device = await self._find_device()
            if device is None:
                return  # only happens if stop() was called

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
                print(f"[BLE] Connection error: {e} — retrying...")
                await asyncio.sleep(2.0)  # brief pause before reconnect attempt

    def set_stage(self, stage: str):
        with self._lock:
            self._stage = stage

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

# ===============================
#  START BUZZER FIRST
#  (retries in background while everything else loads)
# ===============================
buzzer = BuzzerController(DEVICE_NAME, CHAR_UUID)
buzzer.start()
print("[BLE] Scan started in background, will keep retrying until found...")

# ===============================
#  PATH SETUP
# ===============================
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models", "mediapipe_hand"))

from djitellopy import Tello
from pupil_apriltags import Detector
from qai_hub_models.models.mediapipe_hand.app import MediaPipeHandApp
from qai_hub_models.models.mediapipe_hand.model import MediaPipeHand
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from model.keypoint_classifier.preprocess import preprocess_landmark, load_gesture_labels

# ===============================
#  APRILTAG SETTINGS
# ===============================
TARGET_TAG_ID = 0
TAG_MISSING_GRACE = 2.0   # seconds before alarm begins

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
# ===============================
MOVE_DIST = 60

GESTURE_ACTIONS = {
    "Open":    ("move_back", MOVE_DIST),
    "Pointer": ("move_up", MOVE_DIST),
    "Four":    ("move_forward", MOVE_DIST),
    "Peace":   ("move_down", MOVE_DIST),
    "YOLO":    ("move_left", MOVE_DIST),
    "El":      ("move_right", MOVE_DIST),
    "Close":   ("land", 0),
}

EXPECTED_GESTURES = {
    0: ("Open", "backward"),
    1: ("Close", "land"),
    2: ("Pointer", "up"),
    3: ("Four", "forward"),
    4: ("Peace", "down"),
    5: ("YOLO", "left"),
    6: ("El", "right"),
}

# ===============================
#  LOAD MODELS
# ===============================
print("\nLoading MediaPipe Hand model...")
model = MediaPipeHand.from_pretrained()
app = MediaPipeHandApp.from_pretrained(model)
app.min_detector_box_score = 0.3
print("✓ MediaPipe Hand model loaded")

print("Loading gesture classifier...")
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, "models", "mediapipe_hand", "model", "keypoint_classifier")

keypoint_classifier = KeyPointClassifier(
    model_path=os.path.join(model_dir, "keypoint_classifier.tflite")
)
gesture_labels = [
    g.strip().lstrip("\ufeff")
    for g in load_gesture_labels(os.path.join(model_dir, "keypoint_classifier_label.csv"))
]
print(f"✓ Gesture classifier loaded. Labels: {gesture_labels}")

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
    if landmarks is None or landmarks.shape != (21, 3):
        return None, -1, "Wrong shape"

    try:
        pre = preprocess_landmark(landmarks)

        if pre is None or pre.shape[0] != 42:
            return None, -1, f"Bad preprocessed shape: {None if pre is None else pre.shape}"

        gid = keypoint_classifier(pre)

        if 0 <= gid < len(gesture_labels):
            return gesture_labels[gid], gid, "OK"
        return None, gid, f"Invalid ID: {gid}"
    except Exception as e:
        return None, -1, f"Error: {e}"

# ===============================
#  CONNECT TO TELLO (CAMERA ONLY)
# ===============================
print("\nInitializing Tello connection...")
print("⚠️  Make sure you are connected to the Tello WiFi")
print("⚠️  CAMERA ONLY: no takeoff, no movement")

tello = Tello()
tello.connect()
battery = tello.get_battery()
print(f"✓ Connected to Tello. Battery: {battery}%")

tello.streamoff()
tello.streamon()
time.sleep(2)

frame_reader = tello.get_frame_read()
print("✓ Video stream started")

# ===============================
#  CONTROL PARAMETERS
# ===============================
STABLE_FRAMES = 3
COMMAND_COOLDOWN = 1.5

gesture_history = []
gesture_id_history = []
last_cmd_time = 0.0
last_tag_seen_time = time.time()
frame_count = 0
last_status_print = 0.0
last_preview = ""

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
        frame_count += 1

        if frame is None or frame.size == 0:
            continue

        frame = cv2.flip(frame, 1)
        frame_rgb = frame.copy()

        # --------------------------------
        # APRILTAG DETECTION
        # --------------------------------
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        tags = apriltag_detector.detect(gray)

        tag_visible = False
        for tag in tags:
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
        # --------------------------------
        raw = app.predict_landmarks_from_image(frame_rgb, raw_output=True)
        landmarks_batch = raw[3]

        current_gesture = None
        gesture_id = -1
        gesture_info = ""
        hands_detected = 0

        for t in landmarks_batch:
            if t.nelement() > 0:
                lm_batch = t.cpu().numpy()
                hands_detected += lm_batch.shape[0]

                if lm_batch.shape[0] > 0:
                    lm = lm_batch[0]
                    current_gesture, gesture_id, gesture_info = classify_gesture(lm)
                    break

        if tag_visible and current_gesture:
            gesture_history.append(current_gesture)
            gesture_id_history.append(gesture_id)

            if len(gesture_history) > STABLE_FRAMES:
                gesture_history.pop(0)

            if len(gesture_id_history) > 5:
                gesture_id_history.pop(0)
        else:
            gesture_history.clear()
            gesture_id_history.clear()

        stable = None
        if len(gesture_history) == STABLE_FRAMES and all(g == gesture_history[0] for g in gesture_history):
            stable = gesture_history[0]

        next_action = action_preview_from_stable(stable)

        # --------------------------------
        # CAMERA-ONLY "WOULD EXECUTE"
        # --------------------------------
        preview_text = ""
        if tag_visible and stable and (now - last_cmd_time > COMMAND_COOLDOWN):
            action = GESTURE_ACTIONS.get(stable)
            if action:
                cmd, val = action
                if cmd == "land":
                    preview_text = f"WOULD EXECUTE -> {stable.upper()} => LAND"
                else:
                    preview_text = f"WOULD EXECUTE -> {stable.upper()} => {cmd} {val}"

                if preview_text != last_preview:
                    print(preview_text)
                    last_preview = preview_text

                last_cmd_time = now
                gesture_history.clear()
                gesture_id_history.clear()

        # periodic console status
        if now - last_status_print > 2.0:
            auth_text = "AUTHORIZED" if tag_visible else "NOT AUTHORIZED"
            print(
                f"[Frame {frame_count:4d}] "
                f"Hands: {hands_detected} | "
                f"Tag: {auth_text} | "
                f"Missing: {missing_elapsed:.1f}s | "
                f"Alarm: {alarm_stage} | "
                f"Gesture ID: {gesture_id} | "
                f"Label: {current_gesture}"
            )
            last_status_print = now

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
            cv2.putText(frame_bgr, label, c, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        auth_text = "AUTHORIZED" if tag_visible else "NOT AUTHORIZED"

        cv2.putText(frame_bgr, f"Frame: {frame_count} | Hands: {hands_detected}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)

        cv2.putText(frame_bgr, f"Tag: {auth_text} | Missing: {missing_elapsed:.1f}s | Alarm: {alarm_stage}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (0, 255, 0) if tag_visible else (0, 0, 255), 2)

        cv2.putText(frame_bgr, f"Gesture ID: {gesture_id} | Label: {current_gesture} | Stable: {stable}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (0, 255, 255) if tag_visible else (120, 120, 120), 2)

        cv2.putText(frame_bgr, f"Next: {next_action}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (255, 255, 0) if tag_visible else (120, 120, 120), 2)

        recent_ids_str = " ".join(str(i) for i in gesture_id_history)
        cv2.putText(frame_bgr, f"Recent IDs: {recent_ids_str}",
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (255, 255, 0) if tag_visible else (120, 120, 120), 2)

        if gesture_info and gesture_info != "OK":
            cv2.putText(frame_bgr, f"Info: {gesture_info}",
                        (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

        if not tag_visible:
            cv2.putText(frame_bgr, "GESTURES LOCKED: authorized AprilTag not visible",
                        (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 255), 2)

        if preview_text:
            cv2.putText(frame_bgr, preview_text,
                        (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2)

        if gesture_id in EXPECTED_GESTURES:
            name, direction = EXPECTED_GESTURES[gesture_id]
            cv2.putText(frame_bgr, f"EXPECTED: {name} -> {direction.upper()} (ID={gesture_id})",
                        (10, frame_bgr.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        base_y = frame_bgr.shape[0] - 30
        for j, line in enumerate(MAPPING_LINES[::-1]):
            cv2.putText(frame_bgr, line, (10, base_y - j * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.imshow("Tello Camera-Only: Gesture + AprilTag + Buzzer", frame_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("\nExiting...")
            break

except KeyboardInterrupt:
    print("\nCtrl+C - Exiting")

finally:
    print("\nCLEANUP...")

    try:
        buzzer.set_stage("off")
        buzzer.stop()
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
    print(f"✓ Done - Processed {frame_count} frames")