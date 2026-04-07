#!/usr/bin/env python3
"""
AeroPet Gesture + AprilTag Authorization + BLE Buzzer Alarm
REAL FLIGHT VERSION

Behavior:
- Waits for BLE buzzer to connect BEFORE loading models / camera code
- Connects to Tello, starts video, and TAKES OFF automatically
- After takeoff, flies up to ~5ft command height
- Buzzer beeps twice fast when drone is ready to receive commands
- AprilTag countdown starts AFTER ready beeps
- Keeps gesture detection close to the original diagnostic script
- AprilTag gates command authorization (raw gesture detection still runs always)
- If the tag stays missing past a grace period, the buzzer starts beeping:
    slow -> medium -> fast -> continuous tone
- At the START of slow / medium / fast, the drone performs a quick
  look-around sequence:
    rotate 30 deg CCW -> center -> 30 deg CW -> center
- If the tag reappears, buzzer stops immediately
- Executes real flight commands when:
    1) the authorized AprilTag is visible
    2) the gesture is stable for STABLE_FRAMES
    3) command cooldown has passed
- Drone hovers in place when no gesture is detected

Changes from previous version:
- Slow beep cycle slowed down (~1.15s total vs ~0.6s)
- After takeoff, move_up 90cm to reach ~5ft command height
- Buzzer beeps twice fast after reaching height = drone ready signal
- AprilTag countdown (last_tag_seen_time) starts AFTER ready beeps
- STABLE_FRAMES reduced from 5 to 4
- look_around no longer fires on "solid" stage (only slow/medium/fast)
- debug_mode spam reduced (every 60 frames instead of every frame w/ no gesture)

Safety / flight style:
- Auto takeoff after startup
- No t/e/q flight control keys
- Close gesture lands the drone
- q only exits the program window; cleanup will land if still airborne
- Drone hovers automatically when no command is issued
"""

import sys
import os
import time
import cv2
import numpy as np
import asyncio
import threading
import socket

from bleak import BleakScanner, BleakClient

# ===============================
#  BLE BUZZER SETTINGS
# ===============================
DEVICE_NAME = "AEROPET_BLE"
CHAR_UUID = "abcdefab-1234-1234-1234-1234567890ab"
BLE_CONNECT_TIMEOUT = 60.0

# ===============================
#  FLIGHT SETTINGS
# ===============================
AUTO_TAKEOFF = True
MIN_BATTERY_FOR_FLIGHT = 20
POST_TAKEOFF_HOVER_SEC = 2.0
MOVE_DIST = 50
COMMAND_HEIGHT_CM = 60       # additional lift after takeoff to reach ~5ft
COMMAND_HEIGHT_SETTLE = 1.5  # seconds to stabilize after rising
LOOK_AROUND_ANGLE = 30
ROTATE_SETTLE_SEC = 0.35
POST_LAND_DELAY_SEC = 1.0

# ===============================
#  APRILTAG SETTINGS
# ===============================
TARGET_TAG_ID = 0
TAG_MISSING_GRACE = 10.0
EVENT_TEXT_DURATION = 3.0

# ===============================
#  CONTROL PARAMETERS
# ===============================
STABLE_FRAMES = 4            # was 5, 4 frames ~133ms at 30fps
COMMAND_COOLDOWN = 1.5
MAX_GESTURE_HISTORY = 10
MAX_ID_HISTORY = 5


# ===============================
#  BLE BUZZER CONTROLLER
# ===============================
class BuzzerController:
    """
    Runs BLE writes from a background asyncio loop.
    Startup blocks until the first connection succeeds.
    Supports a one-shot ready beep (two fast beeps) triggered externally.
    """

    def __init__(self, device_name: str, char_uuid: str):
        self.device_name = device_name
        self.char_uuid = char_uuid

        self._loop = None
        self._thread = None
        self._stage = "off"
        self._running = False
        self._lock = threading.Lock()
        self._connected_event = threading.Event()
        self._do_ready_beep = False   # one-shot flag for startup ready signal

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()

    def wait_until_connected(self, timeout=None) -> bool:
        return self._connected_event.wait(timeout=timeout)

    def trigger_ready_beep(self):
        """Fire two fast beeps on next BLE loop iteration."""
        with self._lock:
            self._do_ready_beep = True

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
            print("[BLE] Not found, retrying...")
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
                return

            try:
                async with BleakClient(device) as client:
                    self._connected_event.set()
                    print("[BLE] Connected to buzzer")
                    await self._write_safe(client, 0)

                    while self._running:
                        # Check for one-shot ready beep first
                        with self._lock:
                            do_beep = self._do_ready_beep
                            if do_beep:
                                self._do_ready_beep = False

                        if do_beep:
                            # Two fast beeps = drone ready signal
                            for _ in range(2):
                                await self._write_safe(client, 1)
                                await asyncio.sleep(0.08)
                                await self._write_safe(client, 0)
                                await asyncio.sleep(0.15)
                            continue  # skip normal stage logic this cycle

                        with self._lock:
                            stage = self._stage

                        if stage == "off":
                            await self._write_safe(client, 0)
                            await asyncio.sleep(0.10)

                        elif stage == "slow":
                            # Slowed down: ~1.15s total cycle (was ~0.6s)
                            await self._write_safe(client, 1)
                            await asyncio.sleep(0.15)
                            await self._write_safe(client, 0)
                            await asyncio.sleep(1.0)

                        elif stage == "medium":
                            await self._write_safe(client, 1)
                            await asyncio.sleep(0.12)
                            await self._write_safe(client, 0)
                            await asyncio.sleep(0.22)

                        elif stage == "fast":
                            await self._write_safe(client, 1)
                            await asyncio.sleep(0.08)
                            await self._write_safe(client, 0)
                            await asyncio.sleep(0.10)

                        elif stage == "solid":
                            await self._write_safe(client, 1)
                            await asyncio.sleep(0.10)

                    await self._write_safe(client, 0)

            except Exception as e:
                self._connected_event.clear()
                print(f"[BLE] Connection error: {e} — retrying...")
                await asyncio.sleep(2.0)

    def set_stage(self, stage: str):
        with self._lock:
            self._stage = stage

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)


# ===============================
#  HELPERS
# ===============================
def get_alarm_stage(missing_elapsed: float) -> str:
    if missing_elapsed < TAG_MISSING_GRACE:
        return "off"

    alarm_time = missing_elapsed - TAG_MISSING_GRACE
    if alarm_time < 5.0:
        return "slow"
    elif alarm_time < 10.0:
        return "medium"
    elif alarm_time < 15.0:
        return "fast"
    else:
        return "solid"


def get_stage_scan_text(stage: str) -> str:
    return (
        f"ALARM {stage.upper()} START -> LOOK AROUND: "
        f"ROTATE CCW {LOOK_AROUND_ANGLE}, CENTER, "
        f"ROTATE CW {LOOK_AROUND_ANGLE}, CENTER"
    )


def safe_tello_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"[TELLO] Command failed: {e}")
        return None


def look_around_sequence(tello):
    """
    Scan sequence to search for the tag.
    Only fires on slow/medium/fast alarm stages (NOT solid).
    Blocks the main loop — keep ROTATE_SETTLE_SEC low.
    """
    print(f"[TELLO] LOOK AROUND -> CCW {LOOK_AROUND_ANGLE}, CENTER, CW {LOOK_AROUND_ANGLE}, CENTER")

    safe_tello_call(tello.rotate_counter_clockwise, LOOK_AROUND_ANGLE)
    time.sleep(ROTATE_SETTLE_SEC)

    safe_tello_call(tello.rotate_clockwise, LOOK_AROUND_ANGLE)
    time.sleep(ROTATE_SETTLE_SEC)

    safe_tello_call(tello.rotate_clockwise, LOOK_AROUND_ANGLE)
    time.sleep(ROTATE_SETTLE_SEC)

    safe_tello_call(tello.rotate_counter_clockwise, LOOK_AROUND_ANGLE)
    time.sleep(ROTATE_SETTLE_SEC)


# ===============================
#  START BUZZER FIRST, AND WAIT FOR IT
# ===============================
buzzer = BuzzerController(DEVICE_NAME, CHAR_UUID)
buzzer.start()
print("[BLE] Waiting for buzzer to connect before starting camera/model code...")
start_wait = time.time()
while not buzzer.wait_until_connected(timeout=1.0):
    elapsed = time.time() - start_wait
    print(f"[BLE] Still waiting for buzzer connection... ({elapsed:.0f}s)")
    if elapsed > BLE_CONNECT_TIMEOUT:
        raise RuntimeError("BLE buzzer did not connect within timeout. Check power and advertising.")
print("[BLE] Buzzer connected. Continuing startup...")

# ===============================
#  PATH SETUP / IMPORTS
# ===============================
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models", "mediapipe_hand"))

from djitellopy import Tello
from pupil_apriltags import Detector
from qai_hub_models.models.mediapipe_hand.app import MediaPipeHandApp
from qai_hub_models.models.mediapipe_hand.model import MediaPipeHand
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from model.keypoint_classifier.preprocess import preprocess_landmark, load_gesture_labels

# ===============================
#  GESTURE -> MOVEMENT MAP
#  User-relative: YOLO = user's right hand = drone moves left
#                 El   = user's left hand  = drone moves right
# ===============================
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
model_path = os.path.join(model_dir, "keypoint_classifier.tflite")
label_path = os.path.join(model_dir, "keypoint_classifier_label.csv")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
if not os.path.exists(label_path):
    raise FileNotFoundError(f"Label file not found at {label_path}")

keypoint_classifier = KeyPointClassifier(model_path=model_path)
gesture_labels = [label.strip().lstrip("\ufeff") for label in load_gesture_labels(label_path)]
print("✓ Gesture classifier loaded.")
print(f"  Label mapping: {dict(enumerate(gesture_labels))}")

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
    debug=0,
)

# ===============================
#  CLASSIFY GESTURE
# ===============================
def classify_gesture(landmarks: np.ndarray, debug: bool = False):
    if landmarks is None or landmarks.shape != (21, 3):
        return ("UNKNOWN", -1, "Wrong shape")

    try:
        if debug:
            print(
                f"      Landmark ranges: X=[{landmarks[:, 0].min():.3f}, {landmarks[:, 0].max():.3f}], "
                f"Y=[{landmarks[:, 1].min():.3f}, {landmarks[:, 1].max():.3f}], "
                f"Z=[{landmarks[:, 2].min():.3f}, {landmarks[:, 2].max():.3f}]"
            )

        preprocessed_landmarks = preprocess_landmark(landmarks)
        if preprocessed_landmarks is None or preprocessed_landmarks.shape[0] != 42:
            return (
                "UNKNOWN",
                -1,
                f"Wrong preprocessed shape: {None if preprocessed_landmarks is None else preprocessed_landmarks.shape}",
            )

        if debug:
            print(
                f"      Preprocessed range: [{preprocessed_landmarks.min():.3f}, {preprocessed_landmarks.max():.3f}]"
            )

        gesture_id = keypoint_classifier(preprocessed_landmarks)
        if 0 <= gesture_id < len(gesture_labels):
            label = gesture_labels[gesture_id].strip().lstrip("\ufeff")
            return (label, gesture_id, "OK")
        return ("UNKNOWN", gesture_id, f"Invalid ID: {gesture_id}")

    except Exception as e:
        return ("UNKNOWN", -1, f"Error: {e}")


# ===============================
#  CONNECT TO TELLO / AUTO TAKEOFF
# ===============================
print("\nInitializing Tello connection...")
print("⚠️  Make sure you are connected to the Tello WiFi")

print("Testing network connectivity to Tello (192.168.10.1)...")
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(2)
sock.connect(("192.168.10.1", 8889))
sock.close()
print("✓ Network connectivity test passed")

tello = Tello()
tello.connect()
battery = tello.get_battery()
print(f"✓ Connected to Tello. Battery: {battery}%")

if battery < MIN_BATTERY_FOR_FLIGHT:
    raise RuntimeError(f"Battery too low for flight: {battery}%")

safe_tello_call(tello.streamoff)
safe_tello_call(tello.streamon)
time.sleep(2)
frame_reader = tello.get_frame_read()
print("✓ Video stream started")

in_air = False
if AUTO_TAKEOFF:
    print("\nTAKING OFF...")
    safe_tello_call(tello.takeoff)
    in_air = True
    time.sleep(POST_TAKEOFF_HOVER_SEC)

    # Fly up to ~5ft command height (takeoff ~2ft + 90cm ~3ft = ~5ft)
    print(f"Flying up to command height (~5ft, +{COMMAND_HEIGHT_CM}cm)...")
    safe_tello_call(tello.move_up, COMMAND_HEIGHT_CM)
    time.sleep(COMMAND_HEIGHT_SETTLE)

    # Two fast beeps = drone is ready to receive commands
    print("Signaling ready (2 beeps)...")
    buzzer.trigger_ready_beep()
    time.sleep(0.8)  # let beeps finish before loop starts

    # AprilTag countdown starts NOW (not at program start)
    last_tag_seen_time = time.time()
    print("✓ Drone ready. AprilTag countdown started.")

# ===============================
#  STATE
# ===============================
raw_gesture_history = []
stable_label_window = []
last_cmd_time = 0.0
# NOTE: last_tag_seen_time is set above after ready beeps.
# If AUTO_TAKEOFF is False, initialize it here as a fallback.
if not AUTO_TAKEOFF:
    last_tag_seen_time = time.time()

frame_count = 0
last_alarm_stage = "off"
last_event_text = ""
event_text = ""
event_text_until = 0.0
last_raw_gesture = None
last_raw_print_time = 0.0
just_landed = False

# ===============================
#  UI HELPERS
# ===============================
MAPPING_LINES = [
    "Open=BACK  Close=LAND  Pointer=UP  Four=FWD",
    "Peace=DOWN  YOLO=LEFT(user)  El=RIGHT(user)",
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
        if len(frame.shape) != 3 or frame.shape[0] == 0 or frame.shape[1] == 0:
            continue

        frame = cv2.flip(frame, 1)
        frame_rgb = frame.copy()

        # --------------------------------
        # APRILTAG DETECTION
        # --------------------------------
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        tags = apriltag_detector.detect(gray)

        tag_visible = any(tag.tag_id == TARGET_TAG_ID for tag in tags)
        now = time.time()
        if tag_visible:
            last_tag_seen_time = now

        missing_elapsed = now - last_tag_seen_time
        alarm_stage = get_alarm_stage(missing_elapsed)
        buzzer.set_stage("off" if tag_visible else alarm_stage)

        # Stage transition: look-around only on slow/medium/fast (NOT solid)
        if alarm_stage != last_alarm_stage:
            if alarm_stage in {"slow", "medium", "fast"}:
                event_text = get_stage_scan_text(alarm_stage)
                event_text_until = now + EVENT_TEXT_DURATION
                if event_text != last_event_text:
                    print(event_text)
                    last_event_text = event_text

                if in_air and not just_landed:
                    look_around_sequence(tello)

            elif alarm_stage == "solid":
                # Solid = continuous beep only, no more rotations
                event_text = "ALARM SOLID: continuous beep, hovering in place"
                event_text_until = now + EVENT_TEXT_DURATION
                print(event_text)

            elif alarm_stage == "off" and last_alarm_stage != "off":
                event_text = "ALARM OFF -> tag restored, STOP SEARCH / BUZZER"
                event_text_until = now + EVENT_TEXT_DURATION
                if event_text != last_event_text:
                    print(event_text)
                    last_event_text = event_text

            last_alarm_stage = alarm_stage

        if now > event_text_until:
            event_text = ""

        # --------------------------------
        # GESTURE DETECTION
        # --------------------------------
        try:
            raw_result = app.predict_landmarks_from_image(frame_rgb, raw_output=True)
            batched_selected_landmarks = raw_result[3]
        except Exception as e:
            if frame_count % 30 == 0:
                print(f"[Frame {frame_count}] Error processing frame: {e}")
            continue

        current_gesture = None
        gesture_id = -1
        gesture_info = ""
        hands_detected = 0

        for landmarks_tensor in batched_selected_landmarks:
            if landmarks_tensor.nelement() != 0:
                landmarks_np = landmarks_tensor.cpu().numpy()
                hands_detected += landmarks_np.shape[0]

                if landmarks_np.shape[0] > 0:
                    hand_landmarks = landmarks_np[0]
                    if hand_landmarks.shape == (21, 3):
                        # Reduced debug spam: only every 60 frames
                        debug_mode = (frame_count % 60 == 0)
                        current_gesture, gesture_id, gesture_info = classify_gesture(
                            hand_landmarks, debug=debug_mode
                        )
                        break

        if current_gesture and current_gesture != "UNKNOWN":
            raw_gesture_history.append((current_gesture, gesture_id))
            if len(raw_gesture_history) > MAX_GESTURE_HISTORY:
                raw_gesture_history.pop(0)

            stable_label_window.append(current_gesture)
            if len(stable_label_window) > STABLE_FRAMES:
                stable_label_window.pop(0)
        else:
            if hands_detected == 0:
                # No hand at all: clear everything, drone will hover
                raw_gesture_history.clear()
                stable_label_window.clear()
            elif gesture_info != "OK":
                stable_label_window.clear()

        stable = None
        if len(stable_label_window) == STABLE_FRAMES and all(
            g == stable_label_window[0] for g in stable_label_window
        ):
            stable = stable_label_window[0]

        next_action = action_preview_from_stable(stable) if tag_visible else "LOCKED (tag missing)"

        # --------------------------------
        # EXECUTE REAL COMMAND
        # (only when: in air + tag visible + stable gesture + cooldown passed)
        # If no gesture detected, drone hovers automatically — no action needed
        # --------------------------------
        preview_text = ""
        if in_air and tag_visible and stable and (now - last_cmd_time > COMMAND_COOLDOWN):
            action = GESTURE_ACTIONS.get(stable)
            if action:
                cmd, val = action

                if cmd == "land":
                    preview_text = f"EXECUTING -> {stable.upper()} => LAND"
                    print(preview_text)
                    safe_tello_call(tello.land)
                    in_air = False
                    just_landed = True
                    time.sleep(POST_LAND_DELAY_SEC)
                    break
                else:
                    preview_text = f"EXECUTING -> {stable.upper()} => {cmd} {val}"
                    print(preview_text)
                    safe_tello_call(getattr(tello, cmd), val)

                last_cmd_time = time.time()
                stable_label_window.clear()

        # --------------------------------
        # STATUS PRINT
        # --------------------------------
        if current_gesture != last_raw_gesture or (now - last_raw_print_time > 2.0):
            gesture_str = current_gesture if current_gesture else "None"
            print(
                f"[Frame {frame_count:4d}] Hands: {hands_detected} | "
                f"Tag: {'AUTHORIZED' if tag_visible else 'NOT AUTHORIZED'} | "
                f"Missing: {missing_elapsed:.1f}s | Alarm: {alarm_stage} | "
                f"Gesture ID: {gesture_id:2d} | Label: '{gesture_str}' | "
                f"Stable: {stable} | Info: {gesture_info}"
            )
            last_raw_gesture = current_gesture
            last_raw_print_time = now

            if raw_gesture_history:
                recent = raw_gesture_history[-5:]
                ids = [g[1] for g in recent]
                labels = [g[0] for g in recent]
                print(f"         Recent history: IDs={ids} | Labels={labels}")

        # --------------------------------
        # DISPLAY
        # --------------------------------
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        for tag in tags:
            corners = tag.corners.astype(int)
            color = (0, 255, 0) if tag.tag_id == TARGET_TAG_ID else (0, 200, 255)
            for i in range(4):
                p1 = tuple(corners[i])
                p2 = tuple(corners[(i + 1) % 4])
                cv2.line(frame_bgr, p1, p2, color, 2)
            c = tuple(tag.center.astype(int))
            tag_label = f"ID {tag.tag_id}"
            if tag.tag_id == TARGET_TAG_ID:
                tag_label += " (AUTHORIZED)"
            cv2.putText(frame_bgr, tag_label, c, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        auth_text = "AUTHORIZED" if tag_visible else "NOT AUTHORIZED"
        cv2.putText(frame_bgr, f"Frame: {frame_count} | Hands: {hands_detected}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)

        cv2.putText(frame_bgr,
                    f"Flight: {'IN AIR' if in_air else 'LANDED'} | Tag: {auth_text} | Missing: {missing_elapsed:.1f}s | Alarm: {alarm_stage}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.56,
                    (0, 255, 0) if tag_visible else (0, 0, 255), 2)

        cv2.putText(frame_bgr, f"RAW Gesture ID: {gesture_id} | Label: {current_gesture}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (0, 255, 255) if current_gesture and current_gesture != "UNKNOWN" else (120, 120, 120), 2)

        cv2.putText(frame_bgr, f"Stable ({STABLE_FRAMES}f): {stable}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (255, 255, 0) if stable else (120, 120, 120), 2)

        cv2.putText(frame_bgr, f"Next: {next_action}",
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (255, 255, 0) if tag_visible else (120, 120, 120), 2)

        if raw_gesture_history:
            recent_ids_str = " ".join(str(i) for _, i in raw_gesture_history[-MAX_ID_HISTORY:])
            cv2.putText(frame_bgr, f"Recent IDs: {recent_ids_str}",
                        (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        (255, 255, 0), 2)

        if gesture_info and gesture_info != "OK":
            cv2.putText(frame_bgr, f"Info: {gesture_info}",
                        (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

        if not tag_visible:
            cv2.putText(frame_bgr, "COMMANDS LOCKED: authorized AprilTag not visible",
                        (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 255), 2)

        if preview_text:
            cv2.putText(frame_bgr, preview_text,
                        (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2)

        if event_text:
            cv2.putText(frame_bgr, event_text,
                        (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 200, 0), 2)

        if gesture_id in EXPECTED_GESTURES:
            name, direction = EXPECTED_GESTURES[gesture_id]
            cv2.putText(frame_bgr, f"EXPECTED: {name} -> {direction.upper()} (ID={gesture_id})",
                        (10, frame_bgr.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        base_y = frame_bgr.shape[0] - 30
        for j, line in enumerate(MAPPING_LINES[::-1]):
            cv2.putText(frame_bgr, line, (10, base_y - j * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.imshow("AeroPet Real Flight: Gesture + AprilTag + Buzzer", frame_bgr)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
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
        if in_air:
            print("[TELLO] Landing during cleanup...")
            safe_tello_call(tello.land)
            in_air = False
    except Exception:
        pass

    try:
        safe_tello_call(tello.streamoff)
    except Exception:
        pass

    try:
        safe_tello_call(tello.end)
    except Exception:
        pass

    cv2.destroyAllWindows()
    print(f"✓ Done - Processed {frame_count} frames")