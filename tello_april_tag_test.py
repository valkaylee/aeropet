import cv2
from djitellopy import Tello
from pupil_apriltags import Detector

def main():
    tello = Tello()

    # Connect to the drone (this does NOT make it fly)
    tello.connect()
    print("Connected. Battery:", tello.get_battery(), "%")

    # Start camera stream
    tello.streamoff()  # safe reset in case it's already on
    tello.streamon()
    frame_reader = tello.get_frame_read()

    # AprilTag detector
    # NOTE: quad_decimate=0.5 can help with very small tags (like 20mm) at the cost of FPS.
    detector = Detector(
        families="tag36h11",
        nthreads=2,
        quad_decimate=0.5,   # use 1.0 if you want more FPS
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )

    try:
        while True:
            raw = frame_reader.frame
            if raw is None:
                continue

            # djitellopy frames are typically RGB; OpenCV display expects BGR
            frame = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)

            # Detect tags on grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tags = detector.detect(gray)

            # Draw detections
            for tag in tags:
                corners = tag.corners.astype(int)

                # Draw bounding box
                for i in range(4):
                    p1 = tuple(corners[i])
                    p2 = tuple(corners[(i + 1) % 4])
                    cv2.line(frame, p1, p2, (0, 255, 0), 2)

                # Label
                c = tuple(tag.center.astype(int))
                label = f"ID {tag.tag_id}"
                if tag.tag_id == 0:
                    label += " (TARGET)"
                cv2.putText(
                    frame,
                    label,
                    c,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            cv2.imshow("Tello Camera (AprilTag Detection)", frame)

            # ESC to quit
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        # Clean shutdown
        tello.streamoff()
        tello.end()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
