import cv2
from pupil_apriltags import Detector

# Open laptop camera (0 is default webcam)
cap = cv2.VideoCapture(0)

# AprilTag detector
detector = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(gray)

    for tag in tags:
        # Draw box
        for i in range(4):
            pt1 = tuple(tag.corners[i].astype(int))
            pt2 = tuple(tag.corners[(i + 1) % 4].astype(int))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # Show tag ID
        center = tuple(tag.center.astype(int))
        cv2.putText(
            frame,
            f"ID {tag.tag_id}",
            center,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("AprilTag Test", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()