from djitellopy import Tello
import cv2

# Connect to the drone
tello = Tello()
tello.connect()
print(f"Battery level: {tello.get_battery()}%")

# Start the video stream
tello.streamon()

# Get video frames and show them
while True:
    frame = tello.get_frame_read().frame
    cv2.imshow("Tello Camera Test", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
tello.streamoff()
cv2.destroyAllWindows()