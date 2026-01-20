import cv2
from djitellopy import Tello
import time

def test_tello_camera():
    """Test DJI Tello camera feed"""
    
    print("Connecting to Tello...")
    tello = Tello()
    
    try:
        # Connect to drone
        tello.connect()
        print(f"✓ Connected! Battery: {tello.get_battery()}%")
        
        # Start video stream
        print("Starting video stream...")
        tello.streamon()
        time.sleep(2)  # Give it time to initialize
        
        print("✓ Video stream active")
        print("Press 'q' to quit")
        print("Press 't' to takeoff")
        print("Press 'l' to land")
        
        while True:
            # Get frame from Tello
            frame_read = tello.get_frame_read()
            frame = frame_read.frame
            
            if frame is None:
                print("No frame received")
                continue
            
            # Add info overlay
            battery = tello.get_battery()
            cv2.putText(frame, f"Battery: {battery}%", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow("Tello Camera Feed", frame)
            
            # Handle keypresses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('t'):
                print("Taking off...")
                tello.takeoff()
            elif key == ord('l'):
                print("Landing...")
                tello.land()
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        print("Cleaning up...")
        try:
            if tello.is_flying:
                tello.land()
            tello.streamoff()
        except:
            pass
        cv2.destroyAllWindows()
        print("Done!")

if __name__ == "__main__":
    test_tello_camera()