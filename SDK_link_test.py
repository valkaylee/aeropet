from djitellopy import Tello
import time

# Initialize the drone
drone = Tello()
drone.connect()

# Print battery percentage
print(f"Battery: {drone.get_battery()}%")

# Take off
drone.takeoff()
time.sleep(2)

# Move forward 50cm
drone.move_forward(50)
time.sleep(2)

# Move backward 50cm
drone.move_back(50)
time.sleep(2)

# Move up 50cm
drone.move_up(50)
time.sleep(2)

# Move down 50cm
drone.move_down(50)
time.sleep(2)

# Move left 50cm
drone.move_left(50)
time.sleep(2)

# Move right 50cm
drone.move_right(50)
time.sleep(2)

# Land
drone.land()
