import time
from djitellopy import Tello

# Connect to Tello
print("Connecting to Tello...")
tello = Tello()
tello.connect()

# Check battery
battery = tello.get_battery()
print(f"Battery: {battery}%")

if battery < 20:
    print("Battery too low! Charge before flying.")
    exit()

# Takeoff
print("Taking off...")
tello.takeoff()

# # Hover for 2 seconds
# print("Hovering for 2 seconds...")
# time.sleep(2)

# Land
print("Landing...")
tello.land()

print("Done!")