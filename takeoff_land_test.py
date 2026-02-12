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

# Hover for 5 seconds
print("Hovering for 5 seconds...")
time.sleep(5)

# Land
print("Landing...")
tello.land()

print("Done!")