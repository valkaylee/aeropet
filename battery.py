from djitellopy import Tello
t = Tello()
t.connect()
print("Battery:", t.get_battery(), "%")
t.end()
