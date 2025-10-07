from djitellopy import Tello

t = Tello()
t.connect()
print(t.get_battery())
t.takeoff()
t.land()
t.end()