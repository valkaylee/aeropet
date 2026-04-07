import asyncio
from bleak import BleakScanner, BleakClient

DEVICE_NAME = "AEROPET_BLE"
CHAR_UUID = "abcdefab-1234-1234-1234-1234567890ab"

async def beep_once():

    print("Scanning for BLE device...")
    device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=10)

    if device is None:
        print("Device not found")
        return

    print(f"Found device: {device.address}")

    async with BleakClient(device) as client:
        print("Connected!")

        # turn buzzer ON
        await client.write_gatt_char(CHAR_UUID, bytes([1]))

        await asyncio.sleep(0.2)

        # turn buzzer OFF
        await client.write_gatt_char(CHAR_UUID, bytes([0]))

        print("Beep complete")

asyncio.run(beep_once())