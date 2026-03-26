import asyncio
from bleak import BleakScanner, BleakClient

DEVICE_NAME = "AEROPET_BLE"
CHAR_UUID = "abcdefab-1234-1234-1234-1234567890ab"

async def beep(client, on_ms=200, off_ms=200):
    await client.write_gatt_char(CHAR_UUID, bytes([1]), response=True)
    await asyncio.sleep(on_ms / 1000)
    await client.write_gatt_char(CHAR_UUID, bytes([0]), response=True)
    await asyncio.sleep(off_ms / 1000)

async def main():
    print("Scanning for AEROPET_BLE...")
    device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=10.0)
    if device is None:
        print("Device not found!")
        return

    print(f"Found: {device.address}")
    async with BleakClient(device) as client:
        print("Connected! Starting beep pattern...")

        # slow beeps
        for _ in range(3):
            await beep(client, on_ms=180, off_ms=700)

        # medium beeps
        for _ in range(5):
            await beep(client, on_ms=120, off_ms=350)

        # fast beeps
        for _ in range(8):
            await beep(client, on_ms=80, off_ms=140)

        # solid tone for 2 seconds
        await client.write_gatt_char(CHAR_UUID, bytes([1]), response=True)
        await asyncio.sleep(2.0)
        await client.write_gatt_char(CHAR_UUID, bytes([0]), response=True)

        print("Done.")

asyncio.run(main())