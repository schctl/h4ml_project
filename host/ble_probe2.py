"""Replays ble_stream.py's start-up sequence step by step, printing Nano state after each step."""
import asyncio, sys, time
sys.path.insert(0, "host")
import numpy as np, proto
from bleak import BleakClient, BleakScanner
from serial_stream import Link
RX, RES = "6c6b0002-5a1e-4b1a-9c55-6e616e6f3333", "6c6b0003-5a1e-4b1a-9c55-6e616e6f3333"
async def main():
    link = Link("/dev/ttyACM0")
    print("I", {k: link.command("I")[k] for k in ("boot_resetreas", "boot_crumb")})
    dev = await BleakScanner.find_device_by_name("LeakNano", timeout=15)
    got = []
    async with BleakClient(dev) as c:
        await c._backend._acquire_mtu(); print("mtu", c.mtu_size)
        await c.start_notify(RES, lambda _, d: got.append(bytes(d))); print("notify on", c.is_connected)
        await asyncio.sleep(1.0); print("after 1s", c.is_connected)
        print("G", link.command("G")); print("after serial cmd", c.is_connected)
        h = np.load("data/cache/hil_windows.npz")
        frs = proto.window_frames(0, h["frames"][0], c.mtu_size - 3 - proto.OVERHEAD)
        for i, fr in enumerate(frs):
            await c.write_gatt_char(RX, fr, response=False); print("wrote", i, len(fr), c.is_connected)
        await asyncio.sleep(1.0)
        print("results", len(got), "connected", c.is_connected)
        print("G", link.command("G"))
asyncio.run(main())
