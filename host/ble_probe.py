"""Minimal BLE connect probe: connect, list services, read MTU (optionally acquire), disconnect."""
import asyncio, sys
from bleak import BleakClient, BleakScanner
ACQ = "--acquire" in sys.argv
async def main():
    dev = await BleakScanner.find_device_by_name("LeakNano", timeout=15)
    print("found", dev)
    async with BleakClient(dev) as c:
        print("connected", c.is_connected)
        print("services", [s.uuid for s in c.services])
        print("mtu", c.mtu_size)
        if ACQ:
            try:
                await c._backend._acquire_mtu(); print("acquired mtu", c.mtu_size)
            except Exception as e:
                print("acquire err", repr(e))
        await asyncio.sleep(2)
        print("still connected", c.is_connected)
asyncio.run(main())
