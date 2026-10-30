# Local library patches

`ArduinoBLE/` is ArduinoBLE 1.4.0 (LGPL-2.1, unchanged except as listed) with one patch:

- `src/utility/HCICordioTransport.h`: `RingBufferN<256> _rxBuf` -> `RingBufferN<4096> _rxBuf`.
  On mbed (Nano 33 BLE) the Cordio controller thread hands HCI packets to the Arduino side through this ring,
  and `handleRxData()` silently drops a packet that does not fit. With a 247-byte ATT MTU only one write fits,
  so every additional write in the same connection event was lost (measured: `results/ble_link_stock.json`).

Build with the stock library: `firmware/leak_nano/build.sh --stock-ble`.
