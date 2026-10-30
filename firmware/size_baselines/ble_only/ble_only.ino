// Size baseline: mbed core + USB serial + ArduinoBLE peripheral with one service (no DSP/ML).
#include <ArduinoBLE.h>
BLEService svc("6c6b0001-5a1e-4b1a-9c55-6e616e6f3333");
BLECharacteristic rx("6c6b0002-5a1e-4b1a-9c55-6e616e6f3333", BLEWriteWithoutResponse | BLEWrite, 244);
BLECharacteristic res("6c6b0003-5a1e-4b1a-9c55-6e616e6f3333", BLENotify | BLERead, 244);
void setup() {
  Serial.begin(115200);
  BLE.begin();
  svc.addCharacteristic(rx);
  svc.addCharacteristic(res);
  BLE.addService(svc);
  BLE.advertise();
}
void loop() { BLE.poll(); }
