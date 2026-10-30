// Size baseline: mbed core + USB serial only.
void setup() { Serial.begin(115200); }
void loop() { if (Serial.available()) Serial.write(Serial.read()); }
