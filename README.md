# Edge-ML water-leak detection on an Arduino Nano 33 BLE (hardware-in-the-loop)

Real hydrophone leak recordings are streamed from a host PC to an Arduino Nano 33 BLE (nRF52840, Cortex-M4F at
64 MHz) over USB serial or BLE. The Nano computes FFT features and runs a float32 MLP and an int8 MLP that
matches TFLite's arithmetic bit-exactly. Every reported number comes from a script below and is saved in
`results/`.

## Requirements
- Linux host with Bluetooth LE (tested: Arch Linux, BlueZ 5.87, Realtek USB adapter). `uv` and Python 3.11.
- `arduino-cli` (tested 1.5.2), core `arduino:mbed_nano` 4.1.5, libraries `ArduinoBLE` 1.4.0 and
  `Arduino_CMSIS-DSP` 5.7.0.
- An Arduino Nano 33 BLE on USB. The user must be able to access `/dev/ttyACM*` (Arch: group `uucp`; or the
  core's `60-arduino-mbed.rules`).

## Reproduce
```bash
# 0. Environment
uv venv -p 3.11 .venv
uv pip install -p .venv/bin/python -r host/requirements.txt
arduino-cli core install arduino:mbed_nano@4.1.5
arduino-cli lib install ArduinoBLE@1.4.0 Arduino_CMSIS-DSP@5.7.0

# 1. Dataset (Mendeley Data, DOI 10.17632/tbrnp6vrnj.1): hydrophone archive only, 64 MB
mkdir -p data/raw && cd data/raw
curl -fL -o Hydrophone.zip "https://data.mendeley.com/public-files/datasets/tbrnp6vrnj/files/db8d1475-7cb4-4c60-b9e2-7d47a7d95971/file_downloaded"
echo "d070b62306e1146072bc6929c64203af95e2a1df21a52ce6d93fe1a75395a38d  Hydrophone.zip" | sha256sum -c
mkdir hydrophone && unzip -q Hydrophone.zip -d hydrophone && cd ../..

# 2. Inventory and signal checks  -> results/dataset_*.json|csv, results/signal_check.csv
.venv/bin/python -I python/load.py
.venv/bin/python -I python/signal_check.py

# 3. Host experiments  -> results/e1*.json, e2*.json, e3*.json, e4*.json, diag_shift.json
.venv/bin/python -I python/train.py all          # E1, E1b (saves the deployed model), E2, E3, E4
.venv/bin/python -I python/diagnose_shift.py

# 4. Export: TFLite f32/int8, firmware headers, test vectors  -> results/int8_agreement_host.json
.venv/bin/python -I python/export.py

# 5. Firmware build and upload. build.sh links only the 1024-pt CMSIS FFT tables and uses the patched
#    ArduinoBLE in firmware/libraries (see firmware/libraries/PATCHES.md); --stock-ble / --all-fft-tables
#    reproduce the stock variants. No reset button is needed while the sketch runs (1200-baud touch).
firmware/leak_nano/build.sh && cp firmware/leak_nano/compile.log results/build_output.txt
.venv/bin/python -I python/mem_breakdown.py      # -> results/memory_breakdown.json

# 6. HIL over USB serial: self-test + the held-out test windows  -> results/hil_serial_test.json
.venv/bin/python -I host/serial_stream.py --subset test --tag test

# 7. E6 on the Nano (baseline vs optimized kernels, BLE off/on)  -> results/nano_bench.json
.venv/bin/python -I host/bench_nano.py --n 200

# 8. E6 host reference  -> results/host_bench.json
.venv/bin/python -I host/bench_tflite.py --n 2000

# 9. E7 BLE (real-time 4 kHz stream for 30 s; the Nano stays on USB for control)  -> results/ble_link_<tag>.json
.venv/bin/python -I host/ble_stream.py --burst 1 --realtime-s 30 --tag patched_rt

# 9b. Course-aligned analyses: precision/recall/F1/specificity/ROC-AUC, threshold sweep, MACs and
#     effective MAC/s, ReLU sparsity (zero-skipping), pruning sensitivity  -> results/extra_metrics.json
.venv/bin/python -I python/extra_analysis.py

# 10. Live demo: stream the held-out recordings to the Nano in real time (its RGB LED shows the decision)
.venv/bin/python -I host/live_demo.py
```

## Known issues (measured)
- The mbed core's USB-CDC receive path loses bytes under unpaced 2 KB writes. `host/serial_stream.py` writes in
  512-byte pieces 5 ms apart (`results/serial_pacing_sweep.json`).
- BLE: with BlueZ's default connection interval (about 45 ms, estimated from arrival times) the link carries about
  3.7 kB/s, below the 8 kB/s that real-time 4 kHz streaming needs. A 7.5–15 ms interval, and heavy burst traffic,
  hung the Nano. The 60 s watchdog recovers it. These runs are logged in `results/ble_incidents.json`.

## Recovering the board
- The firmware runs a 60 s hardware watchdog, so a hung sketch reboots by itself. Uploads take about 20 s, well
  inside that window.
- If an upload still fails (the 1200-baud touch cannot reach a hung sketch), double-tap the reset button.
  The orange LED then pulses, the board enumerates as `2341:005a`, and you can upload again.
- The firmware also has a `B` command that reboots into the bootloader.
- The nRF52840 has no ROM USB-DFU, so `dfu-util` does not apply. The only no-button fallback is SWD
  (J-Link or CMSIS-DAP on the bottom pads, with openocd).

## Layout
`python/` dataset, features, training, export · `firmware/leak_nano/` Arduino sketch (generated headers:
`feature_data.h`, `model_data.h`, `testvec.h`) · `host/` serial/BLE streaming and benchmarks ·
`docs/feature_spec.md` feature specification · `results/` all measured outputs ·
`figures/` plots, draw.io diagram sources (`figures/drawio/`) and photos of the setup.
