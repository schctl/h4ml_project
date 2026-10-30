#!/bin/sh
# Compile, save the size report, and upload only if compilation succeeded.
#   ./build.sh [--no-upload] [--all-fft-tables]
# By default CMSIS-DSP is built with only the tables a 1024-point arm_rfft_fast_f32 needs
# (cmsis_tables.flags); --all-fft-tables reproduces the stock build (all FFT sizes linked, +111 KB flash).
# ArduinoBLE comes from ../libraries (4 KB HCI RX ring, see ../libraries/PATCHES.md); --stock-ble uses the
# globally installed ArduinoBLE 1.4.0.
set -e
cd "$(dirname "$0")"
FLAGS="$(cat cmsis_tables.flags)"
UPLOAD=1
LIBARGS="--library ../libraries/ArduinoBLE"
for a in "$@"; do
  case "$a" in
    --no-upload) UPLOAD=0 ;;
    --all-fft-tables) FLAGS="" ;;
    --stock-ble) LIBARGS="" ;;
  esac
done
mkdir -p build
arduino-cli compile --fqbn arduino:mbed_nano:nano33ble --build-path build $LIBARGS \
  --build-property "compiler.c.extra_flags=$FLAGS" --build-property "compiler.cpp.extra_flags=$FLAGS" . \
  > compile.log 2>&1 || { grep -E -i "error" compile.log; exit 1; }
grep -E "Sketch uses|Global variables" compile.log
[ "$UPLOAD" = 0 ] || arduino-cli upload -p "${PORT:-/dev/ttyACM0}" --fqbn arduino:mbed_nano:nano33ble --input-dir build . 2>&1 \
  | grep -E -o "Done in [0-9.]+ seconds|rror.*" | tail -1
