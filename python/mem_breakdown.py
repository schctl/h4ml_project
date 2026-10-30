"""Flash/RAM attribution of the firmware ELF by symbol group (arm-none-eabi-nm). Writes results/memory_breakdown.json."""
import json, re, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
elf = ROOT / "firmware/leak_nano/build/leak_nano.ino.elf"
nm = next(Path.home().glob(".arduino15/packages/arduino/tools/arm-none-eabi-gcc/*/bin/arm-none-eabi-nm"))
size = nm.with_name("arm-none-eabi-size")
out = subprocess.run([str(nm), "-S", "--size-sort", "-C", str(elf)], capture_output=True, text=True, check=True).stdout
groups = [("model weights f32 (MD_W*, MD_B*)", r"^MD_(W|B)\d"), ("model weights int8 (MD_Q*)", r"^MD_Q"),
          ("feature tables (FD_*)", r"^FD_"), ("self-test vectors (TV_*)", r"^TV_"),
          ("CMSIS-DSP FFT tables", r"(twiddleCoef|armBitRevIndexTable|arm_cfft_sR|armRecipTable|arm_rfft_sR|sinTable|cos_factors|realCoef|Weights_|arm_cfft_.*_len)"),
          ("app buffers (bench, fft, queues)", r"^(bench_[ab]|buf_in|buf_out|logmag|ble_q|arrivals|srx|tx_buf|txt|asm_serial|asm_ble|act_[ab]|q_[ab]|feat)$")]
res = {g: {"flash": 0, "ram": 0, "symbols": 0} for g, _ in groups}
for line in out.splitlines():
    p = line.split(maxsplit=3)
    if len(p) < 4: continue
    sz, typ, name = int(p[1], 16), p[2], p[3]
    for g, rx in groups:
        if re.search(rx, name):
            kind = "ram" if typ in "bBdD" else "flash"
            res[g][kind] += sz; res[g]["symbols"] += 1
            if typ in "dD": res[g]["flash"] += sz   # initialized data also occupies flash
            break
sec = subprocess.run([str(size), "-A", str(elf)], capture_output=True, text=True, check=True).stdout
sections = {m.group(1): int(m.group(2)) for m in re.finditer(r"^(\.\S+)\s+(\d+)", sec, re.M)}
base = {}
for l in (ROOT / "results/size_baselines.txt").read_text().splitlines():
    m = re.match(r"(\w+): (Sketch uses|Global variables use) (\d+)", l)
    if m: base.setdefault(m.group(1), {})["flash" if m.group(2).startswith("Sketch") else "ram"] = int(m.group(3))
full = {}
for l in (ROOT / "results/build_output.txt").read_text().splitlines():
    m = re.match(r"(Sketch uses|Global variables use) (\d+)", l)
    if m: full["flash" if m.group(1).startswith("Sketch") else "ram"] = int(m.group(2))
j = dict(arduino_cli_report=dict(full_firmware=full, **base), groups=res, elf_sections=sections,
         note="Group sizes from nm -S (static symbols only); arduino-cli 'Global variables' = .data+.bss static RAM. "
              "Heap/stack at runtime are in results/nano_bench.json (mem_after).")
(ROOT / "results/memory_breakdown.json").write_text(json.dumps(j, indent=1))
print(json.dumps(j["arduino_cli_report"]), "\n", json.dumps(res, indent=1))
