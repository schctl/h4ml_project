// Leak detection HIL firmware for the Arduino Nano 33 BLE.
//
// Windows of 1024 int16 samples (4 kHz) arrive over USB serial or BLE using the framing in proto.h.
// Every window runs features -> float32 MLP and features -> int8 MLP; the result goes back on the
// transport the window came from. Commands (FT_CMD, payload[0]):
//   'I' info   'S' self-test   'M' memory stats   'G' get counters   'R' reset counters
//   'A' dump BLE write arrival times   'X' + '0'/'1' disable/enable BLE   'B' reboot to bootloader
//   'P' per-stage feature profile   'O' + '0'/'1' baseline/optimized kernels
//
// ResultLayout (FT_RESULT payload, little-endian):
//   u16 window_id | u8 n_classes | u8 pred_f32 | u8 pred_i8 | u32 feat_us | u32 fft_us |
//   u32 mlp_f32_us | u32 mlp_i8_us | f32 logits_f32[n] | i8 q_out[n]
#include <ArduinoBLE.h>
#include <arm_math.h>
#include <malloc.h>
#include <rtx_os.h>

#include "dsp.h"
#include "mlp.h"
#include "proto.h"
#include "testvec.h"

#define FW_VERSION "1.0.0"
#define WIN_BYTES (DSP_N_FFT * 2)
#define SVC_UUID "6c6b0001-5a1e-4b1a-9c55-6e616e6f3333"
#define RX_UUID "6c6b0002-5a1e-4b1a-9c55-6e616e6f3333"
#define RES_UUID "6c6b0003-5a1e-4b1a-9c55-6e616e6f3333"
#define BLE_MAX_WRITE 244
#define BLE_QUEUE 32
#define N_ARRIVALS 1024

void _ontouch1200bps_();  // mbed core, variant.cpp: sets GPREGRET and resets into the bootloader

enum Transport { T_SERIAL = 0, T_BLE = 1 };

// ---------------- state ----------------
struct Counters {
  uint32_t rx_frames, crc_err, seq_err, len_err, windows_done, notify_fail, ble_queue_overflow;
  uint16_t ble_max_write;
} cnt;

struct Assembler {
  int16_t samples[DSP_N_FFT];
  uint16_t id, nbytes;
  uint8_t next_seq, total;
  bool active;
} asm_serial, asm_ble;

BLEService svc(SVC_UUID);
BLECharacteristic rx_char(RX_UUID, BLEWriteWithoutResponse | BLEWrite, BLE_MAX_WRITE);
BLECharacteristic res_char(RES_UUID, BLENotify | BLERead, BLE_MAX_WRITE);
bool ble_on = false;

struct BleFrame {
  uint16_t len;
  uint8_t data[BLE_MAX_WRITE];
};
BleFrame ble_q[BLE_QUEUE];
volatile uint8_t ble_head = 0, ble_tail = 0;
uint32_t arrivals[N_ARRIVALS];
uint32_t n_arrivals = 0;

static float feat[DSP_N_BANDS];
static uint8_t tx_buf[2048 + PROTO_OVERHEAD];

// ---------------- watchdog + breadcrumb ----------------
// A 60 s hardware watchdog recovers the board from hangs (the 1200-baud bootloader touch cannot run then).
// The nRF52 WDT keeps running through the bootloader; 60 s leaves margin over a ~20 s upload.
// GPREGRET2 survives a WDT reset and records the last loop() stage entered.
#define WDT_TIMEOUT_S 60
enum Crumb : uint8_t { CR_SERIAL = 1, CR_BLE_POLL = 2, CR_BLE_DRAIN = 3, CR_NOTIFY = 4, CR_RX_CB = 5, CR_IDLE = 6 };
static uint32_t boot_resetreas = 0;
static uint8_t boot_crumb = 0;
static inline void crumb(uint8_t c) { NRF_POWER->GPREGRET2 = c; }

static void wdt_start() {
  if (!NRF_WDT->RUNSTATUS) {
    NRF_WDT->CONFIG = (WDT_CONFIG_HALT_Pause << WDT_CONFIG_HALT_Pos) | (WDT_CONFIG_SLEEP_Run << WDT_CONFIG_SLEEP_Pos);
    NRF_WDT->CRV = 32768u * WDT_TIMEOUT_S - 1;
    NRF_WDT->RREN = WDT_RREN_RR0_Msk;
    NRF_WDT->TASKS_START = 1;
  }
}
static inline void wdt_feed() { NRF_WDT->RR[0] = WDT_RR_RR_Reload; }

// ---------------- stack high-water mark (painting) ----------------
static uint32_t *stk_lo, *stk_hi;
#define STK_PAINT 0xA5A5A5A5u

static void stack_paint() {
  osRtxThread_t *th = (osRtxThread_t *)osThreadGetId();
  stk_lo = (uint32_t *)th->stack_mem;
  stk_hi = (uint32_t *)((uint8_t *)th->stack_mem + th->stack_size);
  uint32_t *sp = (uint32_t *)__get_PSP();
  for (uint32_t *p = stk_lo + 1; p < sp - 64; p++) *p = STK_PAINT;  // word 0 is RTX's magic word
}

static uint32_t stack_used() {
  uint32_t *p = stk_lo + 1;
  while (p < stk_hi && *p == STK_PAINT) p++;
  return (uint32_t)(stk_hi - p) * 4;
}

// ---------------- framing / output ----------------
static void send_frame(Transport t, uint8_t type, uint16_t id, uint8_t seq, uint8_t total,
                       const uint8_t *payload, uint16_t len) {
  FrameHdr h{type, id, seq, total, len};
  tx_buf[0] = PROTO_MAGIC0;
  tx_buf[1] = PROTO_MAGIC1;
  hdr_pack(h, tx_buf + 2);
  memcpy(tx_buf + 2 + PROTO_HDR_LEN, payload, len);
  uint16_t crc = crc16_ccitt(tx_buf + 2, PROTO_HDR_LEN + len);
  tx_buf[2 + PROTO_HDR_LEN + len] = crc & 0xFF;
  tx_buf[3 + PROTO_HDR_LEN + len] = crc >> 8;
  const int n = len + PROTO_OVERHEAD;
  if (t == T_SERIAL) {
    Serial.write(tx_buf, n);
  } else if (crumb(CR_NOTIFY), !BLE.connected() || !res_char.subscribed() || !res_char.writeValue(tx_buf, n)) {
    // ArduinoBLE's ACL send busy-waits for controller credits; never notify without a subscribed central
    // (a notify flood across a dropped link hung the sketch in the first BLE test).
    cnt.notify_fail++;
  }
}

static void send_text(Transport t, const char *s) {
  const int n = strlen(s);
  int chunk = 1000;
  if (t == T_BLE) chunk = (cnt.ble_max_write > PROTO_OVERHEAD + 8) ? cnt.ble_max_write - PROTO_OVERHEAD : 9;
  const int total = (n + chunk - 1) / chunk;
  for (int i = 0; i < total; i++) {
    const int len = min(chunk, n - i * chunk);
    send_frame(t, FT_TEXT, 0, i, total, (const uint8_t *)s + i * chunk, len);
  }
}

// ---------------- inference ----------------
static int argmax_f(const float *v, int n) {
  int b = 0;
  for (int i = 1; i < n; i++) if (v[i] > v[b]) b = i;
  return b;
}
static int argmax_q(const int8_t *v, int n) {
  int b = 0;
  for (int i = 1; i < n; i++) if (v[i] > v[b]) b = i;
  return b;
}

static void put_u32(uint8_t *p, uint32_t v) { memcpy(p, &v, 4); }

// On-board RGB LED (active low) shows the int8 decision of the last window: red = leak, green = no leak.
static void show_decision(int leak) {
  digitalWrite(LEDR, leak ? LOW : HIGH);
  digitalWrite(LEDG, leak ? HIGH : LOW);
  digitalWrite(LEDB, HIGH);
}

static void process_window(Transport t, uint16_t id, const int16_t *x) {
  const int nc = mlp_n_classes();
  float logits[MLP_MAX_CLASSES];
  int8_t q_in[DSP_N_BANDS], q_out[MLP_MAX_CLASSES];
  uint32_t fft_us;

  uint32_t t0 = micros();
  dsp_features(x, feat, &fft_us);
  uint32_t t1 = micros();
  mlp_f32(feat, logits);
  uint32_t t2 = micros();
  mlp_quantize_input(feat, q_in);
  mlp_i8(q_in, q_out);
  uint32_t t3 = micros();

  uint8_t p[5 + 16 + 4 * MLP_MAX_CLASSES + MLP_MAX_CLASSES];
  p[0] = id & 0xFF;
  p[1] = id >> 8;
  p[2] = nc;
  p[3] = argmax_f(logits, nc);
  p[4] = argmax_q(q_out, nc);
  put_u32(p + 5, t1 - t0);
  put_u32(p + 9, fft_us);
  put_u32(p + 13, t2 - t1);
  put_u32(p + 17, t3 - t2);
  memcpy(p + 21, logits, 4 * nc);
  memcpy(p + 21 + 4 * nc, q_out, nc);
  send_frame(t, FT_RESULT, id, 0, 1, p, 21 + 5 * nc);
  show_decision(p[4]);  // after the timed section, so the reported latencies are unaffected
  cnt.windows_done++;
}

static void on_window_chunk(Transport t, Assembler &a, const FrameHdr &h, const uint8_t *p) {
  if (h.chunk_seq == 0) {
    a.active = true;
    a.id = h.window_id;
    a.next_seq = 0;
    a.nbytes = 0;
    a.total = h.total_chunks;
  }
  if (!a.active || h.window_id != a.id || h.chunk_seq != a.next_seq || (h.len & 1) ||
      a.nbytes + h.len > WIN_BYTES) {
    cnt.seq_err++;
    a.active = false;
    return;
  }
  memcpy((uint8_t *)a.samples + a.nbytes, p, h.len);
  a.nbytes += h.len;
  a.next_seq++;
  if (a.next_seq == a.total) {
    a.active = false;
    if (a.nbytes == WIN_BYTES) process_window(t, a.id, a.samples);
    else cnt.len_err++;
  }
}

// ---------------- commands ----------------
static char txt[1024];

static void cmd_selftest(Transport t) {
  const int nc = mlp_n_classes();
  float max_feat_err = 0, max_logit_err_ref = 0, max_logit_err_dev = 0;
  int i8_exact = 0, qin_mismatch = 0, qin_max_diff = 0, pred_f32_agree = 0, pred_i8_dev_agree = 0;
  uint32_t fft_us;
  float logits[MLP_MAX_CLASSES];
  int8_t q_in[DSP_N_BANDS], q_out[MLP_MAX_CLASSES];
  for (int v = 0; v < TV_N; v++) {
    dsp_features(TV_X[v], feat, &fft_us);
    for (int i = 0; i < DSP_N_BANDS; i++) max_feat_err = max(max_feat_err, fabsf(feat[i] - TV_FEAT[v][i]));
    // float MLP on reference features isolates the MLP; on device features gives end-to-end.
    mlp_f32(TV_FEAT[v], logits);
    for (int c = 0; c < nc; c++) max_logit_err_ref = max(max_logit_err_ref, fabsf(logits[c] - TV_LOGITS[v][c]));
    mlp_f32(feat, logits);
    for (int c = 0; c < nc; c++) max_logit_err_dev = max(max_logit_err_dev, fabsf(logits[c] - TV_LOGITS[v][c]));
    pred_f32_agree += argmax_f(logits, nc) == argmax_f(TV_LOGITS[v], nc);
    // int8 MLP on the reference int8 input must be bit-exact with the TFLite interpreter.
    mlp_i8(TV_QIN[v], q_out);
    for (int c = 0; c < nc; c++) i8_exact += q_out[c] == TV_QOUT[v][c];
    mlp_quantize_input(feat, q_in);
    for (int i = 0; i < DSP_N_BANDS; i++) {
      int d = abs((int)q_in[i] - (int)TV_QIN[v][i]);
      qin_mismatch += d != 0;
      qin_max_diff = max(qin_max_diff, d);
    }
    mlp_i8(q_in, q_out);
    pred_i8_dev_agree += argmax_q(q_out, nc) == argmax_q(TV_QOUT[v], nc);
  }
  const bool pass = max_feat_err < TV_TOL_FEAT && max_logit_err_ref < TV_TOL_LOGIT && i8_exact == TV_N * nc;
  snprintf(txt, sizeof txt,
           "{\"selftest\":\"%s\",\"n\":%d,\"max_feat_err\":%.3e,\"tol_feat\":%.1e,\"max_logit_err_ref_in\":%.3e,"
           "\"tol_logit\":%.1e,\"max_logit_err_dev_in\":%.3e,\"pred_f32_agree\":%d,\"i8_exact\":%d,\"i8_total\":%d,"
           "\"qin_mismatch\":%d,\"qin_total\":%d,\"qin_max_diff\":%d,\"pred_i8_dev_agree\":%d}",
           pass ? "PASS" : "FAIL", TV_N, max_feat_err, (double)TV_TOL_FEAT, max_logit_err_ref, (double)TV_TOL_LOGIT,
           max_logit_err_dev, pred_f32_agree, i8_exact, TV_N * nc, qin_mismatch, TV_N * DSP_N_BANDS, qin_max_diff,
           pred_i8_dev_agree);
  send_text(t, txt);
}

static void cmd_profile(Transport t) {
  // Per-stage feature timing over the test vectors, repeated; us totals per window.
  const int reps = 8;
  DspProfile acc{}, one;
  for (int r = 0; r < reps; r++)
    for (int v = 0; v < TV_N; v++) {
      dsp_features_profiled(TV_X[v], feat, &one);
      acc.prep_us += one.prep_us; acc.fft_us += one.fft_us; acc.maglog_us += one.maglog_us;
      acc.band_us += one.band_us;
    }
  const float n = reps * TV_N;
  snprintf(txt, sizeof txt, "{\"profile_windows\":%d,\"prep_us\":%.1f,\"fft_us\":%.1f,\"maglog_us\":%.1f,"
           "\"band_std_us\":%.1f}", (int)n, acc.prep_us / n, acc.fft_us / n, acc.maglog_us / n, acc.band_us / n);
  send_text(t, txt);
}

static float bench_a[6400], bench_b[6400];
static void cmd_microbench(Transport t) {
  for (int i = 0; i < 6400; i++) { bench_a[i] = i * 1e-3f; bench_b[i] = 1.0f - i * 1e-4f; }
  uint32_t t0 = micros(), t1 = micros();
  float acc = 0;
  uint32_t t2 = micros();
  arm_dot_prod_f32(bench_a, bench_b, 6306, &acc);
  uint32_t t3 = micros();
  uint32_t t4 = micros();
  mlp_f32(feat, (float *)bench_a);  // flash-resident weights, current opt mode
  uint32_t t5 = micros();
  uint32_t c0 = DWT->CYCCNT;
  arm_dot_prod_f32(bench_a, bench_b, 6306, &acc);
  uint32_t c1 = DWT->CYCCNT;
  float acc2 = mlp_dot_f32(bench_a, bench_b, 6304);
  uint32_t c2 = DWT->CYCCNT;
  snprintf(txt, sizeof txt, "{\"micros_overhead_us\":%lu,\"cmsis_dot6306_ram_us\":%lu,\"cmsis_dot6306_ram_cycles\":%lu,"
           "\"x4_dot6304_ram_cycles\":%lu,\"mlp_f32_flash_us\":%lu,\"dwt_on\":%d,\"acc\":%.3f,\"acc2\":%.3f}",
           (unsigned long)(t1 - t0), (unsigned long)(t3 - t2), (unsigned long)(c1 - c0), (unsigned long)(c2 - c1),
           (unsigned long)(t5 - t4), (int)((DWT->CTRL & 1) != 0), acc, acc2);
  send_text(t, txt);
}

static void cmd_mem(Transport t) {
  struct mallinfo mi = mallinfo();
  snprintf(txt, sizeof txt,
           "{\"stack_size\":%lu,\"stack_used_hwm\":%lu,\"heap_arena\":%d,\"heap_in_use\":%d,\"heap_free_in_arena\":%d}",
           (unsigned long)((stk_hi - stk_lo) * 4), (unsigned long)stack_used(), mi.arena, mi.uordblks, mi.fordblks);
  send_text(t, txt);
}

static void cmd_counters(Transport t) {
  snprintf(txt, sizeof txt,
           "{\"rx_frames\":%lu,\"crc_err\":%lu,\"seq_err\":%lu,\"len_err\":%lu,\"windows_done\":%lu,"
           "\"notify_fail\":%lu,\"ble_queue_overflow\":%lu,\"ble_max_write\":%u,\"n_arrivals\":%lu}",
           cnt.rx_frames, cnt.crc_err, cnt.seq_err, cnt.len_err, cnt.windows_done, cnt.notify_fail,
           cnt.ble_queue_overflow, cnt.ble_max_write, n_arrivals);
  send_text(t, txt);
}

static void cmd_arrivals(Transport t) {
  // Last N_ARRIVALS BLE write arrival times (micros at poll time), oldest first, as binary u32 chunks.
  static uint32_t tmp[250];
  const uint32_t n = min(n_arrivals, (uint32_t)N_ARRIVALS);
  const int total = max(1, (int)((n + 249) / 250));
  for (int c = 0; c < total; c++) {
    const uint32_t k = min((uint32_t)250, n - c * 250);
    for (uint32_t i = 0; i < k; i++) tmp[i] = arrivals[(n_arrivals - n + c * 250 + i) % N_ARRIVALS];
    send_frame(t, FT_ARRIVALS, 0, c, total, (const uint8_t *)tmp, k * 4);
  }
}

static void ble_start();
static void ble_stop();

static void cmd_info(Transport t) {
  snprintf(txt, sizeof txt,
           "{\"fw\":\"%s\",\"built\":\"%s %s\",\"model_id\":\"%s\",\"feature_spec_version\":%d,\"n_fft\":%d,"
           "\"n_bands\":%d,\"n_classes\":%d,\"core_clock_hz\":%lu,\"ble_on\":%d,\"ble_addr\":\"%s\",\"opt\":%d,"
           "\"nvmc_icachecnf\":%lu,\"boot_resetreas\":%lu,\"boot_crumb\":%u,\"wdt_running\":%lu}",
           FW_VERSION, __DATE__, __TIME__, mlp_model_id(), dsp_spec_version(), DSP_N_FFT, DSP_N_BANDS, mlp_n_classes(),
           (unsigned long)SystemCoreClock, ble_on, ble_on ? BLE.address().c_str() : "", g_dsp_opt,
           (unsigned long)NRF_NVMC->ICACHECNF, (unsigned long)boot_resetreas, boot_crumb,
           (unsigned long)NRF_WDT->RUNSTATUS);
  send_text(t, txt);
}

static void on_cmd(Transport t, const uint8_t *p, uint16_t len) {
  if (len < 1) return;
  switch (p[0]) {
    case 'I': cmd_info(t); break;
    case 'S': cmd_selftest(t); break;
    case 'M': cmd_mem(t); break;
    case 'G': cmd_counters(t); break;
    case 'R': memset(&cnt, 0, sizeof cnt); n_arrivals = 0; send_text(t, "{\"reset\":1}"); break;
    case 'A': cmd_arrivals(t); break;
    case 'P': cmd_profile(t); break;
    case 'U': cmd_microbench(t); break;
    case 'O':  // 'O' + '0'/'1': baseline / optimized DSP+MLP kernels
      g_dsp_opt = g_mlp_opt = (len > 1 && p[1] == '1');
      snprintf(txt, sizeof txt, "{\"opt\":%d}", g_dsp_opt);
      send_text(t, txt);
      break;
    case 'X':
      if (len > 1 && p[1] == '1') ble_start();
      else ble_stop();
      send_text(t, ble_on ? "{\"ble_on\":1}" : "{\"ble_on\":0}");
      break;
    case 'B':
      send_text(t, "{\"bootloader\":1}");
      Serial.flush();
      delay(50);
      _ontouch1200bps_();
      break;
    default: send_text(t, "{\"error\":\"unknown command\"}");
  }
}

static void on_frame(Transport t, const uint8_t *hdr_and_payload) {
  const FrameHdr h = hdr_unpack(hdr_and_payload);
  const uint8_t *p = hdr_and_payload + PROTO_HDR_LEN;
  cnt.rx_frames++;
  if (h.type == FT_WINDOW) on_window_chunk(t, t == T_SERIAL ? asm_serial : asm_ble, h, p);
  else if (h.type == FT_CMD) on_cmd(t, p, h.len);
}

// ---------------- serial receive ----------------
static uint8_t srx[PROTO_HDR_LEN + WIN_BYTES + 2];
static int srx_state = 0, srx_pos = 0, srx_need = 0;

// The core's USB-CDC RX path (Serial.read) loses bytes under unpaced 2 KB bursts (measured: 0/10 windows
// survived), so the host paces its writes (host/serial_stream.py). USBCDC::receive_nb cannot be called
// directly: it needs the protected USB lock.
static void serial_poll() {
  while (Serial.available()) {
    const uint8_t b = Serial.read();
    switch (srx_state) {
      case 0: srx_state = (b == PROTO_MAGIC0) ? 1 : 0; break;
      case 1: srx_state = (b == PROTO_MAGIC1) ? 2 : (b == PROTO_MAGIC0 ? 1 : 0); srx_pos = 0; break;
      case 2:
        srx[srx_pos++] = b;
        if (srx_pos == PROTO_HDR_LEN) {
          const FrameHdr h = hdr_unpack(srx);
          if (h.len > WIN_BYTES) { cnt.len_err++; srx_state = 0; break; }
          srx_need = PROTO_HDR_LEN + h.len + 2;
          srx_state = 3;
        }
        break;
      case 3:
        srx[srx_pos++] = b;
        if (srx_pos == srx_need) {
          srx_state = 0;
          const uint16_t got = srx[srx_need - 2] | (srx[srx_need - 1] << 8);
          if (got != crc16_ccitt(srx, srx_need - 2)) { cnt.crc_err++; break; }
          on_frame(T_SERIAL, srx);
        }
        break;
    }
  }
}

// ---------------- BLE ----------------
static void on_rx_written(BLEDevice, BLECharacteristic c) {
  crumb(CR_RX_CB);
  const uint32_t now = micros();
  arrivals[n_arrivals % N_ARRIVALS] = now;
  n_arrivals++;
  const uint8_t next = (ble_head + 1) % BLE_QUEUE;
  if (next == ble_tail) { cnt.ble_queue_overflow++; return; }
  BleFrame &f = ble_q[ble_head];
  f.len = c.valueLength();
  memcpy(f.data, c.value(), f.len);
  if (f.len > cnt.ble_max_write) cnt.ble_max_write = f.len;
  ble_head = next;
}

static void ble_drain() {
  while (ble_tail != ble_head) {
    BleFrame &f = ble_q[ble_tail];
    if (f.len >= PROTO_OVERHEAD && f.data[0] == PROTO_MAGIC0 && f.data[1] == PROTO_MAGIC1) {
      const FrameHdr h = hdr_unpack(f.data + 2);
      const int n = PROTO_HDR_LEN + h.len;
      if (n + 4 != f.len) cnt.len_err++;
      else if ((f.data[2 + n] | (f.data[3 + n] << 8)) != crc16_ccitt(f.data + 2, n)) cnt.crc_err++;
      else on_frame(T_BLE, f.data + 2);
    } else {
      cnt.len_err++;
    }
    ble_tail = (ble_tail + 1) % BLE_QUEUE;
  }
}

static void ble_start() {
  if (ble_on) return;
  if (!BLE.begin()) return;
  BLE.setLocalName("LeakNano");
  BLE.setDeviceName("LeakNano");
  BLE.setConnectionInterval(6, 12);  // request 7.5-15 ms (units of 1.25 ms)
  svc.addCharacteristic(rx_char);
  svc.addCharacteristic(res_char);
  BLE.addService(svc);
  BLE.setAdvertisedService(svc);
  rx_char.setEventHandler(BLEWritten, on_rx_written);
  BLE.advertise();
  ble_on = true;
}

static void ble_stop() {
  if (!ble_on) return;
  BLE.end();
  ble_on = false;
}

// ---------------- main ----------------
void setup() {
  stack_paint();
  boot_resetreas = NRF_POWER->RESETREAS;
  NRF_POWER->RESETREAS = 0xFFFFFFFF;  // write-1-to-clear
  boot_crumb = NRF_POWER->GPREGRET2;
  crumb(0);
  wdt_start();
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;  // enable the DWT cycle counter
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
  Serial.begin(115200);  // USB CDC: baud is ignored except the 1200 bps bootloader touch
  for (int pin : {LEDR, LEDG, LEDB}) {
    pinMode(pin, OUTPUT);
    digitalWrite(pin, HIGH);  // RGB LED off (active low)
  }
  dsp_init();
  ble_start();
}

void loop() {
  wdt_feed();
  crumb(CR_SERIAL);
  serial_poll();
  if (ble_on) {
    crumb(CR_BLE_POLL);
    BLE.poll();
    crumb(CR_BLE_DRAIN);
    ble_drain();
  }
  crumb(CR_IDLE);
}
