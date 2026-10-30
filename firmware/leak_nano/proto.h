// Wire protocol shared by the USB-serial and BLE transports (host side: host/proto.py).
//
// Frame: 0xA5 0x5A | type u8 | window_id u16 | chunk_seq u8 | total_chunks u8 | len u16 | payload | crc16
// All integers little-endian. crc16 = CRC-16/CCITT-FALSE over the 7 header bytes + payload.
// Over BLE each ATT write / notification carries exactly one frame.
#pragma once
#include <stddef.h>
#include <stdint.h>

#define PROTO_MAGIC0 0xA5
#define PROTO_MAGIC1 0x5A
#define PROTO_HDR_LEN 7
#define PROTO_OVERHEAD (2 + PROTO_HDR_LEN + 2)

enum FrameType : uint8_t {
  FT_WINDOW = 0x01,  // payload: int16 samples (a chunk of one window)
  FT_CMD = 0x02,     // payload[0] = command char, rest = args
  FT_RESULT = 0x81,  // payload: see ResultLayout in leak_nano.ino
  FT_TEXT = 0x82,    // payload: ASCII (JSON) reply to a command
  FT_ARRIVALS = 0x84,  // payload: u32[] BLE write arrival times (micros), chunked with seq/total
};

struct FrameHdr {
  uint8_t type;
  uint16_t window_id;
  uint8_t chunk_seq;
  uint8_t total_chunks;
  uint16_t len;
};

static inline uint16_t crc16_ccitt(const uint8_t *d, size_t n, uint16_t crc = 0xFFFF) {
  for (size_t i = 0; i < n; i++) {
    crc ^= (uint16_t)d[i] << 8;
    for (int b = 0; b < 8; b++) crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
  }
  return crc;
}

static inline void hdr_pack(const FrameHdr &h, uint8_t *o) {
  o[0] = h.type;
  o[1] = h.window_id & 0xFF;
  o[2] = h.window_id >> 8;
  o[3] = h.chunk_seq;
  o[4] = h.total_chunks;
  o[5] = h.len & 0xFF;
  o[6] = h.len >> 8;
}

static inline FrameHdr hdr_unpack(const uint8_t *p) {
  FrameHdr h;
  h.type = p[0];
  h.window_id = (uint16_t)(p[1] | (p[2] << 8));
  h.chunk_seq = p[3];
  h.total_chunks = p[4];
  h.len = (uint16_t)(p[5] | (p[6] << 8));
  return h;
}
