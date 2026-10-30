"""Host side of firmware/leak_nano/proto.h."""
import json
import struct

MAGIC = b"\xa5\x5a"
HDR = struct.Struct("<BHBBH")  # type, window_id, chunk_seq, total_chunks, len
OVERHEAD = 2 + HDR.size + 2
FT_WINDOW, FT_CMD, FT_RESULT, FT_TEXT, FT_ARRIVALS = 0x01, 0x02, 0x81, 0x82, 0x84


def crc16(data, crc=0xFFFF):
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def frame(ftype, payload=b"", window_id=0, seq=0, total=1):
    body = HDR.pack(ftype, window_id, seq, total, len(payload)) + payload
    return MAGIC + body + struct.pack("<H", crc16(body))


def window_frames(window_id, samples_int16, chunk_payload):
    """Split one window (np.int16 array) into FT_WINDOW frames with <= chunk_payload payload bytes (even)."""
    raw = samples_int16.astype("<i2").tobytes()
    chunk_payload -= chunk_payload % 2
    parts = [raw[i:i + chunk_payload] for i in range(0, len(raw), chunk_payload)]
    return [frame(FT_WINDOW, p, window_id, i, len(parts)) for i, p in enumerate(parts)]


def cmd(c, args=b""):
    return frame(FT_CMD, c.encode() + args)


def parse_result(p):
    wid, nc, pf, pi = struct.unpack_from("<HBBB", p, 0)
    feat_us, fft_us, mf_us, mi_us = struct.unpack_from("<IIII", p, 5)
    logits = struct.unpack_from(f"<{nc}f", p, 21)
    q = struct.unpack_from(f"<{nc}b", p, 21 + 4 * nc)
    return dict(window_id=wid, n_classes=nc, pred_f32=pf, pred_i8=pi, feat_us=feat_us, fft_us=fft_us,
                mlp_f32_us=mf_us, mlp_i8_us=mi_us, logits_f32=list(logits), q_out=list(q))


class Parser:
    """Incremental frame parser for a byte stream (serial). Yields (header_tuple, payload)."""

    def __init__(self):
        self.buf = bytearray()
        self.crc_errors = 0

    def feed(self, data):
        self.buf += data
        out = []
        while True:
            i = self.buf.find(MAGIC)
            if i < 0:
                del self.buf[:-1]
                break
            del self.buf[:i]
            if len(self.buf) < 2 + HDR.size:
                break
            h = HDR.unpack_from(self.buf, 2)
            n = 2 + HDR.size + h[4] + 2
            if len(self.buf) < n:
                break
            body = bytes(self.buf[2:n - 2])
            if struct.unpack_from("<H", self.buf, n - 2)[0] != crc16(body):
                self.crc_errors += 1
                del self.buf[:2]
                continue
            out.append((h, body[HDR.size:]))
            del self.buf[:n]
        return out


class TextAssembler:
    """Reassembles chunked FT_TEXT replies into JSON objects."""

    def __init__(self):
        self.parts = {}

    def add(self, h, payload):
        _, _, seq, total, _ = h
        self.parts[seq] = payload
        if len(self.parts) == total:
            txt = b"".join(self.parts[i] for i in range(total)).decode()
            self.parts = {}
            return json.loads(txt)
        return None
