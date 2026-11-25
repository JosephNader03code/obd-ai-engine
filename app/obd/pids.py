"""OBD-II Mode 01 PID definitions and decoders (SAE J1979).

Every decoder receives the data bytes that follow the "41 <pid>" header.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List


@dataclass(frozen=True)
class PID:
    pid: int
    key: str            # short key used everywhere in the system
    name: str
    unit: str
    nbytes: int
    decode: Callable[[List[int]], float]
    encode: Callable[[float], List[int]]  # used by the simulator


def _clamp(v: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(v))))


PIDS: Dict[int, PID] = {}


def _reg(p: PID) -> None:
    PIDS[p.pid] = p


_reg(PID(0x04, "load", "Calculated engine load", "%", 1,
         lambda d: d[0] * 100 / 255,
         lambda v: [_clamp(v * 255 / 100, 0, 255)]))
_reg(PID(0x05, "ect", "Engine coolant temperature", "°C", 1,
         lambda d: d[0] - 40,
         lambda v: [_clamp(v + 40, 0, 255)]))
_reg(PID(0x06, "stft", "Short term fuel trim - Bank 1", "%", 1,
         lambda d: d[0] * 100 / 128 - 100,
         lambda v: [_clamp((v + 100) * 128 / 100, 0, 255)]))
_reg(PID(0x07, "ltft", "Long term fuel trim - Bank 1", "%", 1,
         lambda d: d[0] * 100 / 128 - 100,
         lambda v: [_clamp((v + 100) * 128 / 100, 0, 255)]))
_reg(PID(0x0B, "map", "Intake manifold absolute pressure", "kPa", 1,
         lambda d: float(d[0]),
         lambda v: [_clamp(v, 0, 255)]))
_reg(PID(0x0C, "rpm", "Engine RPM", "rpm", 2,
         lambda d: (d[0] * 256 + d[1]) / 4,
         lambda v: [(_clamp(v * 4, 0, 65535) >> 8) & 0xFF, _clamp(v * 4, 0, 65535) & 0xFF]))
_reg(PID(0x0D, "speed", "Vehicle speed", "km/h", 1,
         lambda d: float(d[0]),
         lambda v: [_clamp(v, 0, 255)]))
_reg(PID(0x0E, "timing", "Timing advance", "°", 1,
         lambda d: d[0] / 2 - 64,
         lambda v: [_clamp((v + 64) * 2, 0, 255)]))
_reg(PID(0x0F, "iat", "Intake air temperature", "°C", 1,
         lambda d: d[0] - 40,
         lambda v: [_clamp(v + 40, 0, 255)]))
_reg(PID(0x10, "maf", "MAF air flow rate", "g/s", 2,
         lambda d: (d[0] * 256 + d[1]) / 100,
         lambda v: [(_clamp(v * 100, 0, 65535) >> 8) & 0xFF, _clamp(v * 100, 0, 65535) & 0xFF]))
_reg(PID(0x11, "tps", "Throttle position", "%", 1,
         lambda d: d[0] * 100 / 255,
         lambda v: [_clamp(v * 255 / 100, 0, 255)]))
_reg(PID(0x14, "o2_up", "O2 sensor B1S1 voltage", "V", 2,
         lambda d: d[0] / 200,
         lambda v: [_clamp(v * 200, 0, 255), 0xFF]))
_reg(PID(0x15, "o2_down", "O2 sensor B1S2 voltage", "V", 2,
         lambda d: d[0] / 200,
         lambda v: [_clamp(v * 200, 0, 255), 0xFF]))
_reg(PID(0x1F, "runtime", "Run time since engine start", "s", 2,
         lambda d: float(d[0] * 256 + d[1]),
         lambda v: [(_clamp(v, 0, 65535) >> 8) & 0xFF, _clamp(v, 0, 65535) & 0xFF]))
_reg(PID(0x2F, "fuel_level", "Fuel tank level", "%", 1,
         lambda d: d[0] * 100 / 255,
         lambda v: [_clamp(v * 255 / 100, 0, 255)]))
_reg(PID(0x33, "baro", "Barometric pressure", "kPa", 1,
         lambda d: float(d[0]),
         lambda v: [_clamp(v, 0, 255)]))
_reg(PID(0x42, "voltage", "Control module voltage", "V", 2,
         lambda d: (d[0] * 256 + d[1]) / 1000,
         lambda v: [(_clamp(v * 1000, 0, 65535) >> 8) & 0xFF, _clamp(v * 1000, 0, 65535) & 0xFF]))
_reg(PID(0x46, "ambient", "Ambient air temperature", "°C", 1,
         lambda d: d[0] - 40,
         lambda v: [_clamp(v + 40, 0, 255)]))

KEY_TO_PID = {p.key: p for p in PIDS.values()}

# PIDs polled in the live loop (order = polling order)
LIVE_KEYS = ["rpm", "speed", "load", "ect", "iat", "map", "maf", "tps",
             "stft", "ltft", "o2_up", "o2_down", "timing", "voltage"]


# ---------------------------------------------------------------- DTC helpers
_DTC_LETTER = "PCBU"


def decode_dtc(b1: int, b2: int) -> str:
    """Two raw bytes -> 'P0420' style code (SAE J2012)."""
    letter = _DTC_LETTER[(b1 >> 6) & 0x03]
    d1 = (b1 >> 4) & 0x03
    d2 = b1 & 0x0F
    return f"{letter}{d1}{d2:X}{b2:02X}"


def encode_dtc(code: str) -> List[int]:
    """'P0420' -> [0x04, 0x20] (used by the simulator / firmware tests)."""
    code = code.strip().upper()
    letter = _DTC_LETTER.index(code[0])
    d1 = int(code[1], 16)
    d2 = int(code[2], 16)
    b1 = (letter << 6) | (d1 << 4) | d2
    b2 = int(code[3:5], 16)
    return [b1, b2]


def supported_bitmap(pids: List[int], base: int = 0x00) -> List[int]:
    """Build the 4-byte 'PIDs supported' bitmap for PID base (00, 20, 40...)."""
    bits = 0
    for p in pids:
        if base < p <= base + 0x20:
            bits |= 1 << (32 - (p - base))
    # Flag that the next range is supported if any pid above it exists
    if any(p > base + 0x20 for p in pids):
        bits |= 1
    return [(bits >> 24) & 0xFF, (bits >> 16) & 0xFF, (bits >> 8) & 0xFF, bits & 0xFF]


def parse_supported_bitmap(data: List[int], base: int = 0x00) -> List[int]:
    bits = (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3]
    return [base + i for i in range(1, 33) if bits & (1 << (32 - i))]
