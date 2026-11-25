"""ELM327 protocol driver.

Works with:
  * any ELM327 / ELM327-clone adapter (USB, Bluetooth SPP -> COM port, WiFi TCP)
  * the ESP32 bridge firmware in hardware/firmware (it speaks the same command subset)
  * the built-in simulator (app/obd/simulator.py) through SimTransport

The driver is transport-agnostic: a transport only needs `write(str)` and
`read_until_prompt(timeout) -> str`.
"""
from __future__ import annotations

import re
import socket
import threading
import time
from typing import Dict, List, Optional, Protocol

from .pids import LIVE_KEYS, KEY_TO_PID, PIDS, decode_dtc, parse_supported_bitmap


class Transport(Protocol):
    def write(self, data: str) -> None: ...
    def read_until_prompt(self, timeout: float = 5.0) -> str: ...
    def close(self) -> None: ...


class SerialTransport:
    """USB ELM327, Bluetooth ELM327 (paired -> COMx / /dev/rfcommX) or ESP32 over USB."""

    def __init__(self, port: str, baudrate: int = 38400):
        import serial  # pyserial

        self.ser = serial.Serial(port, baudrate=baudrate, timeout=0.1)

    def write(self, data: str) -> None:
        self.ser.reset_input_buffer()
        self.ser.write(data.encode("ascii"))
        self.ser.flush()

    def read_until_prompt(self, timeout: float = 5.0) -> str:
        buf = bytearray()
        end = time.time() + timeout
        while time.time() < end:
            chunk = self.ser.read(256)
            if chunk:
                buf += chunk
                if b">" in chunk:
                    break
        return buf.decode("ascii", errors="ignore")

    def close(self) -> None:
        self.ser.close()


class TCPTransport:
    """WiFi ELM327 adapters (default 192.168.0.10:35000) or ESP32 in WiFi mode."""

    def __init__(self, host: str = "192.168.0.10", port: int = 35000):
        self.sock = socket.create_connection((host, port), timeout=5)
        self.sock.settimeout(0.2)

    def write(self, data: str) -> None:
        self.sock.sendall(data.encode("ascii"))

    def read_until_prompt(self, timeout: float = 5.0) -> str:
        buf = bytearray()
        end = time.time() + timeout
        while time.time() < end:
            try:
                chunk = self.sock.recv(512)
            except socket.timeout:
                continue
            if not chunk:
                break
            buf += chunk
            if b">" in chunk:
                break
        return buf.decode("ascii", errors="ignore")

    def close(self) -> None:
        self.sock.close()


_NOISE = ("SEARCHING", "BUS INIT", "STOPPED", "OK")
_ERRORS = ("NO DATA", "UNABLE TO CONNECT", "CAN ERROR", "BUS ERROR", "?", "ERROR", "BUFFER FULL")
_HEX_RE = re.compile(r"^[0-9A-F]+$")


class ELM327Error(RuntimeError):
    pass


class ELM327:
    def __init__(self, transport: Transport):
        self.t = transport
        self.lock = threading.Lock()
        self.protocol: str = "?"
        self.is_can: bool = True
        self.version: str = ""
        self.supported: List[int] = []

    # ------------------------------------------------------------ low level
    def command(self, cmd: str, timeout: float = 5.0) -> List[str]:
        with self.lock:
            self.t.write(cmd + "\r")
            raw = self.t.read_until_prompt(timeout)
        lines = []
        for ln in raw.replace("\n", "\r").split("\r"):
            ln = ln.strip().replace(">", "").strip()
            if not ln or ln.upper() == cmd.upper():   # echo
                continue
            lines.append(ln)
        return lines

    def _data_lines(self, cmd: str, timeout: float = 5.0) -> Optional[List[str]]:
        lines = self.command(cmd, timeout)
        out = []
        for ln in lines:
            up = ln.upper()
            if any(up.startswith(n) for n in _NOISE):
                continue
            if any(e in up for e in _ERRORS):
                return None
            out.append(up.replace(" ", ""))
        return out or None

    @staticmethod
    def _join_multiframe(lines: List[str]) -> List[List[int]]:
        """Return a list of byte-responses (one per ECU / message).

        Handles ELM ISO-TP formatting with headers off::

            014
            0:490201314434
            1:475030305235
            2:35423132333435
        """
        responses: List[List[int]] = []
        current: Optional[str] = None
        for ln in lines:
            m = re.match(r"^([0-9A-F]):(.*)$", ln)
            if m:
                if m.group(1) == "0" and current is not None and current != "":
                    responses.append(_hex(current))
                    current = ""
                current = (current or "") + m.group(2)
                continue
            if len(ln) == 3 and _HEX_RE.match(ln):     # byte count line
                if current:
                    responses.append(_hex(current))
                current = ""
                continue
            if current:
                responses.append(_hex(current))
                current = None
            if _HEX_RE.match(ln):
                responses.append(_hex(ln))
        if current:
            responses.append(_hex(current))
        return responses

    # ------------------------------------------------------------ high level
    def initialize(self) -> Dict[str, str]:
        banner = " ".join(self.command("ATZ", timeout=3))
        self.version = banner.strip()
        for c in ("ATE0", "ATL0", "ATS0", "ATH0", "ATAT1", "ATSP0"):
            self.command(c)
        # force protocol search
        self._data_lines("0100", timeout=10)
        dp = self.command("ATDPN")
        self.protocol = (dp[0] if dp else "?").lstrip("A")
        self.is_can = self.protocol in ("6", "7", "8", "9", "A", "B", "C")
        self.supported = self.read_supported_pids()
        return {"version": self.version, "protocol": self.protocol}

    def read_supported_pids(self) -> List[int]:
        supported: List[int] = []
        for base in (0x00, 0x20, 0x40):
            lines = self._data_lines(f"01{base:02X}")
            if not lines:
                break
            for resp in self._join_multiframe(lines):
                if len(resp) >= 6 and resp[0] == 0x41 and resp[1] == base:
                    supported += parse_supported_bitmap(resp[2:6], base)
            if (base + 0x20) not in supported:
                break
        return sorted(set(supported))

    def read_pid(self, pid: int) -> Optional[float]:
        lines = self._data_lines(f"01{pid:02X}", timeout=2)
        if not lines:
            return None
        p = PIDS[pid]
        for resp in self._join_multiframe(lines):
            if len(resp) >= 2 + p.nbytes and resp[0] == 0x41 and resp[1] == pid:
                return round(p.decode(resp[2:2 + p.nbytes]), 3)
        return None

    def read_live(self, keys: List[str] = LIVE_KEYS) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {}
        for k in keys:
            pid = KEY_TO_PID[k].pid
            if self.supported and pid not in self.supported:
                out[k] = None
                continue
            out[k] = self.read_pid(pid)
        return out

    def _read_dtc_mode(self, mode: str) -> List[str]:
        lines = self._data_lines(mode, timeout=5)
        if not lines:
            return []
        resp_id = int(mode, 16) + 0x40
        codes: List[str] = []
        for resp in self._join_multiframe(lines):
            if not resp or resp[0] != resp_id:
                continue
            body = resp[1:]
            if self.is_can and body:
                count = body[0]
                body = body[1:1 + count * 2]
            for i in range(0, len(body) - 1, 2):
                if body[i] == 0 and body[i + 1] == 0:
                    continue
                codes.append(decode_dtc(body[i], body[i + 1]))
        return sorted(set(codes))

    def read_dtcs(self) -> Dict[str, List[str]]:
        return {"stored": self._read_dtc_mode("03"), "pending": self._read_dtc_mode("07"),
                "permanent": self._read_dtc_mode("0A")}

    def clear_dtcs(self) -> bool:
        lines = self.command("04", timeout=5)
        return any(ln.upper().replace(" ", "").startswith("44") for ln in lines)

    def read_vin(self) -> Optional[str]:
        lines = self._data_lines("0902", timeout=5)
        if not lines:
            return None
        for resp in self._join_multiframe(lines):
            if len(resp) > 3 and resp[0] == 0x49 and resp[1] == 0x02:
                vin = bytes(b for b in resp[3:] if 32 < b < 127).decode()
                return vin[-17:] if len(vin) >= 17 else vin
        return None

    def read_battery_voltage(self) -> Optional[float]:
        lines = self.command("ATRV")
        for ln in lines:
            m = re.search(r"(\d+(?:\.\d+)?)\s*V", ln.upper())
            if m:
                return float(m.group(1))
        return None

    def close(self) -> None:
        try:
            self.t.close()
        except Exception:
            pass


def _hex(s: str) -> List[int]:
    s = s[: len(s) - (len(s) % 2)]
    return [int(s[i:i + 2], 16) for i in range(0, len(s), 2)]
