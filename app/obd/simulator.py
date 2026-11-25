"""Mock vehicle + ELM327 emulator.

`VehicleModel` produces physically plausible sensor values for several fault
scenarios. `SimTransport` answers ELM327 text commands exactly like a real
adapter (CAN 11-bit 500k, headers off, spaces off, ISO-TP multi-frame
formatting) so the *real* ELM327 driver and parser are exercised end-to-end.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .pids import KEY_TO_PID, PIDS, encode_dtc, supported_bitmap

AIR_DENSITY = 1.184  # g/L at 25°C
VE = 0.85


@dataclass
class Scenario:
    key: str
    title_ar: str
    title_en: str
    stored: List[str] = field(default_factory=list)
    pending: List[str] = field(default_factory=list)


SCENARIOS: Dict[str, Scenario] = {s.key: s for s in [
    Scenario("normal", "سيارة سليمة", "Healthy vehicle"),
    Scenario("catalyst", "كفاءة الشكمان (الكتلايزر) ضعيفة", "Catalyst efficiency below threshold", ["P0420"]),
    Scenario("vacuum_leak", "تسريب هواء (فاكيوم)", "Vacuum leak", ["P0171"]),
    Scenario("maf_dirty", "حساس MAF متسخ", "Dirty / under-reading MAF sensor", ["P0171"], ["P0101"]),
    Scenario("fuel_delivery", "ضعف طلمبة البنزين / الفلتر", "Weak fuel delivery", ["P0171"]),
    Scenario("o2_slow", "حساس الأكسجين الأمامي بطيء", "Slow upstream O2 sensor", ["P0133"]),
    Scenario("misfire", "تقطيع في السلندر 3", "Cylinder 3 misfire", ["P0300", "P0303"]),
    Scenario("thermostat", "الثرموستات مفتوح باستمرار", "Thermostat stuck open", ["P0128"]),
    Scenario("charging", "ضعف الشحن (الدينامو)", "Charging system fault", ["P0562"]),
    Scenario("catalyst_lean", "P0420 بسبب خلط فقير (تسريب عادم/هواء)", "P0420 caused by lean condition", ["P0420", "P0171"]),
    Scenario("overheating", "ارتفاع حرارة المحرك", "Engine overheating", ["P0217"]),
    Scenario("egr_fault", "عطل صمام EGR", "EGR insufficient flow", ["P0401"]),
    Scenario("evap_leak", "تسريب نظام EVAP", "EVAP system leak", ["P0455"], ["P0442"]),
    Scenario("idle_fault", "عدم استقرار لفات الثبات", "Idle speed control malfunction", ["P0505"]),
    Scenario("sensor_iat", "عطل دائرة حساس حرارة الهواء", "IAT sensor circuit high", ["P0113"]),
    Scenario("speed_sensor", "عطل حساس السرعة", "Vehicle speed sensor fault", ["P0500"]),
]}


class VehicleModel:
    def __init__(self, scenario: str = "normal", seed: Optional[int] = None,
                 displacement_l: float = 1.6, start_time: float = 600.0,
                 severity: float = 1.0):
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario}")
        self.scenario = SCENARIOS[scenario]
        self.rng = random.Random(seed)
        self.disp = displacement_l
        self.t = start_time
        self.severity = severity
        self.vin = "WVWZZZ1KZ8W123456"
        self.stored = list(self.scenario.stored)
        self.pending = list(self.scenario.pending)
        self.cleared_at: Optional[float] = None
        self.ltft_mem = 0.0
        if scenario == "thermostat":
            self.ect = 67.0
        elif scenario == "overheating":
            self.ect = 95.0
        else:
            self.ect = 90.0
        self.supported = sorted(KEY_TO_PID[k].pid for k in KEY_TO_PID)
        # per-instance variation (different "cars")
        self.k_idle = self.rng.uniform(720, 820)
        self.o2_phase = self.rng.uniform(0, 6.28)
        self.cycle_offset = self.rng.uniform(0, 60)

    # ------------------------------------------------------------ physics
    def advance(self, dt: float) -> None:
        self.t += dt
        # DTCs reappear ~2 minutes after a clear if the fault is still present
        if self.cleared_at is not None and self.t - self.cleared_at > 120:
            self.stored = list(self.scenario.stored)
            self.pending = list(self.scenario.pending)
            self.cleared_at = None

    def clear_codes(self) -> None:
        self.stored, self.pending = [], []
        self.cleared_at = self.t

    def _profile(self):
        """Repeating 60 s drive cycle: idle -> cruise -> load -> decel."""
        c = (self.t + self.cycle_offset) % 60
        if c < 20:
            return self.k_idle, 0.0, 1.0  # rpm, speed, phase
        if c < 35:
            return 2300 + 200 * math.sin(c), 60 + 3 * math.sin(c / 2), 2.0
        if c < 45:
            return 3000 + 150 * math.sin(c), 90, 3.0
        f = (c - 45) / 15
        return 3000 * (1 - f) + self.k_idle * f, 90 * (1 - f), 4.0

    def sample(self) -> Dict[str, float]:
        s = self.scenario.key
        sev = self.severity
        r = self.rng
        rpm, speed, phase = self._profile()
        idle = phase == 1.0
        high = phase == 3.0

        if s == "misfire" and idle:
            rpm += r.gauss(0, 55 * sev)
        elif s == "idle_fault" and idle:
            rpm += 180 * sev + r.gauss(0, 35 * sev)
        else:
            rpm += r.gauss(0, 8)

        map_kpa = {1.0: 30, 2.0: 45, 3.0: 75, 4.0: 25}[phase] + r.gauss(0, 1)
        if s in ("vacuum_leak", "catalyst_lean") and idle:
            map_kpa += 4 * sev
        load = min(100, map_kpa / 101 * 100 + r.gauss(0, 1))
        tps = {1.0: 0, 2.0: 18, 3.0: 45, 4.0: 0}[phase] + abs(r.gauss(0, 0.5))

        maf_true = rpm / 120 * self.disp * (map_kpa / 101.3) * AIR_DENSITY * VE
        maf_read = maf_true * (1 - 0.25 * sev if s == "maf_dirty" else 1.0) * (1 + r.gauss(0, 0.02))

        # total fuel correction needed (%), split into LTFT (slow) + STFT (fast)
        need = 1.0
        if s in ("vacuum_leak", "catalyst_lean"):
            need += (18 if idle else 4 if not high else 2) * sev
        elif s == "maf_dirty":
            need += (6 if idle else 14 if not high else 19) * sev
        elif s == "fuel_delivery":
            need += (2 if idle else 7 if not high else 21) * sev
        elif s == "misfire":
            need += 6 * sev if idle else 2
        self.ltft_mem += (need * 0.8 - self.ltft_mem) * 0.08
        stft = (need - self.ltft_mem) + r.gauss(0, 2.2)
        ltft = self.ltft_mem + r.gauss(0, 0.4)

        # O2 sensors
        f_up = 0.22 if s == "o2_slow" else 0.85
        amp = 0.15 if s == "o2_slow" else 0.38
        x = math.sin(2 * math.pi * f_up * self.t + self.o2_phase)
        o2_up = 0.45 + amp * math.tanh(3 * x) + r.gauss(0, 0.03)
        if s in ("catalyst", "catalyst_lean"):
            xd = math.sin(2 * math.pi * f_up * (self.t - 0.3) + self.o2_phase)
            o2_down = 0.45 + 0.32 * sev * math.tanh(3 * xd) + r.gauss(0, 0.03)
        else:
            o2_down = 0.68 + 0.02 * math.sin(self.t / 9) + r.gauss(0, 0.012)
        if s == "catalyst_lean":
            o2_up -= 0.06
        o2_up = min(1.1, max(0.02, o2_up))
        o2_down = min(1.1, max(0.02, o2_down))

        # coolant
        if s == "overheating":
            target = 115.0 + 5 * sev
            self.ect += (target - self.ect) * 0.02 + r.gauss(0, 0.2)
        elif s == "thermostat":
            target = 67.0
            self.ect += (target - self.ect) * 0.05 + r.gauss(0, 0.15)
        else:
            target = 90.0
            self.ect += (target - self.ect) * 0.05 + r.gauss(0, 0.15)

        volt = (12.1 if s == "charging" else 14.1) + r.gauss(0, 0.05)
        timing = (12 if idle else 28 if not high else 22) + r.gauss(0, 1.0)
        if s == "misfire" and idle:
            timing += r.gauss(0, 3)

        iat = 32 + r.gauss(0, 0.3)
        if s == "sensor_iat":
            iat = 120 + r.gauss(0, 2)

        reported_speed = speed
        if s == "speed_sensor":
            reported_speed = 0.0 if r.random() < 0.8 * sev else speed

        return {
            "rpm": max(0, rpm), "speed": max(0, reported_speed), "load": load, "ect": self.ect,
            "iat": iat, "map": map_kpa, "maf": maf_read, "tps": tps,
            "stft": stft, "ltft": ltft, "o2_up": o2_up, "o2_down": o2_down,
            "timing": timing, "voltage": volt, "runtime": self.t, "fuel_level": 55.0,
            "baro": 101, "ambient": 27,
        }


class SimTransport:
    """Speaks ELM327. Uses wall-clock time unless `fast=True` (then each call
    to `tick()` advances the model clock instead)."""

    def __init__(self, vehicle: VehicleModel, realtime: bool = True):
        self.v = vehicle
        self.realtime = realtime
        self._last = time.time()
        self._out = ""
        self.spaces = True
        self.headers = False
        self.echo = True
        self._cache: Optional[Dict[str, float]] = None

    def tick(self, dt: float) -> None:
        """Advance virtual time and take a fresh physical sample."""
        self.v.advance(dt)
        self._cache = self.v.sample()

    def _sample(self) -> Dict[str, float]:
        if self.realtime:
            now = time.time()
            dt = now - self._last
            if dt > 0.2 or self._cache is None:
                self._last = now
                self.tick(min(dt, 2.0))
        elif self._cache is None:
            self._cache = self.v.sample()
        return self._cache  # type: ignore[return-value]

    def _fmt(self, payload: List[int]) -> str:
        sep = " " if self.spaces else ""
        if len(payload) <= 7:
            return sep.join(f"{b:02X}" for b in payload)
        # ISO-TP multi-frame, ELM style (headers off)
        lines = [f"{len(payload):03X}"]
        first, rest = payload[:6], payload[6:]
        lines.append("0:" + sep.join(f"{b:02X}" for b in first))
        idx = 1
        while rest:
            chunk, rest = rest[:7], rest[7:]
            lines.append(f"{idx % 16:X}:" + sep.join(f"{b:02X}" for b in chunk))
            idx += 1
        return "\r".join(lines)

    def _handle(self, cmd: str) -> str:
        c = cmd.strip().upper().replace(" ", "")
        if c == "ATZ":
            self.spaces, self.headers, self.echo = True, False, True
            return "ELM327 v1.5 (SIM)"
        if c == "ATE0":
            self.echo = False
            return "OK"
        if c == "ATS0":
            self.spaces = False
            return "OK"
        if c == "ATS1":
            self.spaces = True
            return "OK"
        if c == "ATDPN":
            return "A6"
        if c == "ATDP":
            return "AUTO, ISO 15765-4 (CAN 11/500)"
        if c == "ATRV":
            return f"{self._sample()['voltage']:.1f}V"
        if c.startswith("AT"):
            return "OK"
        if c.startswith("01") and len(c) >= 4:
            pid = int(c[2:4], 16)
            if pid in (0x00, 0x20, 0x40):
                return self._fmt([0x41, pid] + supported_bitmap(self.v.supported, pid))
            if pid not in PIDS:
                return "NO DATA"
            val = self._sample()[PIDS[pid].key]
            return self._fmt([0x41, pid] + PIDS[pid].encode(val))
        if c in ("03", "07", "0A"):
            codes = {"03": self.v.stored, "07": self.v.pending, "0A": self.v.stored}[c]
            body: List[int] = []
            for code in codes:
                body += encode_dtc(code)
            return self._fmt([int(c, 16) + 0x40, len(codes)] + body)
        if c == "04":
            self.v.clear_codes()
            return "44"
        if c == "0902":
            return self._fmt([0x49, 0x02, 0x01] + list(self.v.vin.encode()))
        return "?"

    # Transport interface
    def write(self, data: str) -> None:
        cmd = data.strip()
        resp = self._handle(cmd)
        prefix = (cmd + "\r") if self.echo else ""
        self._out = prefix + resp + "\r\r>"

    def read_until_prompt(self, timeout: float = 5.0) -> str:
        out, self._out = self._out, ""
        return out

    def close(self) -> None:
        pass
