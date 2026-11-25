"""Connection manager: one place that owns the adapter (real or simulated)."""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from .elm327 import ELM327, SerialTransport, TCPTransport
from .simulator import SimTransport, VehicleModel


class OBDSession:
    def __init__(self, elm: ELM327, kind: str, sim: Optional[SimTransport] = None):
        self.elm = elm
        self.kind = kind          # "simulator" | "serial" | "tcp"
        self.sim = sim
        self.info = elm.initialize()
        self.vin = elm.read_vin()

    # -------------------------------------------------------------- factory
    @classmethod
    def simulator(cls, scenario: str = "normal", seed: Optional[int] = None, realtime: bool = True,
                  severity: float = 1.0) -> "OBDSession":
        sim = SimTransport(VehicleModel(scenario, seed=seed, severity=severity), realtime=realtime)
        return cls(ELM327(sim), "simulator", sim)

    @classmethod
    def serial(cls, port: str, baudrate: int = 38400) -> "OBDSession":
        return cls(ELM327(SerialTransport(port, baudrate)), "serial")

    @classmethod
    def tcp(cls, host: str, port: int) -> "OBDSession":
        return cls(ELM327(TCPTransport(host, port)), "tcp")

    # -------------------------------------------------------------- api
    def live(self) -> Dict[str, Optional[float]]:
        return self.elm.read_live()

    def capture(self, seconds: float = 60.0, dt: float = 0.25) -> List[Dict]:
        """Record a live-data window.

        Simulator: virtual clock (instant). Real adapter: wall clock; dt is
        whatever the bus allows (~0.5-2 s per sweep on ELM327 clones)."""
        samples: List[Dict] = []
        if self.sim is not None:
            prev_rt = self.sim.realtime
            self.sim.realtime = False
            t = 0.0
            while t < seconds:
                self.sim.tick(dt)
                s = self.elm.read_live()
                s["t"] = round(t, 3)
                samples.append(s)
                t += dt
            self.sim.realtime = prev_rt
            return samples
        start = time.time()
        while time.time() - start < seconds:
            s = self.elm.read_live()
            s["t"] = round(time.time() - start, 3)
            samples.append(s)
        return samples

    def effective_dt(self, samples: List[Dict]) -> float:
        if len(samples) < 2:
            return 1.0
        return (samples[-1]["t"] - samples[0]["t"]) / (len(samples) - 1)

    def close(self) -> None:
        self.elm.close()
