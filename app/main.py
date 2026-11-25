"""FastAPI backend.  Run:  uvicorn app.main:app --reload  (from backend/)"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .obd.simulator import SCENARIOS
from .obd.session import OBDSession
from .pipeline import Diagnoser

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"

app = FastAPI(title="OBD AI Diagnostics", version="1.0")

_allowed_origins_env = os.getenv("CORS_ALLOWED_ORIGINS", "")
_allowed_origins = [o.strip() for o in _allowed_origins_env.split(",") if o.strip()] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

diagnoser = Diagnoser()
_lock = threading.Lock()
state: dict = {"session": None, "displacement": 1.6}


class ConnectReq(BaseModel):
    mode: str = Field("simulator", description="simulator | serial | tcp")
    scenario: str = "normal"
    port: Optional[str] = None          # COM3, /dev/ttyUSB0, /dev/rfcomm0
    baudrate: int = 38400
    host: str = "192.168.0.10"
    tcp_port: int = 35000
    displacement_l: float = 1.6


class DiagnoseReq(BaseModel):
    seconds: float = Field(60, ge=5, le=600)
    dt: float = Field(0.25, ge=0.05, le=5)
    lang: str = Field("ar", description="Response language: ar or en")


class ClearReq(BaseModel):
    confirm: bool = False


def _session() -> OBDSession:
    s = state["session"]
    if s is None:
        raise HTTPException(400, "غير متصل. اتصل بالعربية أو بالمحاكي الأول.")
    return s


@app.get("/api/health")
def health():
    return {"ok": True, "llm_provider": os.getenv("LLM_PROVIDER", "mock")}


@app.get("/api/scenarios")
def scenarios():
    return [{"key": s.key, "title_ar": s.title_ar, "title_en": s.title_en} for s in SCENARIOS.values()]


@app.post("/api/connect")
def connect(req: ConnectReq):
    with _lock:
        if state["session"]:
            state["session"].close()
            state["session"] = None
        try:
            if req.mode == "simulator":
                if req.scenario not in SCENARIOS:
                    raise HTTPException(400, "سيناريو غير معروف")
                s = OBDSession.simulator(req.scenario)
            elif req.mode == "serial":
                if not req.port:
                    raise HTTPException(400, "حدد البورت (مثلاً COM3 أو /dev/ttyUSB0)")
                s = OBDSession.serial(req.port, req.baudrate)
            elif req.mode == "tcp":
                s = OBDSession.tcp(req.host, req.tcp_port)
            else:
                raise HTTPException(400, "mode غير معروف")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(502, f"فشل الاتصال بالأداة: {e}")
        state["session"] = s
        state["displacement"] = req.displacement_l
        return {"connected": True, "kind": s.kind, "vin": s.vin, **s.info,
                "supported_pids": [f"{p:02X}" for p in s.elm.supported]}


@app.post("/api/disconnect")
def disconnect():
    with _lock:
        if state["session"]:
            state["session"].close()
        state["session"] = None
    return {"connected": False}


@app.get("/api/status")
def status():
    s = state["session"]
    if not s:
        return {"connected": False}
    return {"connected": True, "kind": s.kind, "vin": s.vin, **s.info}


@app.get("/api/live")
def live():
    s = _session()
    with _lock:
        data = s.live()
        data["battery"] = s.elm.read_battery_voltage()
    return data


@app.get("/api/dtcs")
def dtcs():
    s = _session()
    with _lock:
        return s.elm.read_dtcs()


@app.post("/api/clear_dtcs")
def clear_dtcs(req: ClearReq):
    if not req.confirm:
        raise HTTPException(400, "لازم تأكيد: مسح الأكواد بيمسح بيانات freeze frame وبيصفّر الـ readiness monitors.")
    s = _session()
    with _lock:
        return {"cleared": s.elm.clear_dtcs()}


@app.post("/api/diagnose")
def diagnose(req: DiagnoseReq):
    s = _session()
    with _lock:
        codes = s.elm.read_dtcs()
        samples = s.capture(req.seconds, req.dt)
    dt = s.effective_dt(samples) if s.kind != "simulator" else req.dt
    vehicle = {"vin": s.vin, "protocol": s.info.get("protocol"), "adapter": s.info.get("version"),
               "source": s.kind, "displacement_l": state["displacement"]}
    result = diagnoser.run(samples, dt, codes, vehicle, state["displacement"])
    result["series"] = {k: [smp.get(k) for smp in samples] for k in ("t", "rpm", "stft", "ltft", "o2_up", "o2_down", "ect", "maf")}
    return result


class AnalyzePayload(BaseModel):
    """Payload sent by the .NET backend for AI analysis."""
    vehicle: dict
    dtcs: list = Field(default_factory=list)
    samples: list = Field(default_factory=list)
    sample_interval: float = 0.25
    displacement_liters: float = 1.6
    lang: str = "ar"


@app.post("/api/analyze")
def analyze(payload: AnalyzePayload):
    """Integration endpoint for the .NET backend.
    Accepts pre-collected OBD data and returns the full diagnosis."""
    dtc_codes: list = []
    for d in payload.dtcs:
        if isinstance(d, dict):
            dtc_codes.append(d.get("code", d.get("Code", "")))
        elif isinstance(d, str):
            dtc_codes.append(d)
    dtcs_dict = {"stored": dtc_codes, "pending": []}

    samples = []
    for s in payload.samples:
        sample = {}
        for k, v in s.items():
            key = k[0].lower() + k[1:] if k and k[0].isupper() else k
            field_map = {
                "rpm": "rpm", "speed": "speed", "engineLoad": "load", "coolantTemp": "ect",
                "intakeAirTemp": "iat", "manifoldPressure": "map", "massAirFlow": "maf",
                "throttlePosition": "tps", "shortTermFuelTrim": "stft", "longTermFuelTrim": "ltft",
                "o2SensorUpstream": "o2_up", "o2SensorDownstream": "o2_down", "timingAdvance": "timing",
                "moduleVoltage": "voltage", "barometricPressure": "baro", "catalystTemp": "cat_temp",
                "timestamp": "runtime",
            }
            mapped = field_map.get(key, key)
            sample[mapped] = v
        samples.append(sample)

    vehicle = payload.vehicle
    result = diagnoser.run(
        samples, payload.sample_interval, dtcs_dict, vehicle, payload.displacement_liters
    )
    return result


if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

    @app.get("/")
    def index():
        return FileResponse(FRONTEND / "index.html")
