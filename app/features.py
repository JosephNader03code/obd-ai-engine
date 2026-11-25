"""Turn a time series of live OBD samples into diagnostic features.

The same features feed the Rule Engine, the ML model and the LLM evidence
package, so every number the LLM sees can be traced back to raw data.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

AIR_DENSITY = 1.184
VE = 0.85

FEATURE_NAMES = [
    "ect_max", "ect_rate", "ltft_mean", "stft_mean",
    "trim_idle", "trim_cruise", "trim_high",
    "o2_up_switch_rate", "o2_up_range", "o2_down_std", "o2_down_mean", "o2_corr",
    "maf_ratio", "rpm_idle_std", "rpm_idle_mean", "voltage_running", "map_idle",
    "load_idle", "iat_value", "speed_zero_pct",
]

FEATURE_INFO_AR = {
    "ect_max": "أعلى حرارة مياه (°C)",
    "ect_rate": "معدل ارتفاع حرارة المياه (°C/دقيقة)",
    "ltft_mean": "متوسط التصحيح طويل المدى LTFT (%)",
    "stft_mean": "متوسط التصحيح قصير المدى STFT (%)",
    "trim_idle": "إجمالي تصحيح الوقود في الوضع الثابت (%)",
    "trim_cruise": "إجمالي تصحيح الوقود في السير العادي (%)",
    "trim_high": "إجمالي تصحيح الوقود تحت الحمل العالي (%)",
    "o2_up_switch_rate": "معدل تبديل حساس الأكسجين الأمامي (مرة/دقيقة)",
    "o2_up_range": "مدى جهد حساس الأكسجين الأمامي (V)",
    "o2_down_std": "تذبذب حساس الأكسجين الخلفي (V)",
    "o2_down_mean": "متوسط جهد حساس الأكسجين الخلفي (V)",
    "o2_corr": "تشابه إشارة الحساس الخلفي مع الأمامي (0-1)",
    "maf_ratio": "نسبة قراءة MAF إلى القيمة المتوقعة",
    "rpm_idle_std": "تذبذب اللفات في الوضع الثابت (rpm)",
    "rpm_idle_mean": "متوسط اللفات في الوضع الثابت (rpm)",
    "voltage_running": "جهد البطارية أثناء الدوران (V)",
    "map_idle": "ضغط المنفولد في الوضع الثابت (kPa)",
    "load_idle": "حمل المحرك في الوضع الثابت (%)",
    "iat_value": "درجة حرارة هواء السحب (°C)",
    "speed_zero_pct": "نسبة الوقت بسرعة صفر مع لفات عالية (%)",
}

FEATURE_INFO_EN = {
    "ect_max": "Max coolant temp (°C)",
    "ect_rate": "Coolant temp rise rate (°C/min)",
    "ltft_mean": "Mean long-term fuel trim LTFT (%)",
    "stft_mean": "Mean short-term fuel trim STFT (%)",
    "trim_idle": "Total fuel trim at idle (%)",
    "trim_cruise": "Total fuel trim at cruise (%)",
    "trim_high": "Total fuel trim under high load (%)",
    "o2_up_switch_rate": "Upstream O2 switch rate (crosses/min)",
    "o2_up_range": "Upstream O2 voltage range (V)",
    "o2_down_std": "Downstream O2 voltage std dev (V)",
    "o2_down_mean": "Downstream O2 mean voltage (V)",
    "o2_corr": "Downstream/upstream O2 correlation (0-1)",
    "maf_ratio": "MAF actual/expected ratio",
    "rpm_idle_std": "Idle RPM std dev (rpm)",
    "rpm_idle_mean": "Mean idle RPM (rpm)",
    "voltage_running": "Battery voltage while running (V)",
    "map_idle": "Manifold pressure at idle (kPa)",
    "load_idle": "Engine load at idle (%)",
    "iat_value": "Intake air temperature (°C)",
    "speed_zero_pct": "Zero-speed at high RPM ratio (%)",
}


def _arr(samples: List[Dict], key: str) -> np.ndarray:
    return np.array([s[key] for s in samples if s.get(key) is not None], dtype=float)


def _mean(x: np.ndarray) -> Optional[float]:
    return float(np.mean(x)) if x.size else None


def _switches(x: np.ndarray, thr: float = 0.45) -> int:
    if x.size < 2:
        return 0
    above = x > thr
    return int(np.sum(above[1:] != above[:-1]))


def extract_features(samples: List[Dict], dt: float, displacement_l: float = 1.6) -> Dict[str, Optional[float]]:
    f: Dict[str, Optional[float]] = {k: None for k in FEATURE_NAMES}
    if not samples:
        return f
    rpm = np.array([s.get("rpm") or 0 for s in samples], dtype=float)
    stft = np.array([s.get("stft") if s.get("stft") is not None else np.nan for s in samples], dtype=float)
    ltft = np.array([s.get("ltft") if s.get("ltft") is not None else np.nan for s in samples], dtype=float)
    trim = stft + ltft

    speed = np.array([s.get("speed") or 0 for s in samples], dtype=float)
    idle = (rpm > 400) & (rpm < 1000) & (speed < 2)
    cruise = (rpm >= 1800) & (rpm < 2700)
    high = rpm >= 2700
    running = rpm > 400

    ect = _arr(samples, "ect")
    f["ect_max"] = float(ect.max()) if ect.size else None
    if ect.size >= 4 and dt > 0:
        slope_per_sample = float(np.polyfit(np.arange(ect.size), ect, 1)[0])
        f["ect_rate"] = slope_per_sample * (60 / dt)
    f["ltft_mean"] = float(np.nanmean(ltft)) if np.isfinite(ltft).any() else None
    f["stft_mean"] = float(np.nanmean(stft)) if np.isfinite(stft).any() else None
    for name, mask in (("trim_idle", idle), ("trim_cruise", cruise), ("trim_high", high)):
        vals = trim[mask & np.isfinite(trim)]
        f[name] = float(vals.mean()) if vals.size >= 3 else None

    up = np.array([s.get("o2_up") for s in samples if s.get("o2_up") is not None and (s.get("rpm") or 0) > 400], dtype=float)
    down = np.array([s.get("o2_down") for s in samples if s.get("o2_down") is not None and (s.get("rpm") or 0) > 400], dtype=float)
    duration_min = max(len(up) * dt / 60, 1e-6)
    if up.size >= 10:
        f["o2_up_switch_rate"] = _switches(up) / duration_min
        f["o2_up_range"] = float(np.percentile(up, 95) - np.percentile(up, 5))
    if down.size >= 10:
        f["o2_down_std"] = float(np.std(down))
        f["o2_down_mean"] = float(np.mean(down))
    if up.size >= 10 and down.size == up.size:
        # max correlation over small lags (downstream follows upstream with delay)
        best = 0.0
        for lag in range(0, 4):
            a, b = up[: len(up) - lag], down[lag:]
            if np.std(a) > 1e-6 and np.std(b) > 1e-6:
                best = max(best, float(np.corrcoef(a, b)[0, 1]))
        f["o2_corr"] = max(0.0, best)

    # MAF plausibility vs speed-density estimate
    ratios = []
    for s in samples:
        if all(s.get(k) is not None for k in ("maf", "map", "rpm")) and s["rpm"] > 400 and s["map"] > 10:
            exp = s["rpm"] / 120 * displacement_l * (s["map"] / 101.3) * AIR_DENSITY * VE
            if exp > 0.5:
                ratios.append(s["maf"] / exp)
    f["maf_ratio"] = float(np.median(ratios)) if ratios else None

    if idle.sum() >= 5:
        f["rpm_idle_std"] = float(np.std(rpm[idle]))
        f["rpm_idle_mean"] = float(np.mean(rpm[idle]))
        mp = np.array([s.get("map") for s, m in zip(samples, idle) if m and s.get("map") is not None], dtype=float)
        f["map_idle"] = _mean(mp)
        ld = np.array([s.get("load") for s, m in zip(samples, idle) if m and s.get("load") is not None], dtype=float)
        f["load_idle"] = _mean(ld)
    volt = np.array([s.get("voltage") for s, m in zip(samples, running) if m and s.get("voltage") is not None], dtype=float)
    f["voltage_running"] = _mean(volt)

    iat = _arr(samples, "iat")
    f["iat_value"] = float(np.mean(iat)) if iat.size else None

    if running.sum() >= 5:
        spd_running = speed[running]
        rpm_running = rpm[running]
        stuck = (spd_running < 1) & (rpm_running > 1500)
        f["speed_zero_pct"] = float(np.sum(stuck) / len(spd_running) * 100)

    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in f.items()}
