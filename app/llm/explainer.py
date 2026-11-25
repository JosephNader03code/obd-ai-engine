"""Explanation layer = (LLM or offline narrator) + anti-hallucination guard."""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

from ..ml.model import CLASS_AR
from .providers import LLMProvider, SYSTEM_PROMPT_AR, SYSTEM_PROMPT_EN, get_provider

_DTC_RE = re.compile(r"\b[PCBU][0-3][0-9A-F]{3}\b")
_LIK_ORDER = {"low": 0, "medium": 1, "high": 2}


def _lik(conf: float) -> str:
    return "high" if conf >= 0.75 else "medium" if conf >= 0.5 else "low"


# ====================================================================== mock
def mock_explain(pkg: Dict) -> Dict:
    """Deterministic narrator: turns evidence into Arabic text without any LLM.
    Used offline and as fallback when a real LLM answer fails the guard."""
    ev = pkg["evidence"]
    dtc_ids = {v["code"]: k for k, v in ev.items() if v["type"] == "dtc"}
    kb_ids = {v["doc_id"]: k for k, v in ev.items() if v["type"] == "kb"}
    ml_ids = {v["class"]: k for k, v in ev.items() if v["type"] == "ml"}
    rules = [(k, v) for k, v in ev.items() if v["type"] == "rule"]

    causes: List[Dict] = []
    steps: List[Dict] = []
    covered_codes = set()
    seen_tests = set()
    for rid, r in rules:
        refs = [rid]
        for e in r["evidence"]:
            for code in _DTC_RE.findall(e):
                if code in dtc_ids:
                    refs.append(dtc_ids[code])
                    covered_codes.add(code)
        for doc_id, kid in kb_ids.items():
            if doc_id in pkg.get("_rule_kb", {}).get(r["rule_id"], []):
                refs.append(kid)
        if r["hypothesis"] in ml_ids:
            refs.append(ml_ids[r["hypothesis"]])
        why = "القياسات اللي أكدت القاعدة: " + "، ".join(r["evidence"])
        if r["hypothesis"] in ml_ids:
            m = ev[ml_ids[r["hypothesis"]]]
            why += f". نموذج الـ ML كمان بيرجّح «{m['class_ar']}» بنسبة {m['probability'] * 100:.0f}%"
        causes.append({"cause": r["title"], "likelihood": _lik(r["confidence"]), "evidence": sorted(set(refs)),
                       "why": why})
        for t in r["tests"]:
            if t not in seen_tests and t in pkg["tests"]:
                seen_tests.add(t)
                steps.append({"step": f"{pkg['tests'][t]['title']}: {pkg['tests'][t]['how']}", "test_id": t,
                              "evidence": [rid]})

    # DTCs with no rule explanation -> causes from the knowledge base (low)
    for code, did in dtc_ids.items():
        if code in covered_codes:
            continue
        kb = next((k for k, v in ev.items() if v["type"] == "kb" and v.get("code") == code), None)
        if kb:
            c = ev[kb]["causes"][:3]
            causes.append({"cause": f"الكود {code}: الأسباب الشائعة حسب المرجع الفني ({'، '.join(c)})",
                           "likelihood": "low", "evidence": [did, kb],
                           "why": "الكود موجود لكن القياسات الحالية مش كافية لتأكيد سبب محدد."})

    # ML-only signals
    rule_hyps = {r["hypothesis"] for _, r in rules}
    for cls, mid in ml_ids.items():
        m = ev[mid]
        if cls != "normal" and cls not in rule_hyps and m["probability"] >= 0.5:
            causes.append({"cause": f"نمط بيانات يشبه «{m['class_ar']}» (من نموذج الـ ML فقط)", "likelihood": "low",
                           "evidence": [mid], "why": "مفيش قاعدة أكدت ده، فاعتبره مؤشر يحتاج تأكيد بالفحص."})

    if not causes:
        anomaly = next((v["anomaly_score"] for v in ev.values() if v["type"] == "ml"), 0)
        summary_ar = ("مفيش أكواد أعطال مسجلة، والقياسات في الحدود الطبيعية. "
                      f"درجة اختلاف البيانات عن السيارة السليمة: {anomaly:.2f} (من 0 لـ 1).")
        summary_en = ("No stored DTCs and sensor readings are within normal range. "
                      f"Data deviation from healthy baseline: {anomaly:.2f} (0-1 scale).")
    else:
        top = max(causes, key=lambda c: _LIK_ORDER[c["likelihood"]])
        codes = "، ".join(dtc_ids) or "لا يوجد"
        summary_ar = f"الأكواد المقروءة: {codes}. أقوى احتمال حسب الأدلة: {top['cause']}."
        summary_en = f"DTCs read: {', '.join(dtc_ids) or 'None'}. Most likely cause based on evidence: {top['cause']}."
        if any(r["rule_id"] == "R_CAT_SECONDARY_LEAN" for _, r in rules):
            summary_ar += " مهم: الأدلة بتقول إن الخلط الفقير لازم يتصلح الأول قبل ما تحكم على الكتلايزر أو تغيّره."
            summary_en += " Important: evidence suggests fixing the lean condition first before condemning the catalyst."

    critical = any(r["severity"] == "critical" for _, r in rules)
    high = any(r["severity"] == "high" for _, r in rules)
    if critical:
        safety_ar = ("تحذير عاجل: في عطل خطير! لو الحرارة عالية وقّف فوراً واطفي المحرك عشان متعملش ضرر أكبر. "
                     "لو لمبة المحرك بتفلش وقّف السواقة فوراً. متكملش سواقة لحد ما يتفحص.")
        safety_en = ("URGENT WARNING: Critical fault detected! If temperature is high, stop immediately and turn off the engine "
                     "to prevent further damage. If the check engine light is flashing, stop driving immediately.")
    elif high:
        safety_ar = "في عطل شدته عالية: تجنّب الحمل العالي والسرعات العالية لحد ما يتفحص. لو لمبة المحرك بتفلش وقّف السواقة."
        safety_en = "High severity fault detected: avoid heavy loads and high speeds until inspected. If the check engine light is flashing, stop driving."
    else:
        safety_ar = "مفيش خطر فوري واضح من البيانات، بس متمسحش الأكواد قبل الإصلاح عشان الفني يشوفها."
        safety_en = "No immediate danger indicated by the data, but do not clear codes before repair so the technician can review them."
    missing = pkg.get("_missing", [])
    lim_ar = "نموذج الـ ML متدرب على بيانات محاكاة، فنتايجه دليل مساعد بس."
    lim_en = "ML model is trained on simulated data; its results are supporting evidence only."
    if missing:
        lim_ar += " قياسات ناقصة لم تُستخدم: " + "، ".join(missing) + "."
        lim_en += " Missing measurements not used: " + ", ".join(missing) + "."
    return {"summary": summary_ar, "summary_en": summary_en,
            "likely_causes": causes, "next_steps": steps[:6],
            "safety": safety_ar, "safety_en": safety_en,
            "limitations": lim_ar, "limitations_en": lim_en}


# ===================================================================== guard
def _parse_json(text: str) -> Optional[Dict]:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    else:
        a, b = text.find("{"), text.rfind("}")
        if a >= 0 and b > a:
            text = text[a:b + 1]
    try:
        return json.loads(text)
    except Exception:
        return None


def guard(answer: Dict, pkg: Dict) -> Tuple[Dict, List[str]]:
    """Drop or downgrade anything not backed by evidence."""
    ev = pkg["evidence"]
    valid = set(ev)
    known_codes = {v["code"] for v in ev.values() if v["type"] == "dtc"} | \
                  {v["code"] for v in ev.values() if v["type"] == "kb" and v.get("code")}
    notes: List[str] = []

    def support(refs: List[str]) -> float:
        s = 0.0
        for r in refs:
            v = ev[r]
            if v["type"] == "rule":
                s = max(s, v["confidence"])
            elif v["type"] == "ml":
                s = max(s, min(0.6, v["probability"]))
            elif v["type"] == "dtc":
                s = max(s, 0.45)
        return s

    causes = []
    for c in answer.get("likely_causes", []) or []:
        if not isinstance(c, dict) or not c.get("cause"):
            continue
        refs = [r for r in c.get("evidence", []) if r in valid]
        bad = [r for r in c.get("evidence", []) if r not in valid]
        if bad:
            notes.append(f"حُذفت مراجع غير موجودة {bad} من «{c['cause'][:40]}»")
        if not any(r[0] in "RMD" for r in refs):
            notes.append(f"حُذف سبب بدون دليل من R/M/D: «{c['cause'][:60]}»")
            continue
        unknown = set(_DTC_RE.findall(c["cause"] + " " + c.get("why", ""))) - known_codes
        if unknown:
            notes.append(f"حُذف سبب يذكر أكواد غير مقروءة {sorted(unknown)}")
            continue
        cap = _lik(support(refs))
        lik = c.get("likelihood", "low") if c.get("likelihood") in _LIK_ORDER else "low"
        if _LIK_ORDER[lik] > _LIK_ORDER[cap]:
            notes.append(f"خُفّضت درجة «{c['cause'][:40]}» من {lik} إلى {cap} حسب قوة الدليل")
            lik = cap
        causes.append({"cause": c["cause"], "likelihood": lik, "evidence": refs, "why": c.get("why", "")})

    steps = []
    for s in answer.get("next_steps", []) or []:
        if not isinstance(s, dict) or not s.get("step"):
            continue
        tid = s.get("test_id")
        if tid in ("null", "", None):
            tid = None
        if tid and tid not in pkg["tests"]:
            notes.append(f"اختبار غير معروف {tid} اتشال من الخطوات")
            tid = None
        if set(_DTC_RE.findall(s["step"])) - known_codes:
            notes.append("حُذفت خطوة تذكر كود غير مقروء")
            continue
        steps.append({"step": s["step"], "test_id": tid, "evidence": [r for r in s.get("evidence", []) if r in valid]})

    summary = str(answer.get("summary", ""))
    for code in set(_DTC_RE.findall(summary)) - known_codes:
        summary = summary.replace(code, "[كود غير مؤكد]")
        notes.append(f"اتشال الكود {code} من الملخص لأنه مش مقروء من العربية")
    return {"summary": summary, "likely_causes": causes, "next_steps": steps,
            "safety": str(answer.get("safety", "")), "limitations": str(answer.get("limitations", ""))}, notes


# ================================================================== facade
def explain(pkg: Dict, provider: Optional[LLMProvider] = None, use_env: bool = True) -> Dict:
    if provider is None and use_env:
        try:
            provider = get_provider()
        except Exception as e:  # missing key etc.
            out = mock_explain(pkg)
            out["_meta"] = {"provider": "mock", "fallback_reason": str(e), "guard_notes": []}
            return out
    if provider is None:
        out = mock_explain(pkg)
        out["_meta"] = {"provider": "mock", "guard_notes": []}
        return out
    clean_pkg = {k: v for k, v in pkg.items() if not k.startswith("_")}
    try:
        raw = provider.complete(clean_pkg)
    except Exception as e:
        out = mock_explain(pkg)
        out["_meta"] = {"provider": "mock", "fallback_reason": f"{provider.name} error: {e}", "guard_notes": []}
        return out
    parsed = _parse_json(raw)
    if parsed is None:
        out = mock_explain(pkg)
        out["_meta"] = {"provider": "mock", "fallback_reason": "LLM did not return valid JSON", "guard_notes": []}
        return out
    cleaned, notes = guard(parsed, pkg)
    if not cleaned["likely_causes"] and any(v["type"] in ("rule", "dtc") for v in pkg["evidence"].values()):
        out = mock_explain(pkg)
        out["_meta"] = {"provider": "mock", "fallback_reason": "LLM answer had no evidence-backed causes",
                        "guard_notes": notes}
        return out
    cleaned["_meta"] = {"provider": provider.name, "model": getattr(provider, "model", ""), "guard_notes": notes}
    return cleaned
