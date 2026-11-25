"""Builds the evidence package the LLM is allowed to talk about.

Every fact gets an id (D=DTC, R=rule finding, M=ML result, K=knowledge doc,
F=measured feature, T=test procedure). The LLM must cite ids; the guard
rejects anything that cites nothing or cites ids that do not exist.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..features import FEATURE_INFO_AR, FEATURE_INFO_EN
from ..ml.model import CLASS_AR


def build_evidence(vehicle: Dict, dtcs: Dict[str, List[str]], features: Dict[str, Optional[float]],
                   findings: List[Dict], ml: Dict, kb_hits: List[Dict], tests: Dict[str, Dict]) -> Dict:
    ev: Dict[str, Dict] = {}
    for i, code in enumerate(sorted(set(dtcs.get("stored", []) + dtcs.get("pending", []))), 1):
        status = "stored" if code in dtcs.get("stored", []) else "pending"
        ev[f"D{i}"] = {"type": "dtc", "code": code, "status": status}
    for i, f in enumerate(findings, 1):
        ev[f"R{i}"] = {"type": "rule", "rule_id": f["rule_id"], "title": f["title_ar"], "title_en": f["title_en"],
                       "hypothesis": f["hypothesis"],
                       "confidence": f["confidence"], "evidence": f["evidence"], "tests": f["tests"],
                       "severity": f["severity"]}
    top = [(c, p) for c, p in ml["probabilities"].items() if p >= 0.1][:3]
    for i, (c, p) in enumerate(top, 1):
        ev[f"M{i}"] = {"type": "ml", "class": c, "class_ar": CLASS_AR.get(c, c), "probability": p,
                       "anomaly_score": ml["anomaly_score"], "top_features": ml["top_features"],
                       "note": "model trained on synthetic data" if ml.get("trained_on") == "synthetic-simulator" else ""}
    for i, h in enumerate(kb_hits, 1):
        ev[f"K{i}"] = {"type": "kb", "doc_id": h["id"], "title": h.get("title_ar"), "code": h.get("code"),
                       "set_conditions": h.get("set_conditions_ar") or "", "causes": h.get("causes_ar", []),
                       "text": h.get("text_ar") or h.get("text_en", "")}
    feats = {k: {"value": v, "label": FEATURE_INFO_AR.get(k, k), "label_en": FEATURE_INFO_EN.get(k, k)}
             for k, v in features.items() if v is not None}
    return {"vehicle": vehicle, "evidence": ev, "features": feats,
            "tests": {tid: {"title": t.get("title_ar"), "title_en": t.get("title_en"),
                            "how": t.get("text_ar"), "how_en": t.get("text_en"), "tools": t.get("tools_ar")}
                      for tid, t in tests.items()}}
