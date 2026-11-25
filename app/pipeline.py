"""Diagnosis pipeline: OBD data -> features -> Rules + ML -> RAG -> LLM (guarded)."""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from .features import FEATURE_INFO_AR, FEATURE_INFO_EN, extract_features
from .llm.evidence import build_evidence
from .llm.explainer import explain
from .llm.providers import LLMProvider
from .ml.model import CLASS_AR, FaultModel
from .rag.retriever import KnowledgeBase
from .rules.engine import RuleEngine

HYP_QUERY = {
    "catalyst": "catalyst efficiency downstream oxygen sensor",
    "vacuum_leak": "vacuum leak lean idle unmetered air smoke test",
    "maf_dirty": "mass air flow sensor under reading lean load",
    "fuel_delivery": "fuel pressure pump lean under load",
    "o2_slow": "oxygen sensor slow response switching",
    "misfire": "misfire ignition coil spark plug compression",
    "thermostat": "thermostat coolant temperature warm up",
    "charging": "charging alternator battery voltage",
    "rich": "rich condition injector fuel pressure purge",
    "overheating": "engine overheating coolant temperature radiator water pump",
    "egr_fault": "EGR exhaust gas recirculation insufficient flow carbon",
    "evap_leak": "EVAP evaporative emission leak gas cap canister purge",
    "idle_fault": "idle speed control IAC throttle body rpm unstable",
    "sensor_circuit": "sensor circuit wiring connector voltage signal",
    "speed_sensor": "vehicle speed sensor VSS speedometer transmission",
}


class Diagnoser:
    def __init__(self):
        self.rules = RuleEngine()
        self.model = FaultModel.load()
        self.kb = KnowledgeBase()

    def tests_catalog(self) -> Dict[str, Dict]:
        return {k: v for k, v in self.kb.docs.items() if v["type"] == "test"}

    def run(self, samples: List[Dict], dt: float, dtcs: Dict[str, List[str]], vehicle: Dict,
            displacement_l: float = 1.6, provider: Optional[LLMProvider] = None, use_env_llm: bool = True) -> Dict:
        t0 = time.time()
        all_codes = sorted(set(dtcs.get("stored", []) + dtcs.get("pending", [])))
        features = extract_features(samples, dt, displacement_l)

        findings = [f.to_dict() for f in self.rules.evaluate(all_codes, features)]
        ml = self.model.predict(features)

        # ---- RAG query built from *evidence*, not from free text
        q_parts = [f["title_en"] for f in findings]
        q_parts += [HYP_QUERY.get(f["hypothesis"], "") for f in findings]
        top_ml = [c for c, p in ml["probabilities"].items() if p >= 0.2 and c != "normal"]
        q_parts += [HYP_QUERY.get(c, "") for c in top_ml]
        must = sorted({ref for f in findings for ref in f["kb_refs"]})
        hits = self.kb.search(" ".join(q_parts), codes=all_codes, k=6, must_include=must) if (q_parts or all_codes) else []
        kb_hits = [h.to_dict() for h in hits]

        test_ids: List[str] = []
        for f in findings:
            test_ids += f["tests"]
        for h in kb_hits:
            test_ids += h.get("tests", [])
        catalog = self.tests_catalog()
        tests = {t: catalog[t] for t in dict.fromkeys(test_ids) if t in catalog}

        pkg = build_evidence(vehicle, dtcs, features, findings, ml, kb_hits, tests)
        pkg["_rule_kb"] = {f["rule_id"]: f["kb_refs"] for f in findings}
        pkg["_missing"] = [FEATURE_INFO_AR.get(m, m) for m in ml["missing_features"]]
        explanation = explain(pkg, provider=provider, use_env=use_env_llm)

        return {
            "vehicle": vehicle,
            "dtcs": dtcs,
            "features": features,
            "feature_labels": FEATURE_INFO_AR,
            "feature_labels_en": FEATURE_INFO_EN,
            "rule_findings": findings,
            "ml": {**ml, "class_labels": CLASS_AR},
            "retrieved": [{k: h.get(k) for k in ("id", "type", "code", "title_ar", "title_en", "score",
                                                 "set_conditions_ar", "causes_ar", "text_ar")} for h in kb_hits],
            "tests": {k: {"title_ar": v["title_ar"], "text_ar": v["text_ar"], "tools_ar": v.get("tools_ar")}
                      for k, v in tests.items()},
            "evidence_ids": {k: v for k, v in pkg["evidence"].items()},
            "explanation": explanation,
            "samples": len(samples),
            "sample_dt": round(dt, 3),
            "elapsed_ms": int((time.time() - t0) * 1000),
        }
