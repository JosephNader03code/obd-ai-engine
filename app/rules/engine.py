"""Deterministic Rule Engine.

Rules live in rules.json (editable by a mechanic, no code needed).
Condition grammar::

    {"all": [cond, ...]}   {"any": [cond, ...]}   {"not": cond}
    {"dtc": "P0420"}                      # stored or pending
    {"dtc_prefix": "P030"}                # any code starting with
    {"feature": "ltft_mean", "op": ">", "value": 10}
    {"feature": "maf_ratio", "op": "between", "value": [0.9, 1.1]}

A feature that could not be measured (None) makes its comparison False and is
reported in `missing`, so the engine never guesses.
"""
from __future__ import annotations

import json
import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le,
        "==": operator.eq, "!=": operator.ne}

RULES_PATH = Path(__file__).with_name("rules.json")


@dataclass
class Finding:
    rule_id: str
    title_ar: str
    title_en: str
    hypothesis: str         # machine label (aligned with ML classes when possible)
    confidence: float
    severity: str
    evidence: List[str] = field(default_factory=list)
    tests: List[str] = field(default_factory=list)
    kb_refs: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class RuleEngine:
    def __init__(self, path: Path = RULES_PATH):
        self.rules = json.loads(Path(path).read_text(encoding="utf-8"))["rules"]

    def _eval(self, cond: Dict, ctx: Dict, ev: List[str], missing: List[str]) -> bool:
        if "all" in cond:
            res = [self._eval(c, ctx, ev, missing) for c in cond["all"]]
            return all(res)
        if "any" in cond:
            res = [self._eval(c, ctx, ev, missing) for c in cond["any"]]
            return any(res)
        if "not" in cond:
            return not self._eval(cond["not"], ctx, [], missing)
        if "dtc" in cond:
            code = cond["dtc"].upper()
            hit = code in ctx["dtcs"]
            if hit:
                ev.append(f"DTC {code} موجود")
            return hit
        if "dtc_prefix" in cond:
            hits = [c for c in ctx["dtcs"] if c.startswith(cond["dtc_prefix"].upper())]
            if hits:
                ev.append("DTC " + ", ".join(hits))
            return bool(hits)
        if "feature" in cond:
            name, op, val = cond["feature"], cond["op"], cond["value"]
            x = ctx["features"].get(name)
            if x is None:
                missing.append(name)
                return False
            if op == "between":
                ok = val[0] <= x <= val[1]
                desc = f"{name}={x:.2f} بين {val[0]} و {val[1]}"
            else:
                ok = _OPS[op](x, val)
                desc = f"{name}={x:.2f} {op} {val}"
            if ok:
                ev.append(desc)
            return ok
        raise ValueError(f"bad condition {cond}")

    def evaluate(self, dtcs: List[str], features: Dict[str, Optional[float]]) -> List[Finding]:
        ctx = {"dtcs": {d.upper() for d in dtcs}, "features": features}
        out: List[Finding] = []
        for r in self.rules:
            ev: List[str] = []
            missing: List[str] = []
            if self._eval(r["when"], ctx, ev, missing):
                out.append(Finding(
                    rule_id=r["id"], title_ar=r["title_ar"], title_en=r["title_en"],
                    hypothesis=r["hypothesis"], confidence=r["confidence"],
                    severity=r.get("severity", "medium"), evidence=ev,
                    tests=r.get("tests", []), kb_refs=r.get("kb_refs", []),
                    missing=sorted(set(missing)),
                ))
        # rules may suppress weaker ones (e.g. "fix lean first, cat may be OK")
        suppressed = set()
        for r in self.rules:
            if any(f.rule_id == r["id"] for f in out):
                suppressed.update(r.get("suppresses", []))
        out = [f for f in out if f.rule_id not in suppressed]
        out.sort(key=lambda f: -f.confidence)
        return out
