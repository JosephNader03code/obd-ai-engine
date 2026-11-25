"""Lightweight RAG retriever (TF-IDF + cosine, pure numpy, runs offline).

Exact DTC-code matches get a strong boost so that reading P0420 always
retrieves the P0420 document first, followed by the procedures that are most
similar to the current evidence (rule findings, ML classes, abnormal
features). Swap `TfidfIndex` for an embedding index later without changing
the pipeline: the interface is `search(query, codes, k) -> [Hit]`.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

KB_DIR = Path(__file__).with_name("kb")
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("the a an and or of to in on at is are be for with by as it if from that this than then".split())


def _tok(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


@dataclass
class Hit:
    doc_id: str
    score: float
    doc: Dict

    def to_dict(self) -> Dict:
        d = {k: v for k, v in self.doc.items()}
        d["score"] = round(self.score, 3)
        return d


class KnowledgeBase:
    def __init__(self, kb_dir: Path = KB_DIR):
        self.docs: Dict[str, Dict] = {}
        for p in sorted(kb_dir.glob("*.json")):
            for d in json.loads(p.read_text(encoding="utf-8")):
                self.docs[d["id"]] = d
        self.ids = [i for i, d in self.docs.items() if d["type"] in ("dtc", "procedure")]
        texts = [self._doc_text(self.docs[i]) for i in self.ids]
        tokens = [_tok(t) for t in texts]
        vocab = sorted({t for ts in tokens for t in ts})
        self.vocab = {t: i for i, t in enumerate(vocab)}
        df = Counter(t for ts in tokens for t in set(ts))
        n = len(tokens)
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1 for t in vocab])
        self.matrix = np.vstack([self._vec(ts) for ts in tokens])

    @staticmethod
    def _doc_text(d: Dict) -> str:
        parts = [d.get("code", ""), d.get("title_en", ""), d.get("set_conditions_en", ""), d.get("text_en", "")]
        return " ".join(parts)

    def _vec(self, toks: List[str]) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        for t, c in Counter(toks).items():
            if t in self.vocab:
                v[self.vocab[t]] = (1 + math.log(c)) * self.idf[self.vocab[t]]
        nrm = np.linalg.norm(v)
        return v / nrm if nrm else v

    def get(self, doc_id: str) -> Dict:
        return self.docs.get(doc_id, {})

    def search(self, query: str, codes: List[str] = (), k: int = 5, must_include: List[str] = ()) -> List[Hit]:
        q = self._vec(_tok(query + " " + " ".join(codes)))
        scores = self.matrix @ q
        codes_u = {c.upper() for c in codes}
        for i, doc_id in enumerate(self.ids):
            d = self.docs[doc_id]
            if d.get("code") in codes_u:
                scores[i] += 1.0                     # exact DTC hit
            elif d.get("code") and codes_u and d["code"][:4] in {c[:4] for c in codes_u}:
                scores[i] += 0.3                     # same family (e.g. P030x)
            if doc_id in must_include:
                scores[i] += 0.6                     # referenced by a fired rule
        order = np.argsort(-scores)
        hits = [Hit(self.ids[i], float(scores[i]), self.docs[self.ids[i]]) for i in order[:k] if scores[i] > 0.15]
        return hits
