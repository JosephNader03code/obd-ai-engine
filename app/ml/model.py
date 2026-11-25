"""ML component: fault-pattern classifier + anomaly score (pure numpy).

* Multinomial logistic regression over the engineered features.
* Mahalanobis-style anomaly score vs. the 'normal' class (how far is this car
  from a healthy one, independent of the class labels).

IMPORTANT: the shipped weights are trained on the simulator (synthetic data).
They demonstrate the pipeline; retrain with `python -m app.ml.train --csv`
on real labelled captures before trusting the probabilities.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..features import FEATURE_NAMES

MODEL_PATH = Path(__file__).with_name("model.npz")

CLASSES = ["normal", "catalyst", "vacuum_leak", "maf_dirty", "fuel_delivery",
           "o2_slow", "misfire", "thermostat", "charging",
           "overheating", "egr_fault", "evap_leak", "idle_fault",
           "sensor_circuit", "speed_sensor"]

CLASS_AR = {
    "normal": "طبيعي", "catalyst": "كتلايزر ضعيف", "vacuum_leak": "تسريب هواء",
    "maf_dirty": "حساس MAF", "fuel_delivery": "إمداد البنزين", "o2_slow": "حساس أكسجين بطيء",
    "misfire": "تقطيع", "thermostat": "ثرموستات", "charging": "نظام الشحن",
    "overheating": "ارتفاع حرارة المحرك", "egr_fault": "عطل EGR",
    "evap_leak": "تسريب EVAP", "idle_fault": "عدم استقرار الثبات",
    "sensor_circuit": "عطل دائرة حساس", "speed_sensor": "عطل حساس السرعة",
}


class FaultModel:
    def __init__(self, W: np.ndarray, b: np.ndarray, mu: np.ndarray, sd: np.ndarray,
                 impute: np.ndarray, normal_mu: np.ndarray, normal_icov: np.ndarray,
                 anomaly_ref: float, classes: List[str] = CLASSES):
        self.W, self.b, self.mu, self.sd = W, b, mu, sd
        self.impute, self.normal_mu, self.normal_icov = impute, normal_mu, normal_icov
        self.anomaly_ref = anomaly_ref
        self.classes = list(classes)

    # ------------------------------------------------------------ helpers
    @staticmethod
    def vectorize(features: Dict[str, Optional[float]], impute: np.ndarray) -> np.ndarray:
        x = np.array([np.nan if features.get(n) is None else features[n] for n in FEATURE_NAMES], dtype=float)
        mask = np.isnan(x)
        x[mask] = impute[mask]
        return x

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    # ------------------------------------------------------------ inference
    def predict(self, features: Dict[str, Optional[float]]) -> Dict:
        x = self.vectorize(features, self.impute)
        xs = (x - self.mu) / self.sd
        p = self.softmax(xs @ self.W + self.b)
        d = xs - self.normal_mu
        maha = float(np.sqrt(max(0.0, d @ self.normal_icov @ d)))
        anomaly = float(1 - np.exp(-maha / self.anomaly_ref))
        # top contributing features for the top class (explainability)
        top = int(np.argmax(p))
        contrib = xs * self.W[:, top]
        order = np.argsort(-contrib)[:3]
        missing = [n for n in FEATURE_NAMES if features.get(n) is None]
        ranked = sorted(zip(self.classes, p.tolist()), key=lambda t: -t[1])
        return {
            "probabilities": {c: round(v, 4) for c, v in ranked},
            "top_class": self.classes[top],
            "top_probability": round(float(p[top]), 4),
            "anomaly_score": round(anomaly, 3),
            "top_features": [FEATURE_NAMES[i] for i in order if contrib[i] > 0],
            "missing_features": missing,
            "trained_on": "synthetic-simulator",
        }

    # ------------------------------------------------------------ io
    def save(self, path: Path = MODEL_PATH) -> None:
        np.savez(path, W=self.W, b=self.b, mu=self.mu, sd=self.sd, impute=self.impute,
                 normal_mu=self.normal_mu, normal_icov=self.normal_icov,
                 anomaly_ref=np.array(self.anomaly_ref), classes=np.array(self.classes))

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "FaultModel":
        z = np.load(path, allow_pickle=False)
        return cls(z["W"], z["b"], z["mu"], z["sd"], z["impute"], z["normal_mu"], z["normal_icov"],
                   float(z["anomaly_ref"]), [str(c) for c in z["classes"]])


def fit(X: np.ndarray, y: np.ndarray, classes: List[str] = CLASSES, epochs: int = 3000,
        lr: float = 0.3, l2: float = 1e-3) -> FaultModel:
    """X may contain NaN (unmeasured features)."""
    normal_idx = classes.index("normal")
    impute = np.nanmean(X[y == normal_idx], axis=0)
    impute = np.where(np.isnan(impute), np.nanmean(X, axis=0), impute)
    Xi = np.where(np.isnan(X), impute, X)
    mu, sd = Xi.mean(0), Xi.std(0) + 1e-6
    Xs = (Xi - mu) / sd
    n, d = Xs.shape
    k = len(classes)
    Y = np.eye(k)[y]
    W = np.zeros((d, k))
    b = np.zeros(k)
    for _ in range(epochs):
        P = FaultModel.softmax(Xs @ W + b)
        G = P - Y
        W -= lr * (Xs.T @ G / n + l2 * W)
        b -= lr * G.mean(0)
    Xn = Xs[y == normal_idx]
    nmu = Xn.mean(0)
    cov = np.cov(Xn, rowvar=False) + np.eye(d) * 0.05
    icov = np.linalg.inv(cov)
    dn = np.sqrt(np.einsum("ij,jk,ik->i", Xn - nmu, icov, Xn - nmu))
    ref = float(np.percentile(dn, 95)) * 2
    return FaultModel(W, b, mu, sd, impute, nmu, icov, ref, classes)
