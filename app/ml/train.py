"""Train the fault model.

    python -m app.ml.train                 # synthetic data from the simulator
    python -m app.ml.train --csv data.csv  # real labelled captures (columns = FEATURE_NAMES + label)
"""
from __future__ import annotations

import argparse
import csv
import random

import numpy as np

from ..features import FEATURE_NAMES, extract_features
from ..obd.session import OBDSession
from .model import CLASSES, MODEL_PATH, fit


CLASS_TO_SCENARIO = {
    "sensor_circuit": "sensor_iat",
}


def synthetic_dataset(per_class: int = 40, seconds: float = 60, dt: float = 0.25, seed: int = 7):
    rng = random.Random(seed)
    X, y = [], []
    for ci, cls in enumerate(CLASSES):
        scenario = CLASS_TO_SCENARIO.get(cls, cls)
        for _ in range(per_class):
            sev = 1.0 if cls == "normal" else rng.uniform(0.6, 1.3)
            s = OBDSession.simulator(scenario, seed=rng.randint(0, 10**9), realtime=False, severity=sev)
            samples = s.capture(seconds, dt)
            f = extract_features(samples, dt)
            X.append([np.nan if f[n] is None else f[n] for n in FEATURE_NAMES])
            y.append(ci)
    return np.array(X, dtype=float), np.array(y)


def csv_dataset(path: str):
    X, y = [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            X.append([float(row[n]) if row.get(n) not in (None, "") else np.nan for n in FEATURE_NAMES])
            y.append(CLASSES.index(row["label"]))
    return np.array(X, dtype=float), np.array(y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--per-class", type=int, default=40)
    args = ap.parse_args()
    X, y = csv_dataset(args.csv) if args.csv else synthetic_dataset(args.per_class)
    idx = np.random.default_rng(0).permutation(len(y))
    cut = int(len(y) * 0.8)
    tr, te = idx[:cut], idx[cut:]
    m = fit(X[tr], y[tr])
    correct = sum(m.predict(dict(zip(FEATURE_NAMES, [None if np.isnan(v) else v for v in X[i]])))["top_class"] == CLASSES[y[i]] for i in te)
    print(f"holdout accuracy: {correct}/{len(te)} = {correct / len(te):.3f}")
    m = fit(X, y)
    m.save(MODEL_PATH)
    print(f"saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
