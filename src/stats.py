from __future__ import annotations

import numpy as np
from typing import Optional


def compute_rsi(closes: list[float], n: int = 14) -> Optional[float]:
    if len(closes) < n + 1:
        return None
    arr = np.array(closes, dtype=float)
    deltas = np.diff(arr)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[-n:])
    avg_loss = np.mean(losses[-n:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi)


def realized_vol_pct(closes: list[float], window: int = 30) -> Optional[float]:
    if len(closes) <= window:
        return None
    arr = np.array(closes, dtype=float)
    rets = np.diff(np.log(arr + 1e-9))
    w = rets[-window:]
    if w.size == 0:
        return None
    return float(np.std(w) * np.sqrt(252) * 100)


def beta_vs_benchmark(closes: list[float], bench: list[float]) -> Optional[float]:
    n = min(len(closes), len(bench))
    if n < 60:
        return None
    rc = np.diff(np.log(np.array(closes[-n:], dtype=float) + 1e-9))
    rb = np.diff(np.log(np.array(bench[-n:], dtype=float) + 1e-9))
    if rc.size < 30 or rb.size < 30:
        return None
    cov = np.cov(rc, rb)
    var_b = cov[1, 1]
    if var_b <= 0:
        return None
    beta = cov[0, 1] / var_b
    return float(beta)

