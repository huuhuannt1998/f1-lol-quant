"""Small shared helpers: name matching, time, and scoring metrics."""
from __future__ import annotations

import math
import re
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher

import numpy as np

# ----------------------------------------------------------------- names

_DROP = {"esports", "esport", "gaming", "team", "club", "e-sports", "fc", "academy_"}


def norm_name(s: str | None) -> str:
    """'Movistar KOI Fénix' -> 'movistar koi fenix'. Used as a join key."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def loose_name(s: str | None) -> str:
    """Even looser key: drop filler words like 'Esports' and 'Gaming'."""
    return " ".join(w for w in norm_name(s).split() if w not in _DROP)


def name_similarity(a: str, b: str) -> float:
    a, b = loose_name(a), loose_name(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.9
    return SequenceMatcher(None, a, b).ratio()


# ------------------------------------------------------------------ time

def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def iso(dt: datetime | None) -> str | None:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if dt else None


# --------------------------------------------------------------- metrics
# Probability forecasts are judged on calibration, not just accuracy.
# Log loss punishes confident mistakes hard; Brier is the mean squared error
# of the probability. Lower is better for both.

EPS = 1e-6


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def accuracy(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) >= 0.5) == (np.asarray(y) == 1)))


def calibration_bins(p, y, n_bins: int = 10) -> list[dict]:
    """Bucket forecasts by predicted probability and compare to the hit rate.
    A calibrated model's 70% bucket should win about 70% of the time."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.sum():
            out.append({"lo": round(lo, 2), "hi": round(hi, 2), "n": int(m.sum()),
                        "mean_p": round(float(p[m].mean()), 4),
                        "hit_rate": round(float(y[m].mean()), 4)})
    return out


def calibration_slope(p, y, min_n: int = 20) -> float | None:
    """One number for "too confident or not confident enough".

    Fit logit(P(win)) = b * logit(p) by maximum likelihood. b = 1 means the
    forecasts are as extreme as they should be; b > 1 means the model is
    under-confident (reality is more lopsided than it says, so stretch it);
    b < 1 means over-confident. Each match is used from both teams' sides,
    which removes the intercept (which side is "team1" is arbitrary)."""
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    y = np.asarray(y, float)
    if len(p) < min_n:
        return None
    x = np.log(p / (1 - p))
    x, y = np.r_[x, -x], np.r_[y, 1 - y]
    b = 1.0
    with np.errstate(over="ignore"):         # exp(-b*x) may overflow to inf: q -> 0, harmless
        for _ in range(100):                 # Newton's method; the log-likelihood is concave in b
            q = 1 / (1 + np.exp(-b * x))
            grad = np.sum((y - q) * x)
            hess = np.sum(q * (1 - q) * x * x)
            if not hess > 0:                 # all p = 0.5: the slope is undefined
                return None
            step = float(np.clip(grad / hess, -1, 1))
            b += step
            if abs(step) < 1e-9:
                return round(b, 3)
    return None                              # no convergence (e.g. every favourite won)


def summarize(p, y) -> dict:
    p, y = np.asarray(p, float), np.asarray(y, float)
    if len(p) == 0:
        return {"n": 0}
    return {"n": int(len(p)), "log_loss": round(log_loss(p, y), 4),
            "brier": round(brier(p, y), 4), "accuracy": round(accuracy(p, y), 4)}


def nonfinite_counts(df, cols) -> dict:
    """NaN/inf count per model column (only columns that have any).

    Printed every run: it is what lets us silence the macOS matmul
    warnings without hiding a real overflow or a NaN feature."""
    cols = list(cols)
    bad = ~np.isfinite(df[cols].to_numpy(dtype=float))
    return {c: int(n) for c, n in zip(cols, bad.sum(axis=0)) if n}


def kalshi_fee(price: float, rate: float) -> float:
    """Approximate per-contract fee, rounded up to the cent."""
    return math.ceil(rate * price * (1 - price) * 100) / 100


def has_lightgbm() -> bool:
    """LightGBM needs OpenMP (on macOS: `brew install libomp`). If it is not
    importable, the gradient-boosting models are skipped instead of crashing."""
    try:
        import lightgbm  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 - ImportError or OSError from libomp
        return False
