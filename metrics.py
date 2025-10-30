from __future__ import annotations

import json
from typing import Iterable, Optional

import numpy as np


def smape_np(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Compute Symmetric Mean Absolute Percentage Error (SMAPE) in percent.

    SMAPE = 100 * mean( 2*|y_pred - y_true| / (|y_true| + |y_pred|) )
    When denominator is 0 for a pair, treat it as 1 to avoid NaN.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    denom = np.abs(yt) + np.abs(yp)
    denom = np.where(denom == 0.0, 1.0, denom)
    return float(100.0 * np.mean(2.0 * np.abs(yp - yt) / denom))


def save_json(path: str, obj: dict, ensure_ascii: bool = False) -> None:
    """Save a dictionary to JSON with UTF-8 encoding."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=ensure_ascii, indent=2)


__all__ = ["smape_np", "save_json"]

