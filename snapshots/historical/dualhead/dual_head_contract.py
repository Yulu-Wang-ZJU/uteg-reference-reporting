"""Pure contracts for the Ridge B+predMA dual-head preparation.

This module deliberately contains no scipy/sklearn import and no optimizer.
The preparation runner may validate shapes, masks, weights and gradients, but
cannot train a model or read a test object.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_DIRECTIONS = (
    "HZ+HB_to_SA",
    "SA+HB_to_HZ",
    "SA+HZ_to_HB",
    "HZ_to_SA",
    "HB_to_SA",
    "SA_to_HZ",
    "HB_to_HZ",
    "SA_to_HB",
    "HZ_to_HB",
)
EXPECTED_SEEDS = tuple(range(1000, 1020))
EXPECTED_POSITIONS = (0, 1, 2, 3)


class AuthorizationRequired(RuntimeError):
    """Raised before any future optimizer entry can be reached."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def build_features(
    b: np.ndarray,
    pred_ma: np.ndarray,
    legal_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build the two fixed columns with fold-local median/mean/sd preprocessing.

    Non-finite feature values are imputed from the legal training rows, matching
    the existing LR preprocessing contract.  The legal label/QC mask is not
    silently reduced because a feature is non-finite.
    """

    b = np.asarray(b, dtype=np.float64)
    pred_ma = np.asarray(pred_ma, dtype=np.float64)
    legal_mask = np.asarray(legal_mask, dtype=bool)
    if b.ndim != 1 or pred_ma.ndim != 1 or legal_mask.ndim != 1:
        raise ValueError("B, predMA and legal_mask must be one-dimensional")
    if not (len(b) == len(pred_ma) == len(legal_mask)):
        raise ValueError("B, predMA and legal_mask length mismatch")
    if not np.any(legal_mask):
        raise ValueError("cannot preprocess without legal training rows")

    x = np.column_stack([b, pred_ma])
    finite = np.isfinite(x)
    med = np.empty(2, dtype=np.float64)
    mean = np.empty(2, dtype=np.float64)
    sd = np.empty(2, dtype=np.float64)
    for j in range(2):
        finite_j = finite[legal_mask, j]
        if not np.any(finite_j):
            raise ValueError(f"feature {j} has no finite legal value")
        finite_values = x[legal_mask, j][finite_j]
        # Exact original LR order: statistics use finite raw training values;
        # only after mean/sd are fixed are non-finite values median-filled.
        med[j] = np.median(finite_values)
        mean[j] = np.mean(finite_values)
        sd[j] = max(float(np.std(finite_values, ddof=0)), 1e-6)
    x_filled = np.where(finite, x, med[None, :])
    z = np.clip((x_filled - mean[None, :]) / sd[None, :], -10.0, 10.0)
    return z, {"median": med, "mean": mean, "sd": sd}


def build_head_mask(
    truth: np.ndarray,
    pred_class: np.ndarray,
    pred_ma: np.ndarray,
    known_qc_mask: np.ndarray,
    head: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mask and binary labels for one tail-vs-within head."""

    truth = np.asarray(truth, dtype=np.int64)
    pred_class = np.asarray(pred_class, dtype=np.int64)
    pred_ma = np.asarray(pred_ma, dtype=np.float64)
    known_qc_mask = np.asarray(known_qc_mask, dtype=bool)
    if not (truth.ndim == pred_class.ndim == pred_ma.ndim == known_qc_mask.ndim == 1):
        raise ValueError("head inputs must be one-dimensional")
    if not (len(truth) == len(pred_class) == len(pred_ma) == len(known_qc_mask)):
        raise ValueError("head input length mismatch")
    if head == "low":
        allowed = np.isin(truth, (-1, 0))
        positive = truth == -1
    elif head == "high":
        allowed = np.isin(truth, (0, 1))
        positive = truth == 1
    else:
        raise ValueError("head must be low or high")
    # A numeric within prediction is valid only when its value is finite and
    # its already-derived class is exactly the within class.  In particular,
    # a NaN must never enter through a stale/sentinel class value of zero.
    mask = known_qc_mask & np.isfinite(pred_ma) & (pred_class == 0) & allowed
    return mask, positive[mask].astype(np.float64)


def support_summary(y: np.ndarray, record_ids: np.ndarray | None = None) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.float64)
    if record_ids is not None:
        record_ids = np.asarray(record_ids).astype(str)
        if record_ids.shape != y.shape:
            raise ValueError("record_ids shape mismatch")
    n0 = int(np.sum(y == 0.0))
    n1 = int(np.sum(y == 1.0))
    n = int(y.size)
    duplicate_id_n = None
    unique_class0_n = None
    unique_class1_n = None
    if record_ids is not None:
        duplicate_id_n = int(record_ids.size - np.unique(record_ids).size)
        unique_class0_n = int(np.unique(record_ids[y == 0.0]).size)
        unique_class1_n = int(np.unique(record_ids[y == 1.0]).size)
    return {
        "n": n,
        "class0_n": n0,
        "class1_n": n1,
        "unique_class0_n": unique_class0_n,
        "unique_class1_n": unique_class1_n,
        "duplicate_record_id_n": duplicate_id_n,
        "distinct_class_n": int((n0 > 0) + (n1 > 0)),
        "status": "SUPPORTED" if n >= 8 and n0 >= 2 and n1 >= 2 and (duplicate_id_n in (None, 0)) and (unique_class0_n in (None, 0) or unique_class0_n >= 2) and (unique_class1_n in (None, 0) or unique_class1_n >= 2) else "UNSUPPORTED",
    }


def class_balanced_weights(y: np.ndarray) -> np.ndarray:
    """Equal total weight per observed class; mean weight is exactly one."""

    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n == 0:
        return np.empty(0, dtype=np.float64)
    counts = {cls: int(np.sum(y == cls)) for cls in (0.0, 1.0)}
    if counts[0.0] == 0 or counts[1.0] == 0:
        raise ValueError("both classes are required for a trained head")
    w = np.empty(n, dtype=np.float64)
    for cls, count in counts.items():
        w[y == cls] = n / (2.0 * count)
    return w


def weighted_bce_objective(
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    l2: float = 0.1,
) -> float:
    theta = np.asarray(theta, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if theta.shape != (x.shape[1] + 1,):
        raise ValueError("theta dimension mismatch")
    if not (len(x) == len(y) == len(weights)) or len(x) == 0:
        raise ValueError("objective rows mismatch or empty")
    z = x @ theta[:-1] + theta[-1]
    # Stable BCE with labels in {0,1}; no intercept penalty.
    bce = np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))
    return float(np.mean(weights * bce) + l2 * np.sum(theta[:-1] ** 2))


def weighted_bce_gradient(
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    l2: float = 0.1,
) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if theta.shape != (x.shape[1] + 1,):
        raise ValueError("theta dimension mismatch")
    z = x @ theta[:-1] + theta[-1]
    residual = weights * (sigmoid(z) - y) / len(y)
    grad = np.empty_like(theta)
    grad[:-1] = x.T @ residual + 2.0 * l2 * theta[:-1]
    grad[-1] = np.sum(residual)
    return grad


def score_head(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("score input must have shape (n, 2)")
    if x.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    return sigmoid(x @ np.asarray(theta[:-1]) + theta[-1])


def checkpoint_name(direction: str, seed: int, position: int, head: str) -> str:
    safe_direction = "".join(c if c.isalnum() else "_" for c in direction)
    safe_head = "lo" if head == "low" else "hi"
    name = f"{safe_direction[:24]}__s{int(seed)}__p{int(position)}__h{safe_head}.json"
    if len(name) > 120:
        raise ValueError("checkpoint filename exceeds preparation limit")
    return name


def fit_head_guard(*_args: Any, **_kwargs: Any) -> None:
    """Fail closed; this preparation package cannot call an optimizer."""

    raise AuthorizationRequired(
        "NEW_ROUND_AUTHORIZATION_REQUIRED: optimizer entry disabled in prepare run"
    )


def validate_metadata(path: Path) -> dict[str, Any]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise AssertionError("preflight metadata must be a list")
    required = {
        "direction",
        "seed",
        "position",
        "feature_dim",
        "known_n",
        "legal_n",
        "score_valid_n",
        "test_labels_read",
        "inner_folds",
    }
    keys = Counter()
    for row in rows:
        if not required.issubset(row):
            raise AssertionError("preflight row missing required identity field")
        if row["test_labels_read"] is not False:
            raise AssertionError("metadata contains a test-label access marker")
        forbidden = {"test", "medical", "eval", "evaluation"}
        if forbidden.intersection(row):
            raise AssertionError("test/evaluation payload present in metadata row")
        if int(row["feature_dim"]) != 13:
            raise AssertionError("base MA feature dimension drift")
        if int(row["legal_n"]) > int(row["known_n"]):
            raise AssertionError("legal rows exceed known rows")
        keys[(row["direction"], int(row["seed"]), int(row["position"]))] += 1
    expected = {
        (d, s, p)
        for d in EXPECTED_DIRECTIONS
        for s in EXPECTED_SEEDS
        for p in EXPECTED_POSITIONS
    }
    if len(rows) != 720 or set(keys) != expected or any(v != 1 for v in keys.values()):
        raise AssertionError("direction/seed/position metadata does not cover 720 unique positions")
    return {
        "rows": len(rows),
        "directions": len(EXPECTED_DIRECTIONS),
        "seeds": len(EXPECTED_SEEDS),
        "positions": len(EXPECTED_POSITIONS),
        "feature_dim": 13,
        "known_equals_legal_rows": int(
            sum(1 for row in rows if int(row["known_n"]) == int(row["legal_n"])
        )),
        "test_labels_read_false_rows": int(sum(1 for row in rows if row["test_labels_read"] is False)),
        "actual_inner_v_support_counts": "NOT_AVAILABLE_IN_METADATA",
        "support_count_action": "do_not_fabricate; recover from legal inner-V OOF only after authorization",
    }
