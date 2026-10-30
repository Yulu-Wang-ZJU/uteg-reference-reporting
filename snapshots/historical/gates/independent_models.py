"""Independent, numpy-only replay of saved trust-gate model objects.

This file intentionally does not import ``risk_models`` or any historical
prediction/evaluation function.  It reconstructs the frozen preprocessing,
RBF posterior, and neural state-dict forward passes directly from the plain
model dictionaries saved by the training driver.  It is used after training
to detect a serialization or inference implementation mismatch.
"""

from __future__ import annotations

import runtime as rt  # task-local paths/cache first

import hashlib
from typing import Any, Mapping, Tuple

import numpy as np
from scipy.special import expit


N_FEATURES = 15
CLASS_VALUES = (-1, 0, 1)
CLASS_TO_INDEX = {-1: 0, 0: 1, 1: 2}


def _matrix(F: Any) -> np.ndarray:
    a = np.asarray(F, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != N_FEATURES:
        raise ValueError(f"F must have shape (n,{N_FEATURES}), got {a.shape}")
    return a


def _labels(y: Any, name: str, n: int) -> np.ndarray:
    a = np.asarray(y)
    if a.ndim != 1 or len(a) != n:
        raise ValueError(f"{name} must have shape ({n},), got {a.shape}")
    if not np.isfinite(a.astype(np.float64)).all():
        raise ValueError(f"{name} contains non-finite labels")
    out = a.astype(np.int64)
    if not np.allclose(a.astype(np.float64), out.astype(np.float64)) or not np.isin(out, CLASS_VALUES).all():
        raise ValueError(f"{name} must contain only {-1, 0, 1}")
    return out


def _transform(pre: Mapping[str, Any], F: Any) -> np.ndarray:
    a = _matrix(F)
    mean = np.asarray(pre["mean"], dtype=np.float64)
    sd = np.asarray(pre["sd"], dtype=np.float64)
    med = np.asarray(pre["median"], dtype=np.float64)
    if mean.shape != (N_FEATURES,) or sd.shape != (N_FEATURES,) or med.shape != (N_FEATURES,):
        raise ValueError("saved preprocessing constants have the wrong shape")
    filled = np.where(np.isfinite(a), a, med[None, :])
    return np.clip((filled - mean[None, :]) / sd[None, :], -10.0, 10.0)


def _rbf(X: np.ndarray, Z: np.ndarray, length: float) -> np.ndarray:
    xx = np.sum(X * X, axis=1)[:, None]
    zz = np.sum(Z * Z, axis=1)[None, :]
    sq = np.maximum(xx + zz - 2.0 * X @ Z.T, 0.0)
    return np.exp(-0.5 * sq / (float(length) ** 2))


def _silu(x: np.ndarray) -> np.ndarray:
    return x * expit(x)


def _linear(x: np.ndarray, weight: Any, bias: Any) -> np.ndarray:
    w = np.asarray(weight, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64)
    return x @ w.T + b[None, :]


def _class_prior(model: Mapping[str, Any], n: int) -> np.ndarray:
    p = np.asarray(model.get("class_prior", [1 / 3, 1 / 3, 1 / 3]), dtype=np.float64)
    p = np.clip(p, 1e-15, np.inf)
    p /= np.sum(p)
    return np.repeat(p[None, :], n, axis=0)


def _fallback_scores(model: Mapping[str, Any], yhat: np.ndarray) -> np.ndarray:
    family = str(model.get("family", ""))
    if family in {"FLAT-CLINICAL", "FACTORED-CLINICAL"}:
        p = _class_prior(model, len(yhat))
        out = np.empty(len(yhat), dtype=np.float64)
        for i, c in enumerate(yhat):
            out[i] = 1.0 - p[i, CLASS_TO_INDEX[int(c)]]
        return out
    return np.full(len(yhat), float(model.get("constant_score", 1.0)), dtype=np.float64)


def _clinical_probs(model: Mapping[str, Any], z: np.ndarray) -> np.ndarray:
    ab = _linear(_silu(_linear(z, model["state"]["fc1.weight"], model["state"]["fc1.bias"])),
                 model["state"]["out.weight"], model["state"]["out.bias"])
    if str(model["family"]) == "FLAT-CLINICAL":
        logits = np.column_stack([ab[:, 0], np.zeros(len(ab), dtype=np.float64), ab[:, 1]])
        logits -= np.max(logits, axis=1, keepdims=True)
        p = np.exp(logits)
        return p / np.sum(p, axis=1, keepdims=True)
    rho = expit(ab[:, 0])
    if bool(model.get("beta_frozen", False)):
        beta = np.full(len(ab), float(model.get("beta_prior", 0.5)), dtype=np.float64)
    else:
        beta = expit(ab[:, 1])
    p = np.column_stack([rho * (1.0 - beta), 1.0 - rho, rho * beta])
    p = np.clip(p, 1e-15, np.inf)
    return p / np.sum(p, axis=1, keepdims=True)


def predict_independent(model: Mapping[str, Any], F: Any, yhat: Any) -> np.ndarray:
    """Replay a saved model and return the same lower-is-safer risk score."""

    a = _matrix(F)
    yh = _labels(yhat, "yhat", len(a))
    z = _transform(model["pre"], a)
    family = str(model.get("family", ""))
    if model.get("status") in {"fallback_const", "exception_fallback"} or not model.get("finite", False):
        return np.clip(_fallback_scores(model, yh), 0.0, 1.0)
    try:
        if family == "ERROR-LR":
            coef = np.asarray(model["coef"], dtype=np.float64)
            out = expit(z @ coef[:-1] + coef[-1])
        elif family == "GP-RISK":
            K = _rbf(z, np.asarray(model["train_z"], dtype=np.float64), float(model["length"]))
            out = float(model["constant_mean"]) + K @ np.asarray(model["alpha"], dtype=np.float64)
            out = np.clip(out, 0.0, 1.0)
        elif family == "DOS-GATE":
            h = _silu(_linear(z, model["state"]["fc1.weight"], model["state"]["fc1.bias"]))
            g = expit(_linear(h, model["state"]["out.weight"], model["state"]["out.bias"]).reshape(-1))
            out = 1.0 - g
        elif family in {"FLAT-CLINICAL", "FACTORED-CLINICAL"}:
            p = _clinical_probs(model, z)
            out = np.asarray([1.0 - p[i, CLASS_TO_INDEX[int(c)]] for i, c in enumerate(yh)], dtype=np.float64)
        elif family == "SELECTIVE-GATE":
            h = _silu(_linear(z, model["state"]["fc1.weight"], model["state"]["fc1.bias"]))
            gate = expit(_linear(h, model["state"]["gate.weight"], model["state"]["gate.bias"]).reshape(-1))
            out = 1.0 - gate
        else:
            raise ValueError(f"unknown model family {family!r}")
    except Exception:
        out = _fallback_scores(model, yh)
    return np.clip(np.asarray(out, dtype=np.float64), 0.0, 1.0)


def ci_from_med(med: Any) -> np.ndarray:
    """Independent direct MED4-to-CI readout used in replay audits."""

    a = np.asarray(med, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 4:
        raise ValueError("MED4 must have shape (n,4)")
    return a @ np.asarray([-0.6516, -0.3772, 0.1224, 0.0759], dtype=np.float64) - 7.7922


def labels_from_ci(ci: Any) -> np.ndarray:
    a = np.asarray(ci, dtype=np.float64)
    return np.where(~np.isfinite(a), 9, np.where(a < -3.0, -1, np.where(a > 3.0, 1, 0))).astype(np.int64)


def replay_max_abs_diff(model: Mapping[str, Any], F: Any, yhat: Any,
                        stored_scores: Any) -> float:
    """Return the replay discrepancy for a stored score vector."""

    actual = predict_independent(model, F, yhat)
    expected = np.asarray(stored_scores, dtype=np.float64).reshape(-1)
    if len(actual) != len(expected):
        raise ValueError("stored score vector has the wrong length")
    return float(np.max(np.abs(actual - expected))) if len(actual) else 0.0


__all__ = ["predict_independent", "replay_max_abs_diff", "ci_from_med", "labels_from_ci"]

