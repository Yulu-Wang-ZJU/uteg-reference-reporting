"""Small exact-contract helpers extracted from the current implementation.

These functions are intentionally narrow.  They preserve the computational
definitions used by the current Ridge-10 reporting chain and make the
definitions testable without importing the historical runtime or touching
private data.

Provenance (source files are copied byte-for-byte under ``snapshots/``):

* ``snapshots/ridge10/run_ridge10_fullchain.py``: ``ma_class`` (lines
  414--416), ``ma_features`` (419--425), MA-ERROR-LR preprocessing/fitting/
  scoring (428--498).
* ``snapshots/adaptation/run.py``: physical-range projection, Ridge-10
  forward map, and affine/joint calibration (17--29).
* ``snapshots/sequence/evaluate.py``: frozen action rule and report replay
  (18--23, 89--98); its metric definition is reproduced below.
* ``snapshots/supplement/PROTOCOL.md`` and the current supplementary source
  define the auxiliary J distinction.  The coefficients are retained here
  only as a score helper; J is not an MA label or reporting endpoint.

No function in this module discovers or loads record identifiers, references,
trained objects, or patient data.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


MA_CLASSES = (-1, 0, 1)

# Supplementary Information, current source, Eq. in the adaptation-stages
# paragraph.  The first four values act on (R, K, MA, alpha_R); the last is
# the intercept.  This is the historical auxiliary coagulation-index score J,
# not a relabelled MA score.
AUXILIARY_J_COEFFICIENTS = np.asarray(
    [-0.6516, -0.3772, 0.1224, 0.0759, -7.7922], dtype=float
)


def ma_class(value: Any) -> np.ndarray:
    """Map MA to the fixed ``<50 / 50--70 / >70`` classes.

    Non-finite values receive the historical sentinel ``9``.  The function is
    vectorized and returns ``int8`` as in the source runner.
    """

    x = np.asarray(value, dtype=float)
    return np.where(
        ~np.isfinite(x),
        9,
        np.where(x < 50, -1, np.where(x <= 70, 0, 1)),
    ).astype(np.int8)


def ma_features(x: Any, med: Any, ym: Any, ys: Any) -> np.ndarray:
    """Build the 13-feature MA-ERROR-LR input used by the source runner."""

    x = np.asarray(x, dtype=float)
    med = np.asarray(med, dtype=float)
    pred_ma = med[:, 2]
    predicted_class = ma_class(pred_ma)
    return np.column_stack(
        (
            x,
            (med - np.asarray(ym)[None, :]) / np.asarray(ys)[None, :],
            (pred_ma - 60.0) / 10.0,
            np.minimum(np.abs(pred_ma - 50.0), np.abs(pred_ma - 70.0)) / 10.0,
            (predicted_class == -1).astype(float),
            (predicted_class == 0).astype(float),
            (predicted_class == 1).astype(float),
        )
    )


def _preprocess_gate(features: np.ndarray) -> dict[str, Any]:
    """Exact per-column median/mean/SD preprocessing contract."""

    f = np.asarray(features, dtype=float)
    finite = np.isfinite(f)
    median = np.asarray(
        [np.median(f[finite[:, j], j]) if finite[:, j].any() else 0.0 for j in range(f.shape[1])]
    )
    mean = np.asarray(
        [np.mean(f[finite[:, j], j]) if finite[:, j].any() else 0.0 for j in range(f.shape[1])]
    )
    sd = np.asarray(
        [max(np.std(f[finite[:, j], j]), 1e-6) if finite[:, j].any() else 1.0 for j in range(f.shape[1])]
    )
    raw = (np.where(finite, f, median[None, :]) - mean[None, :]) / sd[None, :]
    return {
        "mean": mean,
        "sd": sd,
        "median": median,
        "ddof": 0,
        "floor": 1e-6,
        "clip": [-10.0, 10.0],
        "n": int(len(f)),
        "p": int(f.shape[1]),
        "missing_fraction": float(np.mean(~finite)) if f.size else 0.0,
        "clip_fraction": float(np.mean(np.abs(raw) > 10.0)) if f.size else 0.0,
    }


def fit_ma_error_lr(
    features: Any,
    predicted_class: Any,
    truth_class: Any,
    *,
    fit_id: str = "synthetic",
    seed: int = 0,
) -> dict[str, Any]:
    """Fit the historical MA-ERROR-LR risk score on development rows.

    This is the source runner's objective and fallback contract, exposed for
    synthetic checks and private-data re-use.  Callers remain responsible for
    fitting only on the legal development partition.
    """

    f = np.asarray(features, dtype=float)
    pred = np.asarray(predicted_class)
    truth = np.asarray(truth_class)
    known = np.isin(pred, MA_CLASSES) & np.isin(truth, MA_CLASSES)
    legal = f[known]
    y = (pred[known] != truth[known]).astype(float)
    n = len(y)
    errors = int(y.sum())
    correct = n - errors
    constant = float((errors + 0.5) / (n + 1.0))
    pre = _preprocess_gate(legal) if n else {
        "mean": np.zeros(f.shape[1]),
        "sd": np.ones(f.shape[1]),
        "median": np.zeros(f.shape[1]),
        "ddof": 0,
        "floor": 1e-6,
        "clip": [-10.0, 10.0],
        "n": 0,
        "p": int(f.shape[1]),
        "missing_fraction": 0.0,
        "clip_fraction": 0.0,
    }
    out: dict[str, Any] = {
        "api_version": "ma-error-lr-v1",
        "id": "MA-ERROR-LR",
        "family": "MA-ERROR-LR",
        "fit_id": str(fit_id),
        "fit_seed": int(seed),
        "feature_dim": int(f.shape[1]),
        "regularizer": 0.1,
        "n": n,
        "errors": errors,
        "correct": correct,
        "error_rate": float(errors / n) if n else None,
        "legal_count": n,
        "finite_input_rows": int(np.isfinite(f).all(axis=1).sum()),
        "pre": pre,
        "constant_score": constant,
        "fallback": False,
        "fallback_reason": None,
        "success": False,
        "finite": True,
        "status": "not_started",
        "iterations": 0,
        "optimizer_steps": 0,
        "coef": np.zeros(f.shape[1]),
        "intercept": float(np.log(constant / (1.0 - constant))) if 0 < constant < 1 else 0.0,
        "train_pred_class_counts": {str(k): int(np.sum(pred[known] == k)) for k in MA_CLASSES},
        "train_true_class_counts": {str(k): int(np.sum(truth[known] == k)) for k in MA_CLASSES},
    }
    if n < 8 or errors < 2 or correct < 2:
        out.update(fallback=True, fallback_reason="small_or_one_sided", success=True, status="fallback_constant")
        return out

    z = np.clip(
        (np.where(np.isfinite(legal), legal, np.asarray(pre["median"])[None, :])
         - np.asarray(pre["mean"])[None, :])
        / np.asarray(pre["sd"])[None, :],
        -10.0,
        10.0,
    )
    p = z.shape[1]

    def bce(theta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = z @ theta[:p] + theta[p]
        prob = expit(eta)
        loss = float(np.mean(np.logaddexp(0.0, eta) - y * eta) + 0.1 * np.sum(theta[:p] ** 2))
        residual = prob - y
        return loss, np.r_[z.T @ residual / n + 0.2 * theta[:p], np.mean(residual)]

    try:
        sol = minimize(
            bce,
            np.zeros(p + 1),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 200, "ftol": 1e-9, "gtol": 1e-9},
        )
        theta = np.asarray(sol.x, dtype=float)
        finite = bool(np.isfinite(theta).all() and np.isfinite(sol.fun))
        if not finite:
            raise FloatingPointError("nonfinite MA gate optimizer result")
        out.update(
            success=bool(sol.success),
            finite=True,
            status=str(sol.message),
            iterations=int(getattr(sol, "nit", 0)),
            optimizer_steps=int(getattr(sol, "nit", 0)),
            objective=float(sol.fun),
            coef=theta[:-1],
            intercept=float(theta[-1]),
        )
    except Exception as exc:
        out.update(
            fallback=True,
            fallback_reason="optimization_exception:" + type(exc).__name__,
            success=True,
            finite=False,
            status="fallback_exception",
        )
    return out


def score_ma_error_lr(model: Mapping[str, Any], features: Any, predicted_class: Any) -> np.ndarray:
    """Score rows and fail closed for non-finite/unknown classes."""

    f = np.asarray(features, dtype=float)
    pre = model["pre"]
    z = np.clip(
        (np.where(np.isfinite(f), f, np.asarray(pre["median"])[None, :])
         - np.asarray(pre["mean"])[None, :])
        / np.asarray(pre["sd"])[None, :],
        -10.0,
        10.0,
    )
    if model.get("fallback"):
        score = np.full(len(f), float(model["constant_score"]))
    else:
        score = expit(z @ np.asarray(model["coef"]) + float(model["intercept"]))
    valid = np.isin(np.asarray(predicted_class), MA_CLASSES) & np.isfinite(f).all(axis=1)
    return np.where(valid, score, np.inf)


def project_readouts(value: Any) -> np.ndarray:
    """Apply the physical-range projection used by adaptation comparisons."""

    y = np.array(value, dtype=float, copy=True)
    y[..., :3] = np.maximum(y[..., :3], 0)
    y[..., 3] = np.clip(y[..., 3], 0, 89.999999)
    return y


def raw_ridge10(model: Mapping[str, Any], x: Any) -> np.ndarray:
    """Apply a frozen four-output Ridge-10 source object."""

    x = np.asarray(x, dtype=float)
    return project_readouts(
        np.asarray(model["ym"]) + np.asarray(model["ys"]) * (
            ((x - np.asarray(model["xm"])) / np.asarray(model["xs"])) @ np.asarray(model["coef"])
            + np.asarray(model["intercept"])
        )
    )


def fit_calibration(predicted: Any, reference: Any, source_model: Mapping[str, Any], arm: str) -> dict[str, Any]:
    """Fit the two current four-output adaptation comparison arms.

    ``arm='affine'`` is the independent per-output least-squares fit;
    ``arm='joint'`` is the identity-centred regularized joint linear fit with
    penalty 10.  Both formulas are copied from the four-way comparison
    runner.  ``reference`` must be the legal target adaptation pool.
    """

    p = np.asarray(predicted, dtype=float)
    y = np.asarray(reference, dtype=float)
    scale = np.maximum(np.asarray(source_model["ys"], dtype=float), 1e-6)
    v = (project_readouts(p) - np.asarray(source_model["ym"], dtype=float)) / scale
    u = (y - np.asarray(source_model["ym"], dtype=float)) / scale
    vm = v.mean(0)
    um = u.mean(0)
    vc = v - vm
    uc = u - um
    if arm == "affine":
        slopes = np.asarray(
            [np.linalg.lstsq(vc[:, j, None], uc[:, j], rcond=None)[0][0] for j in range(4)]
        )
        matrix = np.diag(slopes)
    elif arm == "joint":
        matrix = np.linalg.solve(vc.T @ vc + 10 * np.eye(4), vc.T @ uc + 10 * np.eye(4))
    else:
        raise ValueError("arm must be 'affine' or 'joint'")
    return {"matrix": matrix, "vbar": vm, "ubar": um, "mu": np.asarray(source_model["ym"]), "scale": scale}


def apply_calibration(calibration: Mapping[str, Any], predicted: Any) -> np.ndarray:
    """Apply a frozen calibration object and the same physical projection."""

    p = project_readouts(predicted)
    scale = np.asarray(calibration["scale"])
    return project_readouts(
        np.asarray(calibration["mu"])
        + scale
        * (
            np.asarray(calibration["ubar"])
            + ((p - np.asarray(calibration["mu"])) / scale - np.asarray(calibration["vbar"]))
            @ np.asarray(calibration["matrix"])
        )
    )


def auxiliary_j(readouts: Any) -> np.ndarray:
    """Return the historical auxiliary adaptation score J, not MA labels."""

    y = np.asarray(readouts, dtype=float)
    if y.shape[-1] != 4:
        raise ValueError("readouts must end with four values: R, K, MA, alpha_R")
    return np.tensordot(y, AUXILIARY_J_COEFFICIENTS[:4], axes=([-1], [0])) + AUXILIARY_J_COEFFICIENTS[4]


def class_actions(
    predicted_class: Any,
    base_score: Any,
    dual_score: Any,
    valid: Any,
    thresholds: Mapping[str, Any],
) -> np.ndarray:
    """Apply the frozen dual-head/MA-ERROR-LR release rule."""

    pc = np.asarray(predicted_class)
    bs = np.asarray(base_score, dtype=float)
    ds = np.asarray(dual_score, dtype=float)
    ok = np.asarray(valid, dtype=bool)
    action = np.zeros(len(pc), dtype=bool)
    if thresholds.get("all_withhold"):
        return action
    normal = thresholds.get("normal")
    tail = thresholds.get("tail")
    if normal is not None:
        action |= ok & (pc == 0) & np.isfinite(ds) & (ds <= normal)
    if tail is not None:
        action |= ok & (pc != 0) & np.isfinite(bs) & (bs <= tail)
    return action


def cumulative_first_report(
    checkpoints: Iterable[tuple[int, Any, Any]],
) -> dict[str, np.ndarray]:
    """Replay first-report retention across 5/10/15-minute checkpoints."""

    items = list(checkpoints)
    if not items:
        raise ValueError("at least one checkpoint is required")
    first_action = np.asarray(items[0][1], dtype=bool)
    first_pred = np.asarray(items[0][2])
    if first_action.shape != first_pred.shape or first_action.ndim != 2:
        raise ValueError("checkpoint actions/predictions must have shape (seed, record)")
    ever = np.zeros_like(first_action, dtype=bool)
    first_class = np.full_like(first_pred, 9)
    first_time = np.full(first_pred.shape, -1, dtype=int)
    for horizon, action, predicted in items:
        a = np.asarray(action, dtype=bool)
        p = np.asarray(predicted)
        if a.shape != ever.shape or p.shape != ever.shape:
            raise ValueError("all checkpoints must share the first checkpoint shape")
        new = a & ~ever
        first_class[new] = p[new]
        first_time[new] = int(horizon)
        ever |= a
    return {"first_class": first_class, "first_time": first_time, "reported": ever}


def metrics(action: Any, predicted: Any, truth: Any) -> dict[str, Any]:
    """Compute the frozen sequence endpoint counts used in the source code."""

    a = np.asarray(action, dtype=bool)
    p = np.asarray(predicted)
    y = np.asarray(truth)
    if a.shape != p.shape or p.ndim != 2 or y.shape != (p.shape[1],):
        raise ValueError("action/predicted must be (seed, record); truth must be (record,)")
    good = a & (p == y[None, :])
    bad = a & ~good
    return {
        "total": int(a.size),
        "released": int(a.sum()),
        "correct": int(good.sum()),
        "wrong": int(bad.sum()),
        "unreported": int((~a).sum()),
        "agreement": float(good.sum() / a.sum()) if a.any() else None,
        "coverage": float(a.mean()),
        "correct_yield": float(good.mean()),
        "wrong_yield": float(bad.mean()),
        "low_to_within": int((a & (y[None, :] == -1) & (p == 0)).sum()),
        "high_to_within": int((a & (y[None, :] == 1) & (p == 0)).sum()),
        "normal_to_tail": int((a & (y[None, :] == 0) & (p != 0)).sum()),
        "low_high_cross": int((a & (y[None, :] != 0) & (p == -y[None, :])).sum()),
    }
