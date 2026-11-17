"""Small, data-agnostic mapping models for the MA imbalance study.

This module is a *preparation* component.  It does not discover files, load
patient-level data, choose a split, run a report gate, or write results.  A
caller must provide one legal training partition at a time.  The intended
protocol is described in ``outputs/uteg_mapping_imbalance_design_20261101``.

The implementation keeps the four-readout contract used by the manuscript:
``(R, K, MA, alpha_R)``.  Input and output standardisation is fitted only on
the rows passed to ``fit_mapping`` or ``fit_adapter``.  MA is the third
column, and its fixed states are ``<50`` (low), ``50--70`` (within range), and
``>70`` (high).

Supported mapping/adaptation method names are:

``unweighted``
    Ordinary Ridge/Huber objective (the scientific control).
``class_weighted``
    MA loss weighted by inverse class counts raised to ``gamma``.
``denseweight``
    MA loss weighted from a training-only Gaussian KDE, following the
    density-based continuous-regression idea of Steininger et al. (2021).
``lds``
    MA loss weighted from a Gaussian-smoothed 1-mm label histogram, following
    the label-distribution-smoothing idea of Yang et al. (2021).
``balanced_mse``
    Full-batch Balanced-MSE/BMC-style objective for the scalar MA head.  It is
    deliberately implemented as a numerical optimisation control because it
    is not a sample-weight argument that can be passed to closed-form Ridge.
``mse_optimizer_control``
    Plain-MSE MA objective optimised with the same numerical initialisation
    and budget as ``balanced_mse``.  It separates the loss from the optimiser.

The literature methods are only adapted here as low-capacity regression
losses; this module does not claim to reproduce their original deep-network
or feature-distribution pipelines.  ``diagnostics`` records the support and
weighting actually used so that absent classes cannot be mistaken for
successful reweighting.  Optimisation failures raise an exception; there is
no silent fallback to a different model.

References
----------
Steininger et al., *Density-based weighting for imbalanced regression*,
Machine Learning 110, 2187--2211 (2021),
https://doi.org/10.1007/s10994-021-06023-5

Yang et al., *Delving into Deep Imbalanced Regression*, ICML 2021,
https://proceedings.mlr.press/v139/yang21m.html

Ren et al., *Balanced MSE for Imbalanced Visual Regression*, CVPR 2022,
https://openaccess.thecvf.com/content/CVPR2022/html/Ren_Balanced_MSE_for_Imbalanced_Visual_Regression_CVPR_2022_paper.html
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from scipy.ndimage import convolve1d, gaussian_filter1d
from scipy.optimize import minimize
from scipy.special import logsumexp


MA_INDEX = 2
READOUT_DIM = 4
MA_BOUNDARIES = (50.0, 70.0)
_EPS = 1e-12


def _as_2d(name: str, value: Any, columns: int | None = None) -> np.ndarray:
    """Validate a finite two-dimensional floating array without imputing."""

    array = np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if columns is not None and array.shape[1] != columns:
        raise ValueError(f"{name} must have {columns} columns, got {array.shape[1]}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _validate_pair(X: Any, Y: Any) -> tuple[np.ndarray, np.ndarray]:
    x = _as_2d("X", X)
    y = _as_2d("Y", Y, READOUT_DIM)
    if len(x) != len(y):
        raise ValueError(f"X and Y must have the same number of rows ({len(x)} != {len(y)})")
    if x.shape[1] == 0:
        raise ValueError("X must have at least one feature")
    return x, y


def _standardise(value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(value, axis=0)
    # Keep the source implementation's scale floor.  A constant feature or
    # readout remains finite without silently changing the data split.
    scale = np.maximum(np.std(value, axis=0), 1e-6)
    return (value - mean) / scale, mean, scale


def _parameter_value(parameter: Any, key: str, default: float) -> float:
    if parameter is None:
        value = default
    elif isinstance(parameter, Mapping):
        value = parameter.get(key, default)
    else:
        value = parameter
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"parameter {key!r} must be numeric") from exc
    if not np.isfinite(value):
        raise ValueError(f"parameter {key!r} must be finite")
    return value


def _normalise_and_cap_weights(raw: np.ndarray, ratio_cap: float) -> tuple[np.ndarray, dict[str, float]]:
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 1 or not len(raw) or not np.isfinite(raw).all() or np.any(raw <= 0):
        raise ValueError("raw sample weights must be finite and strictly positive")
    if not np.isfinite(ratio_cap) or ratio_cap < 1:
        raise ValueError("weight_ratio_cap must be finite and at least one")
    maximum = float(np.max(raw))
    minimum_allowed = maximum / float(ratio_cap)
    clipped = np.maximum(raw, minimum_allowed)
    weights = clipped / float(np.mean(clipped))
    ratio = float(np.max(weights) / np.min(weights))
    if ratio > ratio_cap * (1.0 + 1e-10):
        raise RuntimeError(f"weight cap failed: observed ratio {ratio} > {ratio_cap}")
    diagnostics = {
        "weight_mean": float(np.mean(weights)),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "weight_ratio": ratio,
        "effective_sample_size": float(np.sum(weights) ** 2 / np.sum(weights**2)),
    }
    return weights, diagnostics


def _class_labels(ma: np.ndarray) -> np.ndarray:
    ma = np.asarray(ma, dtype=float)
    return np.where(ma < MA_BOUNDARIES[0], -1, np.where(ma <= MA_BOUNDARIES[1], 0, 1)).astype(np.int8)


def _base_weight_diagnostics(ma: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    labels = _class_labels(ma)
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or len(weights) != len(ma) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("diagnostic weights must be finite, positive and aligned with MA")
    weight_sum = float(np.sum(weights))
    out: dict[str, Any] = {
        "n": int(len(ma)),
        "class_counts": {"low": int(np.sum(labels == -1)), "within": int(np.sum(labels == 0)), "high": int(np.sum(labels == 1))},
        "missing_classes": [name for name, code in (("low", -1), ("within", 0), ("high", 1)) if not np.any(labels == code)],
        "weight_mean": float(np.mean(weights)),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "weight_ratio": float(np.max(weights) / np.min(weights)),
        "effective_sample_size": float(weight_sum**2 / np.sum(weights**2)),
    }
    return out


def _class_weights(ma: np.ndarray, gamma: float) -> np.ndarray:
    if gamma < 0:
        raise ValueError("class-weighted gamma must be non-negative")
    labels = _class_labels(ma)
    raw = np.ones(len(ma), dtype=float)
    for code in (-1, 0, 1):
        mask = labels == code
        if np.any(mask):
            raw[mask] = float(np.sum(mask)) ** (-gamma)
    return raw


def _silverman_bandwidth(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 1.0
    # Match the authors' normal-reference bandwidth (population std), rather
    # than silently substituting the robust 0.9 * min(std, IQR/1.34) variant.
    scale = float(np.std(values))
    if scale <= _EPS:
        return 1.0  # Constant training labels receive uniform weights.
    return max(1.06 * scale * len(values) ** (-0.2), 1e-6)


def _denseweight(ma: np.ndarray, alpha: float) -> np.ndarray:
    if alpha < 0:
        raise ValueError("denseweight alpha must be non-negative")
    bandwidth = _silverman_bandwidth(ma)
    distance = (ma[:, None] - ma[None, :]) / bandwidth
    density = np.mean(np.exp(-0.5 * distance**2), axis=1) / (np.sqrt(2.0 * np.pi) * bandwidth)
    if float(np.max(density) - np.min(density)) <= _EPS:
        normalized = np.ones_like(density)
    else:
        normalized = (density - np.min(density)) / (np.max(density) - np.min(density))
    # Paper Eq. (2): min/max over KDE values at the training labels.  Evaluate
    # the Gaussian KDE directly, not the author's FFT/grid approximation.
    # Mean normalisation and the declared study-specific ratio cap follow.
    return np.maximum(1.0 - alpha * normalized, 1e-6)


def _lds_weight(ma: np.ndarray, sigma: float, bin_width: float = 1.0,
                kernel_size: int = 5) -> np.ndarray:
    """Author-repository sqrt_inv LDS order, with MA-specific label bins.

    Reference: YyzHarry/imbalanced-regression, imdb-wiki-dir/datasets.py and
    utils.py.  sqrt(count) -> finite-window smoothing -> reciprocal, not
    sqrt(smooth(count)).  Outside the training-label range, counts are zero.
    The author's fixed age range is replaced by training-only 1-mm MA bins.
    """
    if not np.isfinite(sigma) or not np.isfinite(bin_width) or sigma <= 0 or bin_width <= 0:
        raise ValueError("LDS sigma and bin_width must be positive and finite")
    if type(kernel_size) is not int or kernel_size < 1 or kernel_size % 2 != 1:
        raise ValueError("LDS kernel_size must be a positive odd integer")
    bins = np.floor(ma / bin_width).astype(np.int64)
    indices = bins - bins.min()
    counts = np.bincount(indices).astype(float)
    impulse = np.zeros(kernel_size, dtype=float)
    impulse[kernel_size // 2] = 1.0
    # Match the author's gaussian_filter1d window construction, including
    # its reflected boundary, then zero-pad the histogram convolution.
    kernel = gaussian_filter1d(impulse, sigma=sigma / bin_width, mode="reflect")
    kernel /= kernel.max()
    smoothed = convolve1d(np.sqrt(counts), kernel, mode="constant", cval=0.0)
    return 1.0 / np.maximum(smoothed[indices], _EPS)


def _ma_weights(ma: np.ndarray, method: str, parameter: Any, ratio_cap: float) -> tuple[np.ndarray, dict[str, Any]]:
    method = _canonical_method(method)
    if method == "unweighted" or method in {"balanced_mse", "mse_optimizer_control"}:
        raw = np.ones(len(ma), dtype=float)
        source = "uniform"
        parameter_used: Any = None
    elif method == "class_weighted":
        gamma = _parameter_value(parameter, "gamma", 0.5)
        raw = _class_weights(ma, gamma)
        source = "inverse_class_count_power"
        parameter_used = {"gamma": gamma}
    elif method == "denseweight":
        alpha = _parameter_value(parameter, "alpha", 0.5)
        raw = _denseweight(ma, alpha)
        source = "gaussian_kde_denseweight"
        parameter_used = {"alpha": alpha, "bandwidth": _silverman_bandwidth(ma),
                          "bandwidth_rule": "1.06_std_n^-0.2", "density_normalisation": "training_label_minmax"}
    elif method == "lds":
        # The scalar shorthand denotes sigma only.  The contract fixes the
        # LDS grid to 1 mm; a non-unit bin width is accepted only when the
        # caller uses the explicit mapping form.
        sigma = _parameter_value(parameter, "sigma", 2.0)
        bin_width = (
            _parameter_value(parameter, "bin_width", 1.0)
            if isinstance(parameter, Mapping)
            else 1.0
        )
        kernel_size = parameter.get("kernel_size", 5) if isinstance(parameter, Mapping) else 5
        raw = _lds_weight(ma, sigma, bin_width, kernel_size)
        source = "gaussian_label_distribution_smoothing"
        parameter_used = {"sigma": sigma, "bin_width": bin_width,
                          "kernel_size": kernel_size, "reweight": "sqrt_inv_before_smoothing",
                          "bin_assignment": "floor", "histogram_padding": "constant_zero"}
    else:
        raise ValueError(f"unsupported MA weighting method: {method}")
    weights, diagnostics = _normalise_and_cap_weights(raw, ratio_cap)
    diagnostics.update(_base_weight_diagnostics(ma, weights), method=method, weight_source=source, parameter=parameter_used, weight_ratio_cap=float(ratio_cap))
    return weights, diagnostics


def _canonical_method(method: str) -> str:
    if not isinstance(method, str):
        raise TypeError("method must be a string")
    key = method.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "uniform": "unweighted",
        "ordinary": "unweighted",
        "class": "class_weighted",
        "classweight": "class_weighted",
        "dense": "denseweight",
        "label_distribution_smoothing": "lds",
        "balanced": "balanced_mse",
        "bmc": "balanced_mse",
        "mse_control": "mse_optimizer_control",
    }
    return aliases.get(key, key)


def _ridge_fit(Z: np.ndarray, T: np.ndarray, alpha: float, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if alpha < 0 or not np.isfinite(alpha):
        raise ValueError("ridge_alpha must be finite and non-negative")
    n, p = Z.shape
    W = np.ones(n, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if W.shape != (n,) or not np.isfinite(W).all() or np.any(W <= 0):
        raise ValueError("weights must be positive and match Z rows")
    zbar = np.average(Z, axis=0, weights=W)
    tbar = np.average(T, axis=0, weights=W)
    zc = Z - zbar
    tc = T - tbar
    lhs = zc.T @ (W[:, None] * zc) + alpha * np.eye(p)
    rhs = zc.T @ (W[:, None] * tc)
    try:
        coef = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("Ridge solve failed; no fallback was applied") from exc
    intercept = tbar - zbar @ coef
    return coef, intercept


def _huber_value_grad(residual: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray]:
    if delta <= 0 or not np.isfinite(delta):
        raise ValueError("huber_delta must be finite and positive")
    absolute = np.abs(residual)
    inside = absolute <= delta
    value = np.where(inside, residual**2, 2.0 * delta * absolute - delta**2)
    gradient = np.where(inside, 2.0 * residual, 2.0 * delta * np.sign(residual))
    return value, gradient


def _bmc_value_grad(pred: np.ndarray, target: np.ndarray, sigma: float) -> tuple[float, np.ndarray]:
    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError("balanced_mse sigma must be finite and positive")
    residual = pred[:, None] - target[None, :]
    logits = -(residual**2) / (2.0 * sigma**2)
    diagonal = np.arange(len(target))
    log_normalizer = logsumexp(logits, axis=1)
    # Compute cross-entropy from log-sum-exp rather than clipping a possibly
    # underflowed probability.  This keeps the value and analytic gradient
    # consistent for separated MA values.
    cross_entropy = log_normalizer - logits[diagonal, diagonal]
    probabilities = np.exp(logits - log_normalizer[:, None])
    value = float(2.0 * sigma**2 * np.sum(cross_entropy))
    expected_residual = np.sum(probabilities * residual, axis=1)
    gradient = 2.0 * (residual[diagonal, diagonal] - expected_residual)
    return value, gradient


def _flatten_params(coef: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    return np.r_[np.asarray(coef, dtype=float).ravel(), np.asarray(intercept, dtype=float)]


def _unflatten_params(theta: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    if theta.size != p * READOUT_DIM + READOUT_DIM:
        raise ValueError("unexpected parameter vector size")
    return theta[: p * READOUT_DIM].reshape(p, READOUT_DIM), theta[p * READOUT_DIM :]


def _optimised_mapping(
    Z: np.ndarray,
    T: np.ndarray,
    alpha: float,
    method: str,
    parameter: Any,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    initial_coef, initial_intercept = _ridge_fit(Z, T, alpha)
    initial = _flatten_params(initial_coef, initial_intercept)
    sigma = _parameter_value(parameter, "sigma", 0.5)
    method = _canonical_method(method)
    if method not in {"balanced_mse", "mse_optimizer_control"}:
        raise ValueError(f"optimised mapping is not defined for {method}")
    n, p = Z.shape

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        coef, intercept = _unflatten_params(theta, p)
        prediction = Z @ coef + intercept
        residual = prediction - T
        value = 0.0
        grad_pred = np.zeros_like(prediction)
        if method == "balanced_mse":
            bmc_value, bmc_grad = _bmc_value_grad(prediction[:, MA_INDEX], T[:, MA_INDEX], sigma)
            value += bmc_value
            grad_pred[:, MA_INDEX] = bmc_grad
            other = [j for j in range(READOUT_DIM) if j != MA_INDEX]
            value += float(np.sum(residual[:, other] ** 2))
            grad_pred[:, other] = 2.0 * residual[:, other]
        else:
            value += float(np.sum(residual**2))
            grad_pred = 2.0 * residual
        value += float(alpha * np.sum(coef**2))
        grad_coef = Z.T @ grad_pred + 2.0 * alpha * coef
        grad_intercept = np.sum(grad_pred, axis=0)
        gradient = _flatten_params(grad_coef, grad_intercept)
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise FloatingPointError("non-finite optimised mapping objective")
        return value, gradient

    solution = minimize(
        objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": int(max_iter), "ftol": 1e-12, "gtol": 1e-9, "maxls": 50},
    )
    if not bool(solution.success) or not np.isfinite(solution.x).all():
        raise RuntimeError(f"{method} mapping optimisation failed: {solution.message}")
    coef, intercept = _unflatten_params(np.asarray(solution.x), p)
    return coef, intercept, {"optimizer": "L-BFGS-B", "success": True, "status": str(solution.message), "iterations": int(getattr(solution, "nit", 0)), "objective": float(solution.fun), "sigma": sigma if method == "balanced_mse" else None, "initialisation": "ordinary_ridge"}


@dataclass(frozen=True)
class FittedMapping:
    """Frozen four-output source/target mapping fitted on one legal split."""

    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray
    method: str
    parameter: Any
    ridge_alpha: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def predict_standardized(self, X: Any) -> np.ndarray:
        x = _as_2d("X", X)
        if x.shape[1] != len(self.x_mean):
            raise ValueError(f"X must have {len(self.x_mean)} features")
        z = (x - self.x_mean) / self.x_scale
        return z @ self.coef + self.intercept

    def predict(self, X: Any) -> np.ndarray:
        """Return unprojected physical-unit readouts ``(R,K,MA,alpha_R)``."""

        return self.y_mean + self.y_scale * self.predict_standardized(X)


def fit_mapping(
    X: Any,
    Y: Any,
    *,
    method: str,
    parameter: Any = None,
    ridge_alpha: float = 10.0,
    weight_ratio_cap: float = 5.0,
    max_iter: int = 2000,
) -> FittedMapping:
    """Fit one source/target mapping candidate without reading external data.

    ``method`` selects only the MA objective.  For weighted closed-form Ridge
    candidates, the other three readouts retain the ordinary unweighted
    objective.  Balanced-MSE and its MSE optimiser control use the same Ridge
    initialisation and L-BFGS-B budget; an optimiser failure raises directly.
    """

    x, y = _validate_pair(X, Y)
    if int(max_iter) <= 0:
        raise ValueError("max_iter must be positive")
    method = _canonical_method(method)
    allowed = {"unweighted", "class_weighted", "denseweight", "lds", "balanced_mse", "mse_optimizer_control"}
    if method not in allowed:
        raise ValueError(f"unsupported mapping method: {method}")
    Z, x_mean, x_scale = _standardise(x)
    T, y_mean, y_scale = _standardise(y)
    ma_weights, weight_diag = _ma_weights(y[:, MA_INDEX], method, parameter, weight_ratio_cap)
    if method in {"balanced_mse", "mse_optimizer_control"}:
        coef, intercept, opt_diag = _optimised_mapping(Z, T, float(ridge_alpha), method, parameter, int(max_iter))
    else:
        coef = np.empty((x.shape[1], READOUT_DIM), dtype=float)
        intercept = np.empty(READOUT_DIM, dtype=float)
        for output in range(READOUT_DIM):
            weights = ma_weights if output == MA_INDEX else np.ones(len(y), dtype=float)
            output_coef, output_intercept = _ridge_fit(Z, T[:, [output]], float(ridge_alpha), weights)
            coef[:, output] = output_coef[:, 0]
            intercept[output] = float(output_intercept[0])
        opt_diag = {"optimizer": "closed_form_ridge", "success": True, "status": "solved", "iterations": 0, "objective": None, "initialisation": "not_applicable"}
    diagnostics = {"api_version": "ma-imbalance-mapping-v1", "stage": "source_mapping", "method": method, "parameter": parameter, "ridge_alpha": float(ridge_alpha), "n_features": int(x.shape[1]), "n_records": int(len(x)), "ma_index": MA_INDEX, "ma_boundaries": list(MA_BOUNDARIES), "ma_weighting": weight_diag, "optimizer": opt_diag,
                   "loss_reduction": "sum", "sigma_ma_physical": _parameter_value(parameter, "sigma", 0.5) * float(y_scale[MA_INDEX]) if method == "balanced_mse" else None}
    return FittedMapping(x_mean, x_scale, y_mean, y_scale, coef, intercept, method, parameter, float(ridge_alpha), diagnostics)


def _adaptation_weights(reference_ma: np.ndarray, method: str, parameter: Any, cap: float) -> tuple[np.ndarray, dict[str, Any]]:
    return _ma_weights(reference_ma, method, parameter, cap)


def _adapter_objective(
    theta: np.ndarray,
    Vc: np.ndarray,
    U: np.ndarray,
    reference_ma: np.ndarray,
    method: str,
    parameter: Any,
    weights: np.ndarray,
    penalty: float,
    huber_delta: float,
) -> tuple[float, np.ndarray]:
    n = len(U)
    M = theta[: READOUT_DIM * READOUT_DIM].reshape(READOUT_DIM, READOUT_DIM)
    b = theta[READOUT_DIM * READOUT_DIM :]
    prediction = Vc @ M + b
    residual = prediction - U
    value = float(penalty * np.sum((M - np.eye(READOUT_DIM)) ** 2))
    grad_prediction = np.zeros_like(residual)
    method = _canonical_method(method)
    if method == "balanced_mse":
        sigma = _parameter_value(parameter, "sigma", 0.5)
        bmc_value, bmc_grad = _bmc_value_grad(prediction[:, MA_INDEX], U[:, MA_INDEX], sigma)
        value += bmc_value
        grad_prediction[:, MA_INDEX] = bmc_grad
    elif method == "mse_optimizer_control":
        value += float(np.sum(residual[:, MA_INDEX] ** 2))
        grad_prediction[:, MA_INDEX] = 2.0 * residual[:, MA_INDEX]
    else:
        ma_value, ma_grad = _huber_value_grad(residual[:, MA_INDEX], huber_delta)
        value += float(np.sum(weights * ma_value))
        grad_prediction[:, MA_INDEX] = weights * ma_grad
    for output in range(READOUT_DIM):
        if output == MA_INDEX:
            continue
        other_value, other_grad = _huber_value_grad(residual[:, output], huber_delta)
        value += float(np.sum(other_value))
        grad_prediction[:, output] = other_grad
    grad_M = Vc.T @ grad_prediction + 2.0 * penalty * (M - np.eye(READOUT_DIM))
    grad_b = np.sum(grad_prediction, axis=0)
    gradient = np.r_[grad_M.ravel(), grad_b]
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise FloatingPointError("non-finite adaptation objective")
    return value, gradient


@dataclass(frozen=True)
class FittedAdapter:
    """Frozen source-coordinate joint target adaptation."""

    mapping: FittedMapping
    source_y_mean: np.ndarray
    source_y_scale: np.ndarray
    v_mean: np.ndarray
    u_mean: np.ndarray
    matrix: np.ndarray
    intercept: np.ndarray
    method: str
    parameter: Any
    penalty: float
    huber_delta: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def predict_standardized(self, X: Any) -> np.ndarray:
        source = project_readouts(self.mapping.predict(X))
        V = (source - self.source_y_mean) / self.source_y_scale
        return self.u_mean + (V - self.v_mean) @ self.matrix + self.intercept

    def predict(self, X: Any) -> np.ndarray:
        """Return adapted physical-unit readouts after range projection."""

        return project_readouts(self.source_y_mean + self.source_y_scale * self.predict_standardized(X))


def fit_adapter(
    mapping: FittedMapping,
    X: Any,
    Y: Any,
    *,
    method: str,
    parameter: Any = None,
    weight_ratio_cap: float = 5.0,
    max_iter: int = 2000,
    penalty: float = 10.0,
    huber_delta: float = 1.345,
) -> FittedAdapter:
    """Fit target adaptation in source-standardised coordinates.

    The joint transformation is ``u_mean + (v-v_mean) @ M + intercept`` and
    is penalised by ``penalty * ||M-I||_F^2``.  Only the MA term receives
    imbalance weights; the other three outputs retain the ordinary Huber
    objective.  ``balanced_mse`` replaces the MA Huber term with full-batch
    BMC, while ``mse_optimizer_control`` uses plain MA squared error with the
    same initial identity transform and L-BFGS-B budget.
    """

    if not isinstance(mapping, FittedMapping):
        raise TypeError("mapping must be a FittedMapping returned by fit_mapping")
    x, y = _validate_pair(X, Y)
    if int(max_iter) <= 0 or penalty < 0 or not np.isfinite(penalty):
        raise ValueError("max_iter must be positive and penalty must be finite/non-negative")
    if not np.isfinite(huber_delta) or huber_delta <= 0:
        raise ValueError("huber_delta must be finite and positive")
    method = _canonical_method(method)
    allowed = {"unweighted", "class_weighted", "denseweight", "lds", "balanced_mse", "mse_optimizer_control"}
    if method not in allowed:
        raise ValueError(f"unsupported adaptation method: {method}")
    source = project_readouts(mapping.predict(x))
    V = (source - mapping.y_mean) / mapping.y_scale
    U = (y - mapping.y_mean) / mapping.y_scale
    v_mean = np.mean(V, axis=0)
    u_mean = np.mean(U, axis=0)
    Vc = V - v_mean
    ma_weights, weight_diag = _adaptation_weights(y[:, MA_INDEX], method, parameter, weight_ratio_cap)
    initial_matrix = np.eye(READOUT_DIM, dtype=float)
    initial_intercept = np.zeros(READOUT_DIM, dtype=float)
    initial = np.r_[initial_matrix.ravel(), initial_intercept]
    solution = minimize(
        _adapter_objective,
        initial,
        args=(Vc, U - u_mean, y[:, MA_INDEX], method, parameter, ma_weights, float(penalty), float(huber_delta)),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": int(max_iter), "ftol": 1e-12, "gtol": 1e-9, "maxls": 50},
    )
    if not bool(solution.success) or not np.isfinite(solution.x).all():
        raise RuntimeError(f"{method} adaptation optimisation failed: {solution.message}")
    matrix = np.asarray(solution.x[: READOUT_DIM * READOUT_DIM], dtype=float).reshape(READOUT_DIM, READOUT_DIM)
    intercept = np.asarray(solution.x[READOUT_DIM * READOUT_DIM :], dtype=float)
    diagnostics = {"api_version": "ma-imbalance-mapping-v1", "stage": "target_adaptation", "method": method, "parameter": parameter, "n_records": int(len(x)), "ma_index": MA_INDEX, "ma_boundaries": list(MA_BOUNDARIES), "ma_weighting": weight_diag, "optimizer": {"optimizer": "L-BFGS-B", "success": True, "status": str(solution.message), "iterations": int(getattr(solution, "nit", 0)), "objective": float(solution.fun), "initialisation": "identity_matrix_zero_intercept", "penalty": float(penalty), "huber_delta": float(huber_delta)}, "source_support": mapping.diagnostics.get("ma_weighting", {}).get("class_counts", {}),
                   "loss_reduction": "sum", "sigma_ma_physical": _parameter_value(parameter, "sigma", 0.5) * float(mapping.y_scale[MA_INDEX]) if method == "balanced_mse" else None}
    return FittedAdapter(mapping, np.array(mapping.y_mean, copy=True), np.array(mapping.y_scale, copy=True), v_mean, u_mean, matrix, intercept, method, parameter, float(penalty), float(huber_delta), diagnostics)


def project_readouts(value: Any) -> np.ndarray:
    """Project ``(R,K,MA,alpha_R)`` to the physical ranges used downstream."""

    y = np.array(value, dtype=float, copy=True)
    if y.shape[-1] != READOUT_DIM:
        raise ValueError("readouts must end with four values: R, K, MA, alpha_R")
    y[..., :3] = np.maximum(y[..., :3], 0.0)
    y[..., 3] = np.clip(y[..., 3], 0.0, 89.999999)
    return y

