"""Fixed post-hoc trust/report-or-wait models for the MED4 study.

The module deliberately owns only the small risk models.  The frozen TT-MED4
predictor, the Huber calibrator, nested data construction, threshold selection,
and target evaluation are owned by the experiment driver.  Every public model
is a plain dictionary containing numpy arrays so that the resulting objects are
portable and auditable with joblib/npz without serialising torch modules.

The public contract is intentionally small::

    configs() -> list[dict]
    fit_model(config, F, yhat, ytrue, seed) -> dict
    predict_model(model, F, yhat) -> ndarray

``F`` is the frozen 15-dimensional feature matrix and ``yhat``/``ytrue`` are
the three CI class labels ``{-1, 0, +1}``.  Returned scores are error-oriented:
small means safer to release, and larger means more likely to be wrong.  The
models never change ``yhat``.

The implementation is a task adaptation of the registered literature ideas;
it is not a reimplementation of CGS, FIRMBOUND, DOS, or SelectiveNet.
"""

from __future__ import annotations

import runtime as rt  # must be imported before numerical libraries/cache users

import hashlib
import math
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
from scipy.linalg import cho_solve
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

import torch
from torch import nn


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)


N_FEATURES = 15
CLASS_VALUES = (-1, 0, 1)
CLASS_TO_INDEX = {-1: 0, 0: 1, 1: 2}
GP_LENGTHS = (0.5, 1.0, 2.0)
NEURAL_STEPS = 200
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.001
GRADIENT_CLIP = 5.0
REGULARIZER = 0.1


def configs() -> List[Dict[str, Any]]:
    """Return the nine pre-registered configurations in fixed order."""

    return [
        {"id": "ERROR-LR", "family": "ERROR-LR"},
        {"id": "GP-RISK@0.5", "family": "GP-RISK", "length": 0.5},
        {"id": "GP-RISK@1", "family": "GP-RISK", "length": 1.0},
        {"id": "GP-RISK@2", "family": "GP-RISK", "length": 2.0},
        {"id": "DOS-GATE", "family": "DOS-GATE", "cost": 0.10},
        {"id": "FLAT-CLINICAL", "family": "FLAT-CLINICAL"},
        {"id": "FACTORED-CLINICAL", "family": "FACTORED-CLINICAL"},
        {"id": "SELECTIVE-GATE@0.25", "family": "SELECTIVE-GATE", "q": 0.25},
        {"id": "SELECTIVE-GATE@0.50", "family": "SELECTIVE-GATE", "q": 0.50},
    ]


def _hash_value(value: Any) -> str:
    """Stable content hash used in model metadata."""

    h = hashlib.sha256()

    def visit(v: Any) -> None:
        if isinstance(v, np.ndarray):
            h.update(b"array")
            h.update(str(v.dtype).encode("utf-8"))
            h.update(repr(tuple(v.shape)).encode("utf-8"))
            h.update(np.ascontiguousarray(v).tobytes())
        elif isinstance(v, Mapping):
            h.update(b"mapping")
            for key in sorted(v, key=lambda x: str(x)):
                h.update(str(key).encode("utf-8"))
                visit(v[key])
        elif isinstance(v, (list, tuple)):
            h.update(b"sequence")
            for item in v:
                visit(item)
        elif isinstance(v, (float, np.floating)):
            h.update(repr(float(v)).encode("utf-8"))
        elif isinstance(v, (int, np.integer, bool)):
            h.update(repr(int(v)).encode("utf-8"))
        elif v is None:
            h.update(b"none")
        else:
            h.update(repr(v).encode("utf-8"))

    visit(value)
    return h.hexdigest()


def _as_matrix(F: Any) -> np.ndarray:
    a = np.asarray(F, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != N_FEATURES:
        raise ValueError(f"F must have shape (n,{N_FEATURES}), got {a.shape}")
    return a


def _as_labels(y: Any, name: str, n: int) -> np.ndarray:
    a = np.asarray(y)
    if a.ndim != 1 or len(a) != n:
        raise ValueError(f"{name} must have shape ({n},), got {a.shape}")
    try:
        out = a.astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integer CI classes") from exc
    if not np.isfinite(a.astype(np.float64)).all():
        raise ValueError(f"{name} contains non-finite labels")
    if not np.isin(out, np.asarray(CLASS_VALUES)).all():
        raise ValueError(f"{name} must use only CI classes {-1, 0, 1}")
    if not np.allclose(a.astype(np.float64), out.astype(np.float64)):
        raise ValueError(f"{name} contains non-integral labels")
    return out


def _preprocess(F: np.ndarray) -> Dict[str, Any]:
    """Fit the fixed training-only mean/SD preprocessing."""

    finite = np.isfinite(F)
    med = np.zeros(N_FEATURES, dtype=np.float64)
    mean = np.zeros(N_FEATURES, dtype=np.float64)
    sd = np.ones(N_FEATURES, dtype=np.float64)
    for j in range(N_FEATURES):
        vals = F[finite[:, j], j]
        if vals.size:
            med[j] = float(np.median(vals))
            mean[j] = float(np.mean(vals))
            sd[j] = max(float(np.std(vals, ddof=0)), 1e-6)
        else:
            med[j] = 0.0
            mean[j] = 0.0
            sd[j] = 1.0
    filled = np.where(finite, F, med[None, :])
    raw = (filled - mean[None, :]) / sd[None, :]
    clipped = np.clip(raw, -10.0, 10.0)
    return {
        "mean": mean,
        "sd": sd,
        "median": med,
        "ddof": 0,
        "floor": 1e-6,
        "clip": [-10.0, 10.0],
        "clip_fraction": float(np.mean(raw != clipped)) if raw.size else 0.0,
        "missing_fraction": float(np.mean(~finite)) if finite.size else 0.0,
        "n": int(F.shape[0]),
        "p": int(F.shape[1]),
    }


def _transform(pre: Mapping[str, Any], F: Any) -> np.ndarray:
    a = _as_matrix(F)
    med = np.asarray(pre["median"], dtype=np.float64)
    mean = np.asarray(pre["mean"], dtype=np.float64)
    sd = np.asarray(pre["sd"], dtype=np.float64)
    filled = np.where(np.isfinite(a), a, med[None, :])
    return np.clip((filled - mean[None, :]) / sd[None, :], -10.0, 10.0)


def _const_risk(e: np.ndarray) -> float:
    return float((np.sum(e) + 0.5) / (len(e) + 1.0))


def _binary_supported(e: np.ndarray) -> bool:
    return bool(len(e) >= 8 and np.sum(e > 0.5) >= 2 and np.sum(e <= 0.5) >= 2)


def _clinical_supported(ytrue: np.ndarray) -> bool:
    counts = np.asarray([np.sum(ytrue == c) for c in CLASS_VALUES], dtype=int)
    return bool(len(ytrue) >= 8 and np.sum(counts > 0) >= 2 and np.sum(counts >= 2) >= 2)


def _class_counts(y: np.ndarray) -> Dict[str, int]:
    return {str(c): int(np.sum(y == c)) for c in CLASS_VALUES}


def _base_meta(config: Mapping[str, Any], F: np.ndarray, yhat: np.ndarray,
               ytrue: np.ndarray, seed: int) -> Dict[str, Any]:
    e = (yhat != ytrue).astype(np.float64)
    pre = _preprocess(F)
    initial = {"seed": int(seed), "family": str(config["family"]), "config": dict(config)}
    model: Dict[str, Any] = {
        "api_version": "trust-gate-risk-models-v1",
        "id": str(config["id"]),
        "family": str(config["family"]),
        "config": dict(config),
        "seed": int(seed),
        "feature_dim": int(F.shape[1]),
        "pre": pre,
        "n": int(len(e)),
        "errors": int(np.sum(e)),
        "correct": int(np.sum(e < 0.5)),
        "error_rate": float(np.mean(e)),
        "true_class_counts": _class_counts(ytrue),
        "pred_class_counts": _class_counts(yhat),
        "target_error": e,
        "parameter_count_registered": 0,
        "parameter_count_effective": 0,
        "initial_hash": _hash_value(initial),
        "initial_state_hash": _hash_value(initial),
        "iterations": 0,
        "seconds": 0.0,
        "finite": True,
        "success": False,
    }
    return model


def _finish(model: Dict[str, Any], started: float) -> Dict[str, Any]:
    model["seconds"] = float(time.perf_counter() - started)
    # target_error is useful during fitting but need not be persisted in a
    # portable model object; retain only its hash and counts.
    e = np.asarray(model.pop("target_error"), dtype=np.float64)
    model["target_error_hash"] = _hash_value(e)
    model["model_hash"] = _hash_value(model)
    return model


def _fallback(model: Dict[str, Any], reason: str, score: float | None = None) -> Dict[str, Any]:
    model["status"] = "fallback_const"
    model["fallback"] = str(reason)
    model["success"] = False
    model["finite"] = True
    model["constant_score"] = float(_const_risk(np.asarray(model.get("_e", []), dtype=float)) if score is None else score)
    model["parameter_count_effective"] = 0
    return model


def _stable_bce_logits(eta: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(eta) - target * eta


def _fit_lr(model: Dict[str, Any], z: np.ndarray, e: np.ndarray) -> None:
    model["parameter_count_registered"] = N_FEATURES + 1
    model["parameter_count_effective"] = N_FEATURES + 1
    model["_e"] = e
    if not _binary_supported(e):
        _fallback(model, "insufficient_error_or_correct_support")
        return
    p = z.shape[1]

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = z @ theta[:p] + theta[p]
        value = float(np.mean(np.logaddexp(0.0, eta) - e * eta) + REGULARIZER * np.dot(theta[:p], theta[:p]))
        delta = (expit(eta) - e) / len(e)
        grad = np.r_[z.T @ delta + 2.0 * REGULARIZER * theta[:p], np.sum(delta)]
        return value, grad

    theta0 = np.zeros(p + 1, dtype=np.float64)
    model["initial_hash"] = _hash_value(theta0)
    model["initial_state_hash"] = model["initial_hash"]
    sol = minimize(
        objective,
        theta0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 200, "ftol": 1e-9, "gtol": 1e-9},
    )
    coef = np.asarray(sol.x, dtype=np.float64)
    finite = bool(np.isfinite(coef).all() and np.isfinite(float(sol.fun)))
    model.update(
        coef=coef,
        objective=float(sol.fun) if np.isfinite(sol.fun) else None,
        iterations=int(getattr(sol, "nit", 0)),
        status=str(sol.message),
        success=bool(sol.success and finite),
        finite=finite,
    )
    if not finite:
        model["fallback"] = "nonfinite_optimizer_result"
        model["constant_score"] = _const_risk(e)


def _rbf(X: np.ndarray, Z: np.ndarray, length: float) -> np.ndarray:
    xx = np.sum(X * X, axis=1)[:, None]
    zz = np.sum(Z * Z, axis=1)[None, :]
    sq = np.maximum(xx + zz - 2.0 * X @ Z.T, 0.0)
    return np.exp(-0.5 * sq / (float(length) ** 2))


def _fit_gp(model: Dict[str, Any], z: np.ndarray, e: np.ndarray, length: float) -> None:
    model.update(length=float(length), amplitude=1.0, noise=0.05,
                 parameter_count_registered=0, parameter_count_effective=0)
    model["_e"] = e
    if not _binary_supported(e):
        model["jitter_attempts"] = []
        _fallback(model, "insufficient_error_or_correct_support")
        return
    const = _const_risk(e)
    K = _rbf(z, z, length)
    attempts: List[Dict[str, Any]] = []
    chol = None
    chosen = None
    for jitter in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4):
        try:
            # The registered noise is the diagonal variance; jitter is only a
            # numerical stabilizer and is kept separate in the ledger.
            chol = np.linalg.cholesky(K + (0.05 + jitter) * np.eye(len(z), dtype=np.float64))
            chosen = float(jitter)
            attempts.append({"jitter": float(jitter), "success": True})
            break
        except np.linalg.LinAlgError as exc:
            attempts.append({"jitter": float(jitter), "success": False, "error": str(exc)})
    model["jitter_attempts"] = attempts
    if chol is None or not np.isfinite(chol).all():
        _fallback(model, "cholesky_failure", score=const)
        model["finite"] = False
        return
    alpha = cho_solve((chol, True), e - const)
    finite = bool(np.isfinite(alpha).all())
    model.update(
        train_z=np.asarray(z, dtype=np.float64),
        chol=np.asarray(chol, dtype=np.float64),
        alpha=np.asarray(alpha, dtype=np.float64),
        constant_mean=float(const),
        jitter=chosen,
        status="finite_fixed_kernel" if finite else "nonfinite_cholesky_solution",
        success=finite,
        finite=finite,
    )
    if not finite:
        model["fallback"] = "nonfinite_cholesky_solution"
        model["constant_score"] = const


class _ClinicalNet(nn.Module):
    def __init__(self, out_dim: int, seed: int) -> None:
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(int(seed))
            self.fc1 = nn.Linear(N_FEATURES, 8)
            self.out = nn.Linear(8, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(torch.nn.functional.silu(self.fc1(x)))


class _SelectiveNet(nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(int(seed))
            self.fc1 = nn.Linear(N_FEATURES, 8)
            self.gate = nn.Linear(8, 1)
            self.aux = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.nn.functional.silu(self.fc1(x))
        return self.gate(h), self.aux(h)


def _state_dict_arrays(net: nn.Module) -> Dict[str, np.ndarray]:
    return {str(k): v.detach().cpu().numpy().astype(np.float64, copy=True) for k, v in net.state_dict().items()}


def _state_hash(state: Mapping[str, np.ndarray]) -> str:
    return _hash_value({k: np.asarray(v) for k, v in state.items()})


def _make_net_state(net: nn.Module, state: Mapping[str, np.ndarray]) -> None:
    tensors = {k: torch.as_tensor(np.asarray(v), dtype=torch.float64) for k, v in state.items()}
    net.load_state_dict(tensors, strict=True)


def _fit_dos(model: Dict[str, Any], z: np.ndarray, e: np.ndarray, seed: int, cost: float) -> None:
    model.update(parameter_count_registered=137, parameter_count_effective=137, cost=float(cost), steps=NEURAL_STEPS,
                 learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, gradient_clip=GRADIENT_CLIP)
    model["_e"] = e
    net = _ClinicalNet(1, seed)
    init_state = _state_dict_arrays(net)
    model["initial_hash"] = _state_hash(init_state)
    model["initial_state_hash"] = model["initial_hash"]
    if not _binary_supported(e):
        _fallback(model, "insufficient_error_or_correct_support")
        return
    opt = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    xx = torch.as_tensor(z, dtype=torch.float64)
    ee = torch.as_tensor(e, dtype=torch.float64)
    loss_value = math.nan
    for step in range(NEURAL_STEPS):
        opt.zero_grad()
        g = torch.sigmoid(net(xx).flatten())
        loss = torch.mean(g * ee + (1.0 - g) * float(cost))
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), GRADIENT_CLIP)
        opt.step()
        loss_value = float(loss.detach().cpu())
    state = _state_dict_arrays(net)
    finite = bool(np.isfinite(loss_value) and all(np.isfinite(v).all() for v in state.values()))
    model.update(state=state, loss=loss_value, iterations=NEURAL_STEPS,
                 status="finite_200_step" if finite else "nonfinite_adamw",
                 finite=finite, success=finite)
    if not finite:
        model["fallback"] = "nonfinite_optimizer_result"
        model["constant_score"] = _const_risk(e)


def _fit_flat(model: Dict[str, Any], z: np.ndarray, ytrue: np.ndarray, seed: int) -> None:
    model.update(parameter_count_registered=146, parameter_count_effective=146, steps=NEURAL_STEPS,
                 learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, gradient_clip=GRADIENT_CLIP)
    model["_ytrue"] = ytrue
    net = _ClinicalNet(2, seed)
    init_state = _state_dict_arrays(net)
    model["initial_hash"] = _state_hash(init_state)
    model["initial_state_hash"] = model["initial_hash"]
    counts = np.asarray([np.sum(ytrue == c) for c in CLASS_VALUES], dtype=int)
    model["class_counts"] = {str(c): int(n) for c, n in zip(CLASS_VALUES, counts)}
    prior = (counts.astype(np.float64) + 0.5) / (float(len(ytrue)) + 1.5)
    prior /= np.sum(prior)
    model["class_prior"] = prior
    if not _clinical_supported(ytrue):
        _fallback(model, "insufficient_class_support", score=None)
        model["constant_class_prior"] = prior
        return
    opt = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    xx = torch.as_tensor(z, dtype=torch.float64)
    yy = torch.as_tensor(np.asarray([CLASS_TO_INDEX[int(v)] for v in ytrue]), dtype=torch.long)
    loss_value = math.nan
    for _ in range(NEURAL_STEPS):
        opt.zero_grad()
        ab = net(xx)
        logits = torch.cat([ab[:, :1], torch.zeros_like(ab[:, :1]), ab[:, 1:2]], dim=1)
        loss = torch.nn.functional.cross_entropy(logits, yy)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), GRADIENT_CLIP)
        opt.step()
        loss_value = float(loss.detach().cpu())
    state = _state_dict_arrays(net)
    finite = bool(np.isfinite(loss_value) and all(np.isfinite(v).all() for v in state.values()))
    model.update(state=state, loss=loss_value, iterations=NEURAL_STEPS,
                 status="finite_200_step" if finite else "nonfinite_adamw",
                 finite=finite, success=finite)
    if not finite:
        model["fallback"] = "nonfinite_optimizer_result"
        model["constant_class_prior"] = prior


def _fit_factored(model: Dict[str, Any], z: np.ndarray, ytrue: np.ndarray, seed: int) -> None:
    model.update(parameter_count_registered=146, parameter_count_effective=146, steps=NEURAL_STEPS,
                 learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, gradient_clip=GRADIENT_CLIP)
    model["_ytrue"] = ytrue
    net = _ClinicalNet(2, seed)
    init_state = _state_dict_arrays(net)
    model["initial_hash"] = _state_hash(init_state)
    model["initial_state_hash"] = model["initial_hash"]
    counts = np.asarray([np.sum(ytrue == c) for c in CLASS_VALUES], dtype=int)
    model["class_counts"] = {str(c): int(n) for c, n in zip(CLASS_VALUES, counts)}
    prior = (counts.astype(np.float64) + 0.5) / (float(len(ytrue)) + 1.5)
    prior /= np.sum(prior)
    model["class_prior"] = prior
    if not _clinical_supported(ytrue):
        _fallback(model, "insufficient_class_support", score=None)
        model["constant_class_prior"] = prior
        return
    abnormal = (ytrue != 0).astype(np.float64)
    abnormal_sum = int(np.sum(abnormal))
    direction = (ytrue == 1).astype(np.float64)
    direction_count = int(np.sum(abnormal))
    high_count = int(np.sum(direction * abnormal))
    if abnormal_sum:
        beta_prior = float((high_count + 0.5) / (abnormal_sum + 1.0))
    else:
        beta_prior = 0.5
    beta_frozen = bool(high_count == 0 or high_count == abnormal_sum)
    model.update(beta_prior=beta_prior, beta_frozen=beta_frozen,
                 parameter_count_effective=137 if beta_frozen else 146,
                 abnormal_support=abnormal_sum, high_support=high_count)
    beta_initial_weight = net.out.weight.detach()[1].clone()
    beta_initial_bias = net.out.bias.detach()[1].clone()
    model["beta_initial_hash"] = _state_hash({
        "weight": beta_initial_weight.detach().cpu().numpy(),
        "bias": beta_initial_bias.detach().cpu().numpy(),
    })
    opt = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    xx = torch.as_tensor(z, dtype=torch.float64)
    aa = torch.as_tensor(abnormal, dtype=torch.float64)
    dd = torch.as_tensor(direction, dtype=torch.float64)
    loss_value = math.nan
    for _ in range(NEURAL_STEPS):
        opt.zero_grad()
        ab = net(xx)
        rho = torch.sigmoid(ab[:, 0])
        if beta_frozen:
            beta = torch.full_like(rho, float(beta_prior))
        else:
            beta = torch.sigmoid(ab[:, 1])
        loss_rho = torch.nn.functional.binary_cross_entropy(rho, aa)
        # The direction term is only defined on abnormal reference cases.
        if abnormal_sum:
            loss_beta = torch.nn.functional.binary_cross_entropy(beta[aa > 0.5], dd[aa > 0.5])
        else:
            loss_beta = torch.zeros((), dtype=torch.float64)
        loss = loss_rho + loss_beta * (float(abnormal_sum) / float(len(ytrue)))
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), GRADIENT_CLIP)
        opt.step()
        if beta_frozen:
            # A zero gradient is not sufficient with AdamW: decoupled weight
            # decay would still move the unused direction head.  Restore its
            # exact initial row so the effective parameter count is truthful.
            with torch.no_grad():
                net.out.weight[1].copy_(beta_initial_weight)
                net.out.bias[1].copy_(beta_initial_bias)
        loss_value = float(loss.detach().cpu())
    state = _state_dict_arrays(net)
    model["beta_final_hash"] = _state_hash({"weight": state["out.weight"][1], "bias": state["out.bias"][1]})
    finite = bool(np.isfinite(loss_value) and all(np.isfinite(v).all() for v in state.values()))
    model.update(state=state, loss=loss_value, iterations=NEURAL_STEPS,
                 status="finite_200_step" if finite else "nonfinite_adamw",
                 finite=finite, success=finite)
    if not finite:
        model["fallback"] = "nonfinite_optimizer_result"
        model["constant_class_prior"] = prior


def _fit_selective(model: Dict[str, Any], z: np.ndarray, e: np.ndarray, seed: int, q: float) -> None:
    model.update(parameter_count_registered=146, parameter_count_effective=146, steps=NEURAL_STEPS,
                 learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, gradient_clip=GRADIENT_CLIP,
                 q=float(q))
    model["_e"] = e
    net = _SelectiveNet(seed)
    init_state = _state_dict_arrays(net)
    model["initial_hash"] = _state_hash(init_state)
    model["initial_state_hash"] = model["initial_hash"]
    if not _binary_supported(e):
        _fallback(model, "insufficient_error_or_correct_support")
        return
    opt = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    xx = torch.as_tensor(z, dtype=torch.float64)
    ee = torch.as_tensor(e, dtype=torch.float64)
    loss_value = math.nan
    for _ in range(NEURAL_STEPS):
        opt.zero_grad()
        gate_logits, aux_logits = net(xx)
        g = torch.sigmoid(gate_logits.flatten())
        h = torch.sigmoid(aux_logits.flatten())
        selective = torch.sum(g * ee) / torch.clamp(torch.sum(g), min=1e-6)
        coverage_penalty = 32.0 * torch.relu(float(q) - torch.mean(g)) ** 2
        aux = torch.nn.functional.binary_cross_entropy(h, ee)
        loss = 0.5 * (selective + coverage_penalty) + 0.5 * aux
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), GRADIENT_CLIP)
        opt.step()
        loss_value = float(loss.detach().cpu())
    state = _state_dict_arrays(net)
    finite = bool(np.isfinite(loss_value) and all(np.isfinite(v).all() for v in state.values()))
    model.update(state=state, loss=loss_value, iterations=NEURAL_STEPS,
                 status="finite_200_step" if finite else "nonfinite_adamw",
                 finite=finite, success=finite)
    if not finite:
        model["fallback"] = "nonfinite_optimizer_result"
        model["constant_score"] = _const_risk(e)


def fit_model(config: Mapping[str, Any], F: Any, yhat: Any, ytrue: Any, seed: int) -> Dict[str, Any]:
    """Fit one fixed post-hoc report/wait model.

    Only training-side labels are consumed here.  The caller is responsible
    for constructing cross-fitted labels/features before invoking this API.
    """

    if not isinstance(config, Mapping) or "family" not in config or "id" not in config:
        raise ValueError("config must contain id and family")
    Fm = _as_matrix(F)
    yh = _as_labels(yhat, "yhat", len(Fm))
    yt = _as_labels(ytrue, "ytrue", len(Fm))
    family = str(config["family"])
    if family not in {str(c["family"]) for c in configs()}:
        raise ValueError(f"unknown risk-model family {family!r}")
    model = _base_meta(config, Fm, yh, yt, int(seed))
    e = (yh != yt).astype(np.float64)
    z = _transform(model["pre"], Fm)
    started = time.perf_counter()
    try:
        if family == "ERROR-LR":
            _fit_lr(model, z, e)
        elif family == "GP-RISK":
            _fit_gp(model, z, e, float(config["length"]))
        elif family == "DOS-GATE":
            _fit_dos(model, z, e, int(seed), float(config.get("cost", 0.10)))
        elif family == "FLAT-CLINICAL":
            _fit_flat(model, z, yt, int(seed))
        elif family == "FACTORED-CLINICAL":
            _fit_factored(model, z, yt, int(seed))
        elif family == "SELECTIVE-GATE":
            _fit_selective(model, z, e, int(seed), float(config["q"]))
    except Exception as exc:
        # A numerical/optimizer failure is retained as an explicit fallback;
        # it is never silently removed from the candidate table.
        model["status"] = "exception_fallback"
        model["fallback"] = f"{type(exc).__name__}: {exc}"
        model["success"] = False
        model["finite"] = False
        model["constant_score"] = _const_risk(e)
    model.pop("_e", None)
    model.pop("_ytrue", None)
    model["fit_seconds_internal"] = float(time.perf_counter() - started)
    return _finish(model, started)


def _forward_clinical(model: Mapping[str, Any], z: np.ndarray) -> np.ndarray:
    net = _ClinicalNet(2, int(model.get("seed", 0)))
    _make_net_state(net, model["state"])
    with torch.no_grad():
        ab = net(torch.as_tensor(z, dtype=torch.float64)).cpu().numpy()
    return ab


def _forward_dos(model: Mapping[str, Any], z: np.ndarray) -> np.ndarray:
    net = _ClinicalNet(1, int(model.get("seed", 0)))
    _make_net_state(net, model["state"])
    with torch.no_grad():
        return expit(net(torch.as_tensor(z, dtype=torch.float64)).cpu().numpy().reshape(-1))


def _forward_selective(model: Mapping[str, Any], z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    net = _SelectiveNet(int(model.get("seed", 0)))
    _make_net_state(net, model["state"])
    with torch.no_grad():
        g, h = net(torch.as_tensor(z, dtype=torch.float64))
    return expit(g.cpu().numpy().reshape(-1)), expit(h.cpu().numpy().reshape(-1))


def _prior_probs(model: Mapping[str, Any], n: int) -> np.ndarray:
    p = np.asarray(model.get("class_prior", [1 / 3, 1 / 3, 1 / 3]), dtype=np.float64)
    p = np.clip(p, 1e-15, np.inf)
    p /= np.sum(p)
    return np.repeat(p[None, :], n, axis=0)


def predict_class_probs(model: Mapping[str, Any], F: Any) -> np.ndarray:
    """Return class probabilities for clinical candidates (test helper/API)."""

    z = _transform(model["pre"], F)
    family = str(model["family"])
    if family not in {"FLAT-CLINICAL", "FACTORED-CLINICAL"}:
        raise ValueError("class probabilities are defined only for clinical models")
    if model.get("status") == "fallback_const" or "state" not in model or not model.get("finite", False):
        return _prior_probs(model, len(z))
    ab = _forward_clinical(model, z)
    if family == "FLAT-CLINICAL":
        logits = np.column_stack([ab[:, 0], np.zeros(len(ab)), ab[:, 1]])
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


def predict_model(model: Mapping[str, Any], F: Any, yhat: Any) -> np.ndarray:
    """Predict error-oriented release scores; lower values are safer."""

    a = _as_matrix(F)
    yh = _as_labels(yhat, "yhat", len(a))
    z = _transform(model["pre"], a)
    family = str(model["family"])
    if model.get("status") in {"fallback_const", "exception_fallback"} or not model.get("finite", False):
        if family in {"FLAT-CLINICAL", "FACTORED-CLINICAL"}:
            p = _prior_probs(model, len(a))
            return np.asarray([1.0 - p[i, CLASS_TO_INDEX[int(c)]] for i, c in enumerate(yh)], dtype=np.float64)
        return np.full(len(a), float(model.get("constant_score", 1.0)), dtype=np.float64)
    try:
        if family == "ERROR-LR":
            coef = np.asarray(model["coef"], dtype=np.float64)
            score = expit(z @ coef[:-1] + coef[-1])
        elif family == "GP-RISK":
            Kx = _rbf(z, np.asarray(model["train_z"], dtype=np.float64), float(model["length"]))
            score = float(model["constant_mean"]) + Kx @ np.asarray(model["alpha"], dtype=np.float64)
            score = np.clip(score, 0.0, 1.0)
        elif family == "DOS-GATE":
            g = _forward_dos(model, z)
            score = 1.0 - g
        elif family in {"FLAT-CLINICAL", "FACTORED-CLINICAL"}:
            p = predict_class_probs(model, a)
            score = np.asarray([1.0 - p[i, CLASS_TO_INDEX[int(c)]] for i, c in enumerate(yh)], dtype=np.float64)
        elif family == "SELECTIVE-GATE":
            g, _ = _forward_selective(model, z)
            score = 1.0 - g
        else:
            raise ValueError(f"unknown model family {family!r}")
    except Exception:
        score = np.full(len(a), float(model.get("constant_score", 1.0)), dtype=np.float64)
    return np.clip(np.asarray(score, dtype=np.float64), 0.0, 1.0)


__all__ = ["configs", "fit_model", "predict_model", "predict_class_probs"]
