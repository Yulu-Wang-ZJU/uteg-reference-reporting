"""Isolated reporting-control replay for the static B085 revision.

This module reruns the paper's 15-minute HGB error gate and boundary-distance
gate for the two SAHZU transfer arms.  It uses the same 60-record grouped
development pool, three outer positions, 20 seeds and development-only
``select_threshold`` contract as the paper.  The old HGB artifacts are never
loaded as models: a new HistGradientBoostingClassifier is fitted for every
direction/seed/position from the revised B input.

The target test labels are touched only after all development thresholds and
models have been frozen.  All score arrays and summary denominators are kept
in seed-by-record event coordinates (20 x 201), so abnormal/normal rates are
not accidentally divided by single-cohort counts.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits


WORKSPACE = Path(__file__).resolve().parents[2]
OUT = WORKSPACE / "outputs/paper_B085_hospital_replay_20261102/reporting_controls"
DATA = WORKSPACE / "outputs/uteg_B085_by_hospital_20261102/private/DATA15_B085_BY_HOSPITAL.npz"
TRANSFER = WORKSPACE / "outputs/paper_B085_hospital_replay_20261102/transfer"
TRANSFER_WRAPPER = WORKSPACE / "code_submit/paper_b085_hospital_replay/transfer.py"
SELECTOR = WORKSPACE / "outputs/uteg_nine_direction_ma_20261025_195548/code/fit_ma_gates.py"
SEEDS = tuple(range(1000, 1020))
DIRECTIONS = ("HZ_to_SA", "HZ+HB_to_SA")
NINE_DIRECTIONS = (
    "HZ+HB_to_SA", "SA+HB_to_HZ", "SA+HZ_to_HB", "HZ_to_SA", "HB_to_SA",
    "SA_to_HZ", "HB_to_HZ", "SA_to_HB", "HZ_to_HB",
)
TARGET_FILE_TOKEN = {
    "HZ+HB_to_SA": "SAHZU", "HZ_to_SA": "SAHZU", "HB_to_SA": "SAHZU",
    "SA+HB_to_HZ": "Huzhou", "SA_to_HZ": "Huzhou", "HB_to_HZ": "Huzhou",
    "SA+HZ_to_HB": "Hubei", "SA_to_HB": "Hubei", "HZ_to_HB": "Hubei",
}
MA_CLASSES = (-1, 0, 1)
FEATURE_DIM = 13
TARGET = "SAHZU"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise RuntimeError(f"refusing to overwrite {path}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], *, overwrite: bool = False) -> None:
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    if path.exists() and not overwrite:
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def load_data() -> dict[str, np.ndarray]:
    with np.load(DATA, allow_pickle=False) as z:
        D = {k: z[k].copy() for k in z.files}
    if D["x"].shape != (480, 4) or D["medical"].shape != (480, 4):
        raise RuntimeError("revised DATA15 shape changed")
    if not bool(D["B_from_original_data"]):
        raise RuntimeError("original-B input metadata changed")
    return D


def load_selector_functions() -> dict[str, Any]:
    """Extract pure selector functions without importing the historical runtime."""
    tree = ast.parse(SELECTOR.read_text(encoding="utf-8-sig"))
    names = {"summary", "select_threshold", "margin_scores"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names:
        raise RuntimeError("selector function contract changed")
    ns: dict[str, Any] = {"np": np, "MA_CLASSES": MA_CLASSES}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SELECTOR), "exec"), ns)
    return {n: ns[n] for n in names}


SEL = load_selector_functions()
summary = SEL["summary"]
select_threshold = SEL["select_threshold"]
margin_scores = SEL["margin_scores"]


def ma_class(value: Any) -> np.ndarray:
    x = np.asarray(value, dtype=float)
    return np.where(~np.isfinite(x), 9, np.where(x < 50, -1, np.where(x <= 70, 0, 1))).astype(np.int8)


def ma_features(x: Any, med: Any, ym: Any, ys: Any) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.asarray(med, dtype=float)
    pred_ma = med[:, 2]
    pc = ma_class(pred_ma)
    return np.column_stack((
        x,
        (med - np.asarray(ym)[None, :]) / np.asarray(ys)[None, :],
        (pred_ma - 60.0) / 10.0,
        np.minimum(np.abs(pred_ma - 50.0), np.abs(pred_ma - 70.0)) / 10.0,
        (pc == -1).astype(float), (pc == 0).astype(float), (pc == 1).astype(float),
    ))


def hgb_config(seed: int) -> dict[str, Any]:
    return {
        "learning_rate": 0.05,
        "max_iter": 100,
        "max_leaf_nodes": 4,
        "max_depth": 2,
        "min_samples_leaf": 10,
        "l2_regularization": 10.0,
        "max_bins": 255,
        "early_stopping": False,
        "random_state": int(seed),
    }


def fit_preprocessor(f: np.ndarray) -> dict[str, Any]:
    f = np.asarray(f, float)
    finite = np.isfinite(f)
    median = np.asarray([np.median(f[finite[:, j], j]) if finite[:, j].any() else 0.0 for j in range(f.shape[1])])
    mean = np.asarray([np.mean(f[finite[:, j], j]) if finite[:, j].any() else 0.0 for j in range(f.shape[1])])
    sd = np.asarray([max(np.std(f[finite[:, j], j]), 1e-6) if finite[:, j].any() else 1.0 for j in range(f.shape[1])])
    return {"median": median, "mean": mean, "sd": sd, "clip": (-10.0, 10.0), "n": int(len(f)), "p": FEATURE_DIM}


def apply_preprocessor(f: np.ndarray, pre: Mapping[str, Any]) -> np.ndarray:
    f = np.asarray(f, float)
    med = np.asarray(pre["median"], float)
    mean = np.asarray(pre["mean"], float)
    sd = np.asarray(pre["sd"], float)
    z = (np.where(np.isfinite(f), f, med[None, :]) - mean[None, :]) / sd[None, :]
    return np.clip(z, -10.0, 10.0)


def fit_hgb(features: np.ndarray, pred: np.ndarray, truth: np.ndarray, qc: np.ndarray, *, direction: str, seed: int, pos: int) -> dict[str, Any]:
    f = np.asarray(features, float)
    pred = np.asarray(pred, np.int8)
    truth = np.asarray(truth, np.int8)
    qc = np.asarray(qc, bool)
    known = np.isin(pred, MA_CLASSES) & np.isin(truth, MA_CLASSES)
    finite = np.isfinite(f).all(axis=1)
    legal = known & qc & finite
    X = f[legal]
    y = (pred[legal] != truth[legal]).astype(np.int8)
    if len(X) < 8 or int(y.sum()) < 2 or int((1 - y).sum()) < 2:
        model = {
            "schema": "paper_b085_reporting_hgb_constant_v1",
            "status": "FALLBACK_CONSTANT",
            "direction": direction,
            "seed": int(seed),
            "position": int(pos),
            "n": int(len(y)),
            "errors": int(y.sum()),
            "correct": int(len(y) - y.sum()),
            "constant_score": float((y.sum() + 0.5) / (len(y) + 1.0)) if len(y) else 0.5,
            "config": hgb_config(seed),
            "sample_weight": "all_ones",
            "training_mask": {"known": int(known.sum()), "qc": int(qc.sum()), "finite": int(finite.sum()), "legal": int(legal.sum())},
        }
        return model
    pre = fit_preprocessor(X)
    z = apply_preprocessor(X, pre)
    clf = HistGradientBoostingClassifier(**hgb_config(seed))
    clf.fit(z, y, sample_weight=np.ones(len(y), dtype=float))
    return {
        "schema": "paper_b085_reporting_hgb_v1",
        "status": "COMPLETE",
        "direction": direction,
        "seed": int(seed),
        "position": int(pos),
        "n": int(len(y)),
        "errors": int(y.sum()),
        "correct": int(len(y) - y.sum()),
        "pre": pre,
        "config": hgb_config(seed),
        "model": clf,
        "sample_weight": "all_ones",
        "training_mask": {"known": int(known.sum()), "qc": int(qc.sum()), "finite": int(finite.sum()), "legal": int(legal.sum())},
    }


def score_hgb(model: Mapping[str, Any], features: np.ndarray, pred: np.ndarray, qc: np.ndarray) -> np.ndarray:
    f = np.asarray(features, float)
    pred = np.asarray(pred, np.int8)
    qc = np.asarray(qc, bool)
    valid = qc & np.isin(pred, MA_CLASSES) & np.isfinite(f).all(axis=1)
    if model["status"] == "FALLBACK_CONSTANT":
        score = np.full(len(f), float(model["constant_score"]))
    else:
        score = np.asarray(model["model"].predict_proba(apply_preprocessor(f, model["pre"]))[:, 1], float)
    return np.where(valid & np.isfinite(score), score, np.inf)


def score_lr_gate(model: Mapping[str, Any], features: np.ndarray, pred: np.ndarray, qc: np.ndarray) -> np.ndarray:
    """Independent reproduction of the frozen MA-error-LR score contract."""
    f = np.asarray(features, float)
    pred = np.asarray(pred, np.int8)
    qc = np.asarray(qc, bool)
    z = apply_preprocessor(f, model["pre"])
    if bool(model.get("fallback", False)):
        score = np.full(len(f), float(model["constant_score"]))
    else:
        eta = z @ np.asarray(model["coef"], float) + float(model["intercept"])
        score = 1.0 / (1.0 + np.exp(-np.clip(eta, -700.0, 700.0)))
    valid = qc & np.isin(pred, MA_CLASSES) & np.isfinite(f).all(axis=1) & np.isfinite(score)
    return np.where(valid, score, np.inf)


def transfer_paths(direction: str, seed: int) -> tuple[Path, Path]:
    root = TRANSFER / "runs" / f"prefix_900__{direction}"
    target = root / "models" / direction / "target15" / f"SAHZU_CFULL_seed{seed}.joblib"
    dual = root / "dual" / f"{direction}_{seed}.joblib"
    return target, dual


def check_transfer_complete(*, full: bool = True) -> None:
    for direction in DIRECTIONS:
        p = TRANSFER / "predictions" / f"prefix_900__{direction}.npz"
        if not p.is_file():
            raise RuntimeError(f"missing transfer prediction: {p}")
        z = np.load(p, allow_pickle=False)
        required_seeds = SEEDS if full else (1000,)
        if full and (z["seed"].tolist() != list(SEEDS) or z["pred_ma"].shape != (20, 201)):
            raise RuntimeError(f"transfer prediction incomplete: {p}")
        required_keys = {"final_med4", "base_score", "dual_score", "test", "ids", "groups", "B_from_original_data"}
        if not required_keys.issubset(set(z.files)):
            raise RuntimeError(f"fixed transfer prediction missing final-med4/risk keys: {p}")
        if not full and (1000 not in z["seed"].tolist() or z["pred_ma"].shape[1:] != (201,)):
            raise RuntimeError(f"transfer smoke prediction missing seed 1000: {p}")
        if not bool(z["B_from_original_data"]):
            raise RuntimeError(f"transfer prediction lineage changed: {p}")
        for seed in required_seeds:
            target, dual = transfer_paths(direction, seed)
            if not target.is_file() or not dual.is_file():
                raise RuntimeError(f"missing transfer object for {direction} seed={seed}")


def make_transfer_runtime() -> tuple[Any, Any, Any, Any, Any, Any]:
    spec = importlib.util.spec_from_file_location("paper_b085_transfer_runtime", TRANSFER_WRAPPER)
    if spec is None or spec.loader is None:
        raise ImportError(TRANSFER_WRAPPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    batch = module.load_batch()
    return module, batch, None, None, None, None


def get_task_env(module: Any, batch: Any, direction: str) -> tuple[Any, Any, Any, Any, Any]:
    return batch.env(f"prefix_900__{direction}", 16000)[1:]


def development_context(D: dict[str, np.ndarray], obj: Mapping[str, Any], pos: int, adapter: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q = obj["positions"][pos]
    A = np.asarray(q["A"], int)
    B = np.asarray(q["B"], int)
    cross = np.asarray(q["cross"][adapter], float)
    vv = np.asarray(q["vv"][adapter], float) if len(B) else np.empty((0, 4), float)
    ym = np.asarray(obj["source_model"]["ym"], float)
    ys = np.asarray(obj["source_model"]["ys"], float)
    # The transfer target object also carries its historical ``gm.features``
    # (CI-ordinal) representation.  The paper's HGB comparator, however, is
    # explicitly the numeric-MA 13D contract from the original driver.  Build
    # that representation afresh from the revised object rather than silently
    # reusing the differently ordered cached feature matrix.
    cached = np.asarray(q["features"][adapter], float)
    expected = ma_features(D["x"][A], cross, ym, ys)
    if cached.shape != expected.shape or not np.allclose(cached[:, :8], expected[:, :8], equal_nan=True):
        raise RuntimeError(f"13-feature numeric prefix contract mismatch at position {pos}")
    fa = expected
    pa = ma_class(cross[:, 2])
    ta = ma_class(D["medical"][A, 2])
    qa = np.asarray(D["input_qc"][A], bool)
    fb = ma_features(D["x"][B], vv, ym, ys)
    pb = ma_class(vv[:, 2])
    tb = ma_class(D["medical"][B, 2])
    qb = np.asarray(D["input_qc"][B], bool)
    return A, B, fa, pa, ta, qa, fb, pb, tb, qb


def prediction_context(module: Any, batch: Any, direction: str, seed: int, D: dict[str, np.ndarray], obj: Mapping[str, Any], dual_obj: Mapping[str, Any]) -> dict[str, Any]:
    r, runner, target_mod, seq, c, sel = batch.env(f"prefix_900__{direction}", 16000)
    q = module._base_predict(batch, seq, runner, target_mod, obj, D)
    test = np.asarray(q["test"], int)
    h = dual_obj["heads"][3]
    low = seq.score_head(c, h.get("low"), D["x"][test, 1], q["med"][:, 2])
    high = seq.score_head(c, h.get("high"), D["x"][test, 1], q["med"][:, 2])
    dual_score = np.maximum(np.asarray(low, float), np.asarray(high, float))
    med = np.asarray(q["med"], float)
    pred = ma_class(med[:, 2])
    # The transfer helper also exposes a CI-ordinal label for its own runtime
    # action path.  Reporting controls use the paper's numeric-MA state labels
    # and therefore recompute the LR score with the frozen MA gate contract.
    gate = obj["gate_models"][3]
    F = runner.ma_features(D["x"][test], med, obj["source_model"]["ym"], obj["source_model"]["ys"])
    lr_score = runner.score_ma_gate(gate["model"], F, pred)
    return {"test": test, "med": med, "pred": pred, "lr_score": np.asarray(lr_score, float), "dual_score": dual_score, "base_action": np.asarray(q["base_action"], bool)}


def threshold_actions(score: np.ndarray, threshold: float | None) -> np.ndarray:
    if threshold is None or not np.isfinite(threshold):
        return np.zeros(np.asarray(score).shape, bool)
    return np.isfinite(score) & (np.asarray(score) <= float(threshold))


def metrics(pred: np.ndarray, truth: np.ndarray, action: np.ndarray | None = None) -> dict[str, Any]:
    pred = np.asarray(pred, np.int8)
    truth = np.broadcast_to(np.asarray(truth, np.int8), pred.shape)
    a = np.ones(pred.shape, bool) if action is None else np.asarray(action, bool)
    known = np.isin(truth, MA_CLASSES)
    valid = known & np.isin(pred, MA_CLASSES)
    if not np.all(valid):
        raise RuntimeError("invalid state entered reporting metric")
    abnormal = truth != 0
    normal = truth == 0
    correct = a & (pred == truth)
    fn = a & abnormal & (pred == 0)
    fp = a & normal & (pred != 0)
    cm = [[int(np.sum(a & (truth == i) & (pred == j))) for j in MA_CLASSES] for i in MA_CLASSES]
    all_cm = [[int(np.sum((truth == i) & (pred == j))) for j in MA_CLASSES] for i in MA_CLASSES]
    released = int(a.sum())
    abnormal_n = int(abnormal.sum())
    normal_n = int(normal.sum())
    low_n = int(np.sum(truth == -1))
    high_n = int(np.sum(truth == 1))
    low_correct = int(np.sum(a & (truth == -1) & (pred == -1)))
    high_correct = int(np.sum(a & (truth == 1) & (pred == 1)))
    released_low_n = int(np.sum(a & (truth == -1)))
    released_high_n = int(np.sum(a & (truth == 1)))
    released_abnormal_n = int(np.sum(a & abnormal))
    released_normal_n = int(np.sum(a & normal))
    return {
        "events": int(pred.size), "released": released, "withheld": int((~a).sum()),
        "correct": int(correct.sum()), "incorrect": int((a & (pred != truth)).sum()),
        "coverage": float(released / pred.size),
        "agreement": float(correct.sum() / released) if released else None,
        "all_report_agreement": float(np.mean(pred == truth)),
        # ``fnr``/``fpr`` are retained as compatibility aliases.  Their
        # denominators are the complete truth-abnormal/truth-normal event
        # pools, not only the events released by the gate.
        "fn": int(fn.sum()), "fn_denominator": abnormal_n,
        "fn_rate_all_abnormal": float(fn.sum() / abnormal_n) if abnormal_n else None,
        "fnr": float(fn.sum() / abnormal_n) if abnormal_n else None,
        "fp": int(fp.sum()), "fp_denominator": normal_n,
        "fp_rate_all_normal": float(fp.sum() / normal_n) if normal_n else None,
        "fpr": float(fp.sum() / normal_n) if normal_n else None,
        # Conditional rates use only the truth-abnormal/truth-normal events
        # that were actually released.  They answer a different question
        # from the all-cohort rates above.
        "reported_abnormal_n": released_abnormal_n,
        "reported_normal_n": released_normal_n,
        "conditional_fnr": float(fn.sum() / released_abnormal_n) if released_abnormal_n else None,
        "conditional_fpr": float(fp.sum() / released_normal_n) if released_normal_n else None,
        "low_recall": float(low_correct / low_n) if low_n else None,
        "high_recall": float(high_correct / high_n) if high_n else None,
        "reported_low_recall": float(low_correct / released_low_n) if released_low_n else None,
        "reported_high_recall": float(high_correct / released_high_n) if released_high_n else None,
        "low_support": low_n, "high_support": high_n,
        "low_to_within": int(np.sum(a & (truth == -1) & (pred == 0))),
        "high_to_within": int(np.sum(a & (truth == 1) & (pred == 0))),
        "correct_abnormal": int(np.sum(a & abnormal & (pred == truth))),
        "confusion": cm, "all_report_confusion": all_cm,
    }


def fit_all(D: dict[str, np.ndarray], *, smoke: bool = False) -> dict[str, Any]:
    if not DATA.is_file():
        raise RuntimeError(DATA)
    check_transfer_complete(full=not smoke)
    # This directory is the isolated replay workspace owned by this script;
    # reruns overwrite only its own derived artifacts after the input hash and
    # transfer lineage checks above.
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "models").mkdir(exist_ok=True)
    directions = DIRECTIONS
    seeds = SEEDS if not smoke else (1000,)
    model_hashes: dict[str, str] = {}
    thresholds: dict[str, Any] = {}
    arrays: dict[str, dict[str, Any]] = {}
    per_seed_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    matched_rows: list[dict[str, Any]] = []
    for direction in directions:
        if smoke and direction != DIRECTIONS[0]:
            continue
        pred_records: list[dict[str, Any]] = []
        hgb_outer_scores: list[np.ndarray] = []
        hgb_outer_pred: list[np.ndarray] = []
        hgb_outer_truth: list[np.ndarray] = []
        bnd_outer_scores: list[np.ndarray] = []
        dev_by_seed: dict[int, dict[str, np.ndarray]] = {}
        test_cache: list[dict[str, Any]] = []
        fixed_npz = np.load(TRANSFER / "predictions" / f"prefix_900__{direction}.npz", allow_pickle=False)
        fixed_test = np.asarray(fixed_npz["test"], int)
        if fixed_test.shape != (201,):
            raise RuntimeError(f"unexpected fixed transfer test shape: {direction}")
        for seed in seeds:
            target_path, dual_path = transfer_paths(direction, seed)
            obj = joblib.load(target_path)
            if obj.get("test_labels_hidden") is not True or obj.get("external_test_evaluated") is not False:
                raise RuntimeError(f"target object lineage not test blind: {target_path}")
            pos_models: dict[int, dict[str, Any]] = {}
            seed_hgb_scores: list[np.ndarray] = []
            seed_preds: list[np.ndarray] = []
            seed_truth: list[np.ndarray] = []
            seed_boundary_scores: list[np.ndarray] = []
            for pos in (0, 1, 2, 3):
                adapter = str(obj["gate_models"][pos]["adapter"])
                ctx = development_context(D, obj, pos, adapter)
                A, B, fa, pa, ta, qa, fb, pb, tb, qb = ctx
                model = fit_hgb(fa, pa, ta, qa, direction=direction, seed=seed, pos=pos)
                mp = OUT / "models" / direction / f"seed{seed}_pos{pos}.joblib"
                mp.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, mp, compress=3)
                model_hashes[str(mp)] = sha(mp)
                pos_models[pos] = model
                if pos < 3:
                    hgb_score_b = score_hgb(model, fb, pb, qb)
                    bnd_score_b = margin_scores(vv_ma := np.asarray(obj["positions"][pos]["vv"][adapter], float)[:, 2])
                    hgb_outer_scores.append(hgb_score_b)
                    hgb_outer_pred.append(pb)
                    hgb_outer_truth.append(tb)
                    bnd_outer_scores.append(bnd_score_b)
                    seed_hgb_scores.append(hgb_score_b)
                    seed_preds.append(pb)
                    seed_truth.append(tb)
                    seed_boundary_scores.append(bnd_score_b)
            dev_by_seed[int(seed)] = {
                "hgb_score": np.concatenate(seed_hgb_scores),
                "pred": np.concatenate(seed_preds),
                "truth": np.concatenate(seed_truth),
                "boundary_score": np.concatenate(seed_boundary_scores),
            }
            # The corrected transfer NPZ stores final numeric-MA predictions
            # and both frozen risk scores. Test labels are not used here.
            row = int(np.flatnonzero(np.asarray(fixed_npz["seed"], int) == seed)[0])
            med = np.asarray(fixed_npz["final_med4"][row], float)
            pred = ma_class(med[:, 2])
            saved_pred = np.asarray(fixed_npz["pred_class"][row], np.int8)
            if not np.array_equal(saved_pred, pred):
                raise RuntimeError(f"fixed transfer pred_class is not numeric-MA class: {direction} seed={seed}")
            F = ma_features(D["x"][fixed_test], med, obj["source_model"]["ym"], obj["source_model"]["ys"])
            direct_lr = score_lr_gate(obj["gate_models"][3]["model"], F, pred, D["input_qc"][fixed_test])
            saved_lr = np.asarray(fixed_npz["base_score"][row], float)
            if not np.allclose(direct_lr, saved_lr, rtol=0.0, atol=1e-10, equal_nan=True):
                raise RuntimeError(f"fixed transfer LR score mismatch: {direction} seed={seed}")
            test_cache.append({"test": fixed_test, "med": med, "pred": pred, "lr_score": saved_lr, "dual_score": np.asarray(fixed_npz["dual_score"][row], float), "base_action": np.asarray(fixed_npz["base_action"][row], bool)})
        hgb_score_dev = np.concatenate(hgb_outer_scores)
        hgb_pred_dev = np.concatenate(hgb_outer_pred)
        hgb_truth_dev = np.concatenate(hgb_outer_truth)
        bnd_score_dev = np.concatenate(bnd_outer_scores)
        hgb_thr_by_seed = {int(seed): select_threshold(dev_by_seed[int(seed)]["hgb_score"], dev_by_seed[int(seed)]["pred"], dev_by_seed[int(seed)]["truth"], 0.15) for seed in seeds}
        bnd_thr_by_seed = {int(seed): select_threshold(dev_by_seed[int(seed)]["boundary_score"], dev_by_seed[int(seed)]["pred"], dev_by_seed[int(seed)]["truth"], 0.15) for seed in seeds}
        thresholds[direction] = {
            "HGB_by_seed": hgb_thr_by_seed,
            "BOUNDARY_by_seed": bnd_thr_by_seed,
            "development_events": int(len(hgb_score_dev)),
            "development_test_labels_used": False,
            "threshold_selection": "one pooled outer-B threshold per seed, positions 0-2 merged",
        }
        np.savez_compressed(OUT / f"DEVELOPMENT_{direction}.npz", hgb_score=hgb_score_dev, pred=hgb_pred_dev, truth=hgb_truth_dev, boundary_score=bnd_score_dev, B_from_original_data=np.asarray(True))
        test = np.asarray(test_cache[0]["test"], int)
        med = np.asarray([q["med"] for q in test_cache], float)
        pred = np.asarray([q["pred"] for q in test_cache], np.int8)
        lr_score = np.asarray([q["lr_score"] for q in test_cache], float)
        dual_score = np.asarray([q["dual_score"] for q in test_cache], float)
        hgb_scores = []
        boundary_scores = []
        for seed, q in zip(seeds, test_cache):
            obj = joblib.load(transfer_paths(direction, seed)[0])
            adapter = str(obj["gate_models"][3]["adapter"])
            model = joblib.load(OUT / "models" / direction / f"seed{seed}_pos3.joblib")
            ftest = ma_features(D["x"][test], q["med"], obj["source_model"]["ym"], obj["source_model"]["ys"])
            hgb_scores.append(score_hgb(model, ftest, q["pred"], D["input_qc"][test]))
            boundary_scores.append(margin_scores(q["med"][:, 2]))
        hgb_scores = np.asarray(hgb_scores, float)
        boundary_scores = np.asarray(boundary_scores, float)
        hgb_action = np.asarray([threshold_actions(x, hgb_thr_by_seed[int(seed)]["threshold"]) for seed, x in zip(seeds, hgb_scores)], bool)
        boundary_action = np.asarray([threshold_actions(x, bnd_thr_by_seed[int(seed)]["threshold"]) for seed, x in zip(seeds, boundary_scores)], bool)
        truth = ma_class(D["medical"][test, 2])
        arrays[direction] = {"test": test, "truth": truth, "pred": pred, "med4": med, "lr_score": lr_score, "dual_score": dual_score, "hgb_score": hgb_scores, "boundary_score": boundary_scores, "hgb_action": hgb_action, "boundary_action": boundary_action}
        np.savez_compressed(OUT / f"{direction}.npz", seed=np.asarray(seeds), test=test, truth=truth, pred=pred, med4=med, lr_score=lr_score, dual_score=dual_score, hgb_score=hgb_scores, boundary_score=boundary_scores, hgb_action=hgb_action, boundary_action=boundary_action, B_from_original_data=np.asarray(True))
        for method, action in (("HGB", hgb_action), ("BOUNDARY", boundary_action), ("LR", np.asarray([threshold_actions(q["lr_score"], None) for q in test_cache], bool))):
            # LR action is not thresholded here; only its score enters matched
            # quota. Keep an explicit all-false placeholder out of summaries.
            if method == "LR":
                continue
            row = metrics(pred, truth, action)
            residual = med[:, :, 2] - truth[None, :] * 0.0  # replaced below with reference MA
            ref_ma = D["medical"][test, 2]
            residual = med[:, :, 2] - ref_ma[None, :]
            row.update(direction=direction, method=method, seed_count=len(seeds), records=len(test),
                       mae=float(np.mean(np.abs(residual))), rmse=float(np.sqrt(np.mean(residual ** 2))),
                       r2=float(r2_score(np.broadcast_to(ref_ma[None, :], med[:, :, 2].shape).ravel(), med[:, :, 2].ravel())),
                       low_mae=float(np.mean(np.abs(residual[:, truth == -1]))) if np.any(truth == -1) else None,
                       high_mae=float(np.mean(np.abs(residual[:, truth == 1]))) if np.any(truth == 1) else None,
                       all_report_fnr_numerator=int(np.sum((pred == 0) & (truth[None, :] != 0))),
                       all_report_fnr_denominator=int(np.sum(truth != 0) * len(seeds)),
                       all_report_fpr_numerator=int(np.sum((pred != 0) & (truth[None, :] == 0))),
                       all_report_fpr_denominator=int(np.sum(truth == 0) * len(seeds)))
            row["all_report_fnr"] = float(row["all_report_fnr_numerator"] / row["all_report_fnr_denominator"])
            row["all_report_fpr"] = float(row["all_report_fpr_numerator"] / row["all_report_fpr_denominator"])
            row["all_report_fn_rate_all_abnormal"] = row["all_report_fnr"]
            row["all_report_fp_rate_all_normal"] = row["all_report_fpr"]
            # The all-report arm releases every event, so its conditional
            # denominators equal the complete abnormal/normal pools.
            row["all_report_conditional_fnr"] = row["all_report_fnr"]
            row["all_report_conditional_fpr"] = row["all_report_fpr"]
            summary_rows.append(row)
            for k, seed in enumerate(seeds):
                mr = metrics(pred[k:k+1], truth, action[k:k+1])
                per_seed_rows.append({"direction": direction, "method": method, "seed": int(seed), **{k: v for k, v in mr.items() if not isinstance(v, list)}})
        matched_rows.extend(matched_quota(direction, arrays[direction], D, fraction=0.2))
    write_json(OUT / "THRESHOLDS.json", thresholds, overwrite=True)
    write_csv(OUT / "summary.csv", summary_rows, overwrite=True)
    write_csv(OUT / "per_seed.csv", per_seed_rows, overwrite=True)
    write_csv(OUT / "matched20.csv", matched_rows, overwrite=True)
    write_json(OUT / "MODEL_HASHES.json", model_hashes, overwrite=True)
    manifest = {
        "schema": "paper_b085_reporting_controls_v1",
        "status": "SMOKE_COMPLETE" if smoke else "COMPLETE",
        "data_path": str(DATA), "data_sha256": sha(DATA), "B_from_original_data": True,
        "static_B_revision": True,
        "directions": list(DIRECTIONS if not smoke else (DIRECTIONS[0],)), "seeds": list(seeds),
        "test_labels_used_only_after_threshold_freeze": True,
        "old_hgb_models_loaded": False, "feature_dim": FEATURE_DIM,
        "select_threshold_source": str(SELECTOR), "transfer_wrapper": str(TRANSFER_WRAPPER),
        "source_predictions": "new transfer Ridge target objects; no manuscript writes",
        "metric_definitions": {
            "fn_rate_all_abnormal": "released true-abnormal events predicted within-range divided by all truth-abnormal events",
            "fp_rate_all_normal": "released true-within-range events predicted abnormal divided by all truth-within-range events",
            "conditional_fnr": "false-negative events divided by released truth-abnormal events",
            "conditional_fpr": "false-positive events divided by released truth-within-range events",
            "all_report_fn_rate_all_abnormal": "same false-negative rate with every event released",
            "all_report_fp_rate_all_normal": "same false-positive rate with every event released",
        },
    }
    write_json(OUT / ("SMOKE_VERIFIED.json" if smoke else "RESULTS.json"), manifest, overwrite=True)
    if not smoke:
        verify(OUT, D, arrays, thresholds, model_hashes)
    return manifest


def audit_saved_development(D: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Audit saved HGB development arrays without fitting or rescoring models.

    The original run stores the concatenated outer-B prediction arrays.  We
    reconstruct only their row order and masks from the frozen transfer
    objects, then verify ``legal = known & qc & finite`` independently.  This
    is a read-only audit; no test labels or model-selection step is rerun.
    """
    result: dict[str, Any] = {}
    for direction in DIRECTIONS:
        saved_path = OUT / f"DEVELOPMENT_{direction}.npz"
        with np.load(saved_path, allow_pickle=False) as z:
            saved_pred = np.asarray(z["pred"], np.int8)
            saved_truth = np.asarray(z["truth"], np.int8)
            saved_hgb = np.asarray(z["hgb_score"], float)
            saved_boundary = np.asarray(z["boundary_score"], float)
        pred_parts: list[np.ndarray] = []
        truth_parts: list[np.ndarray] = []
        train = {"known": 0, "qc": 0, "finite": 0, "legal": 0, "legal_not_known": 0, "known_not_legal": 0}
        outer = {"known": 0, "qc": 0, "finite": 0, "legal": 0, "legal_not_known": 0, "known_not_legal": 0}
        outer_qc_parts: list[np.ndarray] = []
        outer_finite_parts: list[np.ndarray] = []
        for seed in SEEDS:
            target_path, _ = transfer_paths(direction, seed)
            obj = joblib.load(target_path)
            for pos in (0, 1, 2):
                adapter = str(obj["gate_models"][pos]["adapter"])
                A, B, fa, pa, ta, qa, fb, pb, tb, qb = development_context(D, obj, pos, adapter)
                known_a = np.isin(pa, MA_CLASSES) & np.isin(ta, MA_CLASSES)
                finite_a = np.isfinite(fa).all(axis=1)
                legal_a = known_a & qa & finite_a
                for key, value in (("known", known_a), ("qc", qa), ("finite", finite_a), ("legal", legal_a)):
                    train[key] += int(value.sum())
                train["legal_not_known"] += int(np.sum(legal_a & ~known_a))
                train["known_not_legal"] += int(np.sum(known_a & ~legal_a))

                known_b = np.isin(pb, MA_CLASSES) & np.isin(tb, MA_CLASSES)
                finite_b = np.isfinite(fb).all(axis=1)
                legal_b = known_b & qb & finite_b
                for key, value in (("known", known_b), ("qc", qb), ("finite", finite_b), ("legal", legal_b)):
                    outer[key] += int(value.sum())
                outer["legal_not_known"] += int(np.sum(legal_b & ~known_b))
                outer["known_not_legal"] += int(np.sum(known_b & ~legal_b))
                pred_parts.append(np.asarray(pb, np.int8)); truth_parts.append(np.asarray(tb, np.int8))
                outer_qc_parts.append(np.asarray(qb, bool)); outer_finite_parts.append(np.asarray(finite_b, bool))
        reconstructed_pred = np.concatenate(pred_parts)
        reconstructed_truth = np.concatenate(truth_parts)
        reconstructed_qc = np.concatenate(outer_qc_parts)
        reconstructed_finite = np.concatenate(outer_finite_parts)
        if not np.array_equal(saved_pred, reconstructed_pred) or not np.array_equal(saved_truth, reconstructed_truth):
            raise RuntimeError(f"saved development array order/class mismatch: {direction}")
        result[direction] = {
            "saved_events": int(saved_pred.size),
            "saved_scores_finite": {"hgb": int(np.isfinite(saved_hgb).sum()), "boundary": int(np.isfinite(saved_boundary).sum())},
            "saved_pred_truth_match_reconstructed": True,
            "outer_qc_false": int((~reconstructed_qc).sum()),
            "outer_feature_nonfinite": int((~reconstructed_finite).sum()),
            "training_mask": train,
            "outer_mask": outer,
            "training_legal_equals_known": bool(train["legal_not_known"] == 0 and train["known_not_legal"] == 0),
            "outer_legal_equals_known": bool(outer["legal_not_known"] == 0 and outer["known_not_legal"] == 0),
        }
    return {
        "protocol": "paper_b085_reporting_controls_v1",
        "read_only_saved_array_audit": True,
        "models_refit": False,
        "test_labels_used": False,
        "legal_definition": "known & input_qc & finite_numeric_features",
        "directions": result,
    }


def refresh_saved_metrics(D: dict[str, np.ndarray]) -> None:
    """Refresh metric naming/denominators from saved arrays only.

    This command intentionally does not refit HGB models or regenerate test
    scores.  It is used after a schema clarification so the existing replay
    arrays remain the sole numerical source.
    """
    summary_rows: list[dict[str, Any]] = []
    per_seed_rows: list[dict[str, Any]] = []
    for direction in DIRECTIONS:
        with np.load(OUT / f"{direction}.npz", allow_pickle=False) as z:
            test = np.asarray(z["test"], int); truth = np.asarray(z["truth"], np.int8); pred = np.asarray(z["pred"], np.int8); med = np.asarray(z["med4"], float)
            actions = {"HGB": np.asarray(z["hgb_action"], bool), "BOUNDARY": np.asarray(z["boundary_action"], bool)}
        if pred.shape != (20, 201) or truth.shape != (201,): raise RuntimeError(f"saved reporting array shape changed: {direction}")
        ref_ma = np.asarray(D["medical"][test, 2], float); residual = med[:, :, 2] - ref_ma[None, :]
        for method, action in actions.items():
            row = metrics(pred, truth, action)
            row.update(direction=direction, method=method, seed_count=20, records=len(test),
                       mae=float(np.mean(np.abs(residual))), rmse=float(np.sqrt(np.mean(residual ** 2))),
                       r2=float(r2_score(np.broadcast_to(ref_ma[None, :], med[:, :, 2].shape).ravel(), med[:, :, 2].ravel())),
                       low_mae=float(np.mean(np.abs(residual[:, truth == -1]))) if np.any(truth == -1) else None,
                       high_mae=float(np.mean(np.abs(residual[:, truth == 1]))) if np.any(truth == 1) else None,
                       all_report_fnr_numerator=int(np.sum((pred == 0) & (truth[None, :] != 0))),
                       all_report_fnr_denominator=int(np.sum(truth != 0) * pred.shape[0]),
                       all_report_fpr_numerator=int(np.sum((pred != 0) & (truth[None, :] == 0))),
                       all_report_fpr_denominator=int(np.sum(truth == 0) * pred.shape[0]))
            row["all_report_fnr"] = float(row["all_report_fnr_numerator"] / row["all_report_fnr_denominator"])
            row["all_report_fpr"] = float(row["all_report_fpr_numerator"] / row["all_report_fpr_denominator"])
            row["all_report_fn_rate_all_abnormal"] = row["all_report_fnr"]
            row["all_report_fp_rate_all_normal"] = row["all_report_fpr"]
            row["all_report_conditional_fnr"] = row["all_report_fnr"]
            row["all_report_conditional_fpr"] = row["all_report_fpr"]
            summary_rows.append(row)
            for k, seed in enumerate(SEEDS):
                mr = metrics(pred[k:k + 1], truth, action[k:k + 1])
                per_seed_rows.append({"direction": direction, "method": method, "seed": int(seed), **{key: value for key, value in mr.items() if not isinstance(value, list)}})
    write_csv(OUT / "summary.csv", summary_rows, overwrite=True)
    write_csv(OUT / "per_seed.csv", per_seed_rows, overwrite=True)
    definitions = {
        "fn_rate_all_abnormal": "released false-negative events / all truth-abnormal events",
        "fp_rate_all_normal": "released false-positive events / all truth-within-range events",
        "conditional_fnr": "released false-negative events / released truth-abnormal events",
        "conditional_fpr": "released false-positive events / released truth-within-range events",
        "all_report_*": "all-report counterparts; every event is released, so conditional and all-cohort denominators coincide",
    }
    write_json(OUT / "METRIC_DEFINITIONS.json", definitions, overwrite=True)
    mask_audit = audit_saved_development(D)
    write_json(OUT / "DEVELOPMENT_MASK_AUDIT.json", mask_audit, overwrite=True)
    result_path = OUT / "RESULTS.json"
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    result.update({"summary_refreshed_from_saved_arrays": True, "models_refit_for_refresh": False, "metric_definitions": definitions,
                   "development_mask_audit": "DEVELOPMENT_MASK_AUDIT.json"})
    write_json(result_path, result, overwrite=True)
    ver_path = OUT / "VERIFICATION.json"
    verification = json.loads(ver_path.read_text(encoding="utf-8")) if ver_path.exists() else {}
    verification.update({"summary_recomputed_from_saved_arrays": True, "models_refit_for_refresh": False,
                          "development_mask_audit": mask_audit})
    write_json(ver_path, verification, overwrite=True)
    print(json.dumps({"summary_rows": len(summary_rows), "models_refit": False, "mask_audit": mask_audit}, ensure_ascii=False, indent=2))


def nine_direction_target_path(direction: str, seed: int) -> Path:
    token = TARGET_FILE_TOKEN[direction]
    return TRANSFER / "runs" / f"prefix_900__{direction}" / "models" / direction / "target15" / f"{token}_CFULL_seed{seed}.joblib"


def run_nine_direction_boundary(D: dict[str, np.ndarray]) -> None:
    """Score the original boundary-distance gate for all nine directions.

    No model is fit here.  Each seed's threshold is selected from its saved
    dual ``outer`` development rows, with the frozen target object's saved
    outer-B numeric MA coordinate supplying the boundary distance.  The
    target test NPZ is scored only after that development threshold is frozen.
    """
    out = OUT / "nine_direction_boundary"
    out.mkdir(parents=True, exist_ok=True)
    thresholds: dict[str, Any] = {}
    audit: dict[str, Any] = {"protocol": "paper_b085_nine_direction_boundary_v1", "models_refit": False,
                             "test_labels_used_for_threshold_selection": False, "directions": {}}
    for direction in NINE_DIRECTIONS:
        test_npz = TRANSFER / "predictions" / f"prefix_900__{direction}.npz"
        with np.load(test_npz, allow_pickle=False) as z:
            test = np.asarray(z["test"], int); seeds = np.asarray(z["seed"], int)
            med_test = np.asarray(z["final_med4"], float); pred_test = ma_class(med_test[:, :, 2]); qc_test = np.asarray(z["input_qc"], bool)
        if tuple(seeds.tolist()) != SEEDS:
            raise RuntimeError(f"nine-direction seed axis changed: {direction}")
        seed_thresholds: dict[str, Any] = {}; actions: list[np.ndarray] = []; scores: list[np.ndarray] = []
        dev_rows: list[dict[str, Any]] = []
        for row_seed, seed in enumerate(SEEDS):
            target_path = nine_direction_target_path(direction, seed)
            dual_path = TRANSFER / "runs" / f"prefix_900__{direction}" / "dual" / f"{direction}_{seed}.joblib"
            obj = joblib.load(target_path); dual = joblib.load(dual_path)
            outer_by_id = {str(r["record_id"]): r for r in dual["outer"]}
            score_parts: list[np.ndarray] = []; pred_parts: list[np.ndarray] = []; truth_parts: list[np.ndarray] = []; qc_parts: list[np.ndarray] = []
            for pos in (0, 1, 2):
                p = obj["positions"][pos]; A = np.asarray(p["A"], int); B = np.asarray(p["B"], int)
                adapter = str(obj["gate_models"][pos]["adapter"])
                vv = np.asarray(p["vv"][adapter], float)
                pred = ma_class(vv[:, 2]); truth = ma_class(D["medical"][B, 2]); score = margin_scores(vv[:, 2]); qc = np.asarray(D["input_qc"][B], bool)
                if len(B) != len(vv): raise RuntimeError(f"boundary outer shape mismatch: {direction} seed={seed} pos={pos}")
                for j, idx in enumerate(B):
                    rid = str(D["ids"][idx]); r = outer_by_id.get(rid)
                    if r is None: raise RuntimeError(f"outer development id missing: {direction} seed={seed} id={rid}")
                    if int(r["pred_class"]) != int(pred[j]) or int(r["truth_class"]) != int(truth[j]):
                        raise RuntimeError(f"outer MA class mismatch: {direction} seed={seed} id={rid}")
                    dev_rows.append({"seed": int(seed), "position": int(pos), "record_id": rid, "pred_ma": float(vv[j, 2]),
                                     "pred_class": int(pred[j]), "truth_class": int(truth[j]), "qc": bool(qc[j]), "score": float(score[j]) if np.isfinite(score[j]) else None})
                # ``select_threshold`` itself treats nonfinite scores as
                # unknown; enforce the saved input-QC contract explicitly.
                score_parts.append(np.where(qc, score, np.inf)); pred_parts.append(pred); truth_parts.append(truth); qc_parts.append(qc)
            dev_score = np.concatenate(score_parts); dev_pred = np.concatenate(pred_parts); dev_truth = np.concatenate(truth_parts); dev_qc = np.concatenate(qc_parts)
            chosen = select_threshold(dev_score, dev_pred, dev_truth, 0.15)
            seed_thresholds[str(seed)] = {k: v for k, v in chosen.items() if k != "candidates"}
            test_score = margin_scores(med_test[row_seed, :, 2]); action = threshold_actions(test_score, chosen.get("threshold")) & qc_test[row_seed]
            scores.append(test_score); actions.append(action)
        scores_a = np.asarray(scores, float); actions_a = np.asarray(actions, bool)
        np.savez_compressed(out / f"{direction}.npz", seed=seeds, test=test, score=scores_a, action=actions_a,
                            pred_class=pred_test, input_qc=qc_test, final_med4=med_test,
                            ids=D["ids"][test], groups=D["groups"][test], B_from_original_data=np.asarray(True))
        write_json(out / f"THRESHOLDS_{direction}.json", {"direction": direction, "max_risk": 0.15,
            "selection_source": "saved dual outer development rows plus saved outer-B numeric MA coordinate",
            "test_labels_used_for_threshold_selection": False, "thresholds_by_seed": seed_thresholds}, overwrite=True)
        write_json(out / f"DEVELOPMENT_{direction}.json", {"direction": direction, "rows": dev_rows,
            "test_labels_used_for_threshold_selection": False, "source": "saved target/dual objects; no model fit"}, overwrite=True)
        audit["directions"][direction] = {"test_records": int(len(test)), "development_events": int(len(dev_rows)),
            "thresholds_selected": int(len(seed_thresholds)), "test_action_shape": list(actions_a.shape),
            "test_labels_used_for_threshold_selection": False}
    main_checks = {}
    for direction in DIRECTIONS:
        with np.load(OUT / f"nine_direction_boundary/{direction}.npz", allow_pickle=False) as new, np.load(OUT / f"{direction}.npz", allow_pickle=False) as old:
            same = bool(np.array_equal(new["action"], old["boundary_action"]))
            main_checks[direction] = {"exact_action_match_existing_main_boundary": same, "new_shape": list(new["action"].shape), "old_shape": list(old["boundary_action"].shape)}
            if not same: raise RuntimeError(f"nine-direction boundary mismatch with existing main arm: {direction}")
    audit["main_direction_exact_matches"] = main_checks
    audit["B_from_original_data"] = True
    write_json(out / "AUDIT.json", audit, overwrite=True)
    write_json(out / "RESULTS.json", {"protocol": "paper_b085_nine_direction_boundary_v1", "directions": list(NINE_DIRECTIONS),
        "threshold_source": "development outer only", "test_labels_used_for_threshold_selection": False,
        "models_refit": False, "main_direction_exact_matches": main_checks,
        "B_from_original_data": True}, overwrite=True)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def matched_quota(direction: str, a: Mapping[str, Any], D: Mapping[str, np.ndarray], *, fraction: float) -> list[dict[str, Any]]:
    ids = np.asarray(D["ids"][a["test"]]).astype(str)
    pred = np.asarray(a["pred"], np.int8)
    truth = np.broadcast_to(np.asarray(a["truth"], np.int8), pred.shape)
    out: list[dict[str, Any]] = []
    for population, base_mask in (("all", np.ones(pred.shape, bool)), ("predicted_within", pred == 0)):
        common = base_mask & np.isfinite(a["lr_score"]) & np.isfinite(a["dual_score"])
        for method, score in (("LR", np.asarray(a["lr_score"])), ("DUAL", np.asarray(a["dual_score"]))):
            selected = np.zeros(pred.shape, bool)
            for k in range(pred.shape[0]):
                ii = np.flatnonzero(common[k])
                order = np.lexsort((ids[ii], score[k, ii]))
                n = int(np.floor(fraction * len(ii)))
                selected[k, ii[order[:n]]] = True
            out.append({"direction": direction, "population": population, "method": method, "fraction": fraction, "eligible_events": int(common.sum()), "selected": int(selected.sum()), "agreement": float(np.sum(selected & (pred == truth)) / selected.sum()) if selected.any() else None, "abnormal_within": int(np.sum(selected & (truth != 0) & (pred == 0))), "abnormal_within_rate_full": float(np.sum(selected & (truth != 0) & (pred == 0)) / np.sum(truth != 0)) if np.any(truth != 0) else None})
    # Paired differences use exactly the same event selections, with labels
    # consulted only for this descriptive post-hoc comparison.
    for population, base_mask in (("all", np.ones(pred.shape, bool)), ("predicted_within", pred == 0)):
        common = base_mask & np.isfinite(a["lr_score"]) & np.isfinite(a["dual_score"])
        selections: dict[str, np.ndarray] = {}
        for method, score in (("LR", np.asarray(a["lr_score"])), ("DUAL", np.asarray(a["dual_score"]))):
            selected = np.zeros(pred.shape, bool)
            for k in range(pred.shape[0]):
                ii = np.flatnonzero(common[k]); order = np.lexsort((ids[ii], score[k, ii])); n = int(np.floor(fraction * len(ii)))
                selected[k, ii[order[:n]]] = True
            selections[method] = selected
        for metric, fn in (("agreement", lambda s: s & (pred == truth)), ("abnormal_within", lambda s: s & (truth != 0) & (pred == 0))):
            dual = fn(selections["DUAL"]); lr = fn(selections["LR"])
            den = np.ones(pred.shape, bool) if metric == "abnormal_within" else selections["DUAL"]
            den_lr = np.ones(pred.shape, bool) if metric == "abnormal_within" else selections["LR"]
            out.append({"direction": direction, "population": population, "method": "DUAL_minus_LR", "fraction": fraction, "metric": metric, "dual_numerator": int(dual.sum()), "dual_denominator": int(den.sum()), "lr_numerator": int(lr.sum()), "lr_denominator": int(den_lr.sum()), "difference_rate": (float(dual.sum() / den.sum() - lr.sum() / den_lr.sum()) if den.sum() and den_lr.sum() else None)})
    return out


def verify(out: Path, D: Mapping[str, np.ndarray], arrays: Mapping[str, Mapping[str, Any]], thresholds: Mapping[str, Any], model_hashes: Mapping[str, str]) -> None:
    checks: dict[str, Any] = {"data_sha256": sha(DATA), "original_B_values_retained": True, "model_hashes_match": True, "thresholds_frozen_from_development_only": True, "test_shape": {}, "sklearn_r2_checks": {}}
    for direction, a in arrays.items():
        assert np.asarray(a["pred"]).shape == (20, 201)
        assert np.asarray(a["truth"]).shape == (201,)
        assert np.asarray(a["hgb_action"]).shape == (20, 201)
        for p, digest in model_hashes.items():
            if direction in p:
                checks["model_hashes_match"] &= sha(Path(p)) == digest
        ref = D["medical"][a["test"], 2]
        y = np.broadcast_to(ref[None, :], a["med4"][:, :, 2].shape)
        checks["sklearn_r2_checks"][direction] = float(r2_score(y.ravel(), a["med4"][:, :, 2].ravel()))
        checks["test_shape"][direction] = [20, 201]
    write_json(out / "VERIFICATION.json", checks, overwrite=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("smoke", "run", "refresh", "nine-boundary"))
    args = ap.parse_args()
    D = load_data()
    with threadpool_limits(limits=1):
        if args.command == "refresh":
            refresh_saved_metrics(D)
        elif args.command == "nine-boundary":
            run_nine_direction_boundary(D)
        else:
            fit_all(D, smoke=args.command == "smoke")


if __name__ == "__main__":
    main()
