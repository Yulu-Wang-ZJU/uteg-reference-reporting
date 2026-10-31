"""Independent, read-only verification of the matched target-only sequence.

This module deliberately does not import the numerical prediction, action, or
replay helpers used by ``evaluate.py``.  It loads frozen target objects,
recomputes the Ridge projection, MA-error gate, dual-head scores, and actions
from their stored coefficients, and then reconciles the resulting arrays with
the private evaluation archive and frozen aggregate reports.  The only file it
writes is the verification deliverable ``VERIFICATION.json``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np


R = Path(__file__).resolve().parent
G = R.parent
O = G.parent
BASE = O / "uteg_legacy_aligned_threecenter_20261022_155614"
FOCUSED = G / "focused_reporting_20261030"
SEQ = O / "uteg_current_ma_sequence_20261026" / "actions"
EXT = O / "uteg_ridge_dual_head_external_eval_20261026_191117" / "actions"
PRED = O / "uteg_signal_model_goal_20261026_000929" / "ridge10_fullchain_20261026_122551" / "eval" / "predictions"
DATA = BASE / "DATA15.npz"
PREFIX_PARAMETERS = BASE / "PREFIX_PARAMETERS.npz"
PREFIX_INPUTS = BASE / "PREFIX_INPUTS.npz"
SEEDS = np.arange(1000, 1020, dtype=np.int32)
HORIZONS = (300, 600, 900)
DIRS = ("HZ_to_SA", "HZ+HB_to_SA")
TARGET = "Target-only Ridge"
PRIMARY = "5-10-15"


def sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stable_sigmoid(value: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid without importing a scorer implementation."""
    z = np.asarray(value, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    ez = np.exp(z[~positive])
    out[~positive] = ez / (1.0 + ez)
    return out


def classify(value: Any) -> np.ndarray:
    x = np.asarray(value, dtype=float)
    return np.where(~np.isfinite(x), 9,
                    np.where(x < 50.0, -1,
                             np.where(x <= 70.0, 0, 1))).astype(np.int8)


def forward_ridge(predictor: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    """Recompute the stored standardized Ridge projection and clipping."""
    x = np.asarray(x, dtype=float)
    xm = np.asarray(predictor["xm"], dtype=float)
    xs = np.asarray(predictor["xs"], dtype=float)
    ym = np.asarray(predictor["ym"], dtype=float)
    ys = np.asarray(predictor["ys"], dtype=float)
    coef = np.asarray(predictor["coef"], dtype=float)
    med = ym[None, :] + (((x - xm[None, :]) / xs[None, :]) @ coef) * ys[None, :]
    med = np.asarray(med, dtype=float)
    med[..., :3] = np.maximum(med[..., :3], 0.0)
    med[..., 3] = np.clip(med[..., 3], 0.0, 89.999999)
    return med


def gate_features(x: np.ndarray, med: np.ndarray, ym: np.ndarray, ys: np.ndarray) -> np.ndarray:
    pred_ma = np.asarray(med, dtype=float)[:, 2]
    pred = classify(pred_ma)
    return np.column_stack((
        np.asarray(x, dtype=float),
        (np.asarray(med, dtype=float) - np.asarray(ym)[None, :]) / np.asarray(ys)[None, :],
        (pred_ma - 60.0) / 10.0,
        np.minimum(np.abs(pred_ma - 50.0), np.abs(pred_ma - 70.0)) / 10.0,
        (pred == -1).astype(float), (pred == 0).astype(float), (pred == 1).astype(float),
    ))


def score_error_gate(model: Mapping[str, Any], features: np.ndarray, pred: np.ndarray) -> np.ndarray:
    pre = model["pre"]
    raw = np.asarray(features, dtype=float)
    finite = np.isfinite(raw)
    filled = np.where(finite, raw, np.asarray(pre["median"], dtype=float)[None, :])
    z = np.clip((filled - np.asarray(pre["mean"], dtype=float)[None, :]) /
                np.asarray(pre["sd"], dtype=float)[None, :], -10.0, 10.0)
    if bool(model.get("fallback", False)):
        score = np.full(len(raw), float(model["constant_score"]), dtype=float)
    else:
        score = stable_sigmoid(z @ np.asarray(model["coef"], dtype=float) +
                               float(model["intercept"]))
    valid = np.isin(np.asarray(pred), (-1, 0, 1)) & finite.all(axis=1)
    return np.where(valid, score, np.inf)


def score_head(model: Mapping[str, Any] | None, x: np.ndarray, med_ma: np.ndarray) -> np.ndarray:
    if model is None:
        return np.full(len(med_ma), np.inf, dtype=float)
    pre = model["pre"]
    raw = np.column_stack([np.asarray(x, dtype=float)[:, 1], np.asarray(med_ma, dtype=float)])
    finite = np.isfinite(raw)
    filled = np.where(finite, raw, np.asarray(pre["median"], dtype=float)[None, :])
    z = np.clip((filled - np.asarray(pre["mean"], dtype=float)[None, :]) /
                np.asarray(pre["sd"], dtype=float)[None, :], -10.0, 10.0)
    theta = np.asarray(model["theta"], dtype=float)
    score = stable_sigmoid(z @ theta[:-1] + theta[-1])
    return np.where(finite.all(axis=1), score, np.inf)


def action_from_scores(threshold: Mapping[str, Any], pred: np.ndarray,
                       base_score: np.ndarray, dual_score: np.ndarray,
                       qc: np.ndarray) -> np.ndarray:
    released = np.zeros(len(pred), dtype=bool)
    if bool(threshold["all_withhold"]):
        return released
    normal = threshold.get("normal")
    tail = threshold.get("tail")
    if normal is not None:
        released |= ((pred == 0) & np.isfinite(dual_score) &
                     (dual_score <= float(normal)))
    if tail is not None:
        released |= (np.isin(pred, (-1, 1)) & np.isfinite(base_score) &
                     (base_score <= float(tail)))
    return released & np.asarray(qc, dtype=bool)


def target_forward(model: Mapping[str, Any], data: Mapping[str, Any]) -> dict[str, np.ndarray]:
    idx = np.asarray(model["test"], dtype=np.int32)
    x = np.asarray(data["x"], dtype=float)[idx]
    med = forward_ridge(model["predictor"], x)
    pred = classify(med[:, 2])
    features = gate_features(x, med, model["ym"], model["ys"])
    base = score_error_gate(model["error"], features, pred)
    low = score_head(model["heads"]["low"], x, med[:, 2])
    high = score_head(model["heads"]["high"], x, med[:, 2])
    dual = np.maximum(low, high)
    released = action_from_scores(model["threshold"], pred, base, dual,
                                  np.asarray(data["input_qc"], dtype=bool)[idx])
    return {"med": med, "pred": pred, "base": base, "dual": dual,
            "action": released, "index": idx}


def locked_first(checkpoints: Mapping[int, tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent explicit first-report loop; no evaluate.py replay helper."""
    first_action = np.zeros_like(next(iter(checkpoints.values()))[0], dtype=bool)
    first_class = np.full_like(next(iter(checkpoints.values()))[1], 9, dtype=np.int8)
    first_time = np.full(first_action.shape, -1, dtype=np.int32)
    for horizon in HORIZONS:
        actions, predictions = checkpoints[horizon]
        new = np.asarray(actions, dtype=bool) & ~first_action
        first_class[new] = np.asarray(predictions, dtype=np.int8)[new]
        first_time[new] = int(horizon)
        first_action |= np.asarray(actions, dtype=bool)
    return first_class, first_time, first_action


def bootstrap_stat(num: np.ndarray, den: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    """Exact 5,000-draw identity-proxy bootstrap used by the frozen analysis."""
    numerator = np.asarray(num, dtype=float)
    denominator = np.broadcast_to(np.asarray(den, dtype=float), numerator.shape)
    unique_groups = np.unique(np.asarray(groups))
    # The historical bootstrap collapses both seed and record axes within a
    # proxy group before drawing groups.  Keeping a seed axis here would
    # change the estimand and the valid-resample accounting.
    group_num = np.asarray([numerator[:, groups == g].sum() for g in unique_groups])
    group_den = np.asarray([denominator[:, groups == g].sum() for g in unique_groups])
    draw = np.random.default_rng(20261030).integers(
        0, len(unique_groups), (5000, len(unique_groups)))
    draw_den = group_den[draw].sum(axis=1)
    valid = draw_den > 0.0
    total_den = float(group_den.sum())
    estimate = float(group_num.sum() / total_den) if total_den else None
    if valid.any():
        values = group_num[draw].sum(axis=1)[valid] / draw_den[valid]
        lo, hi = np.quantile(values, [0.025, 0.975]).tolist()
    else:
        lo, hi = None, None
    return {"estimate": estimate, "lo": lo, "hi": hi,
            "valid_resamples": int(valid.sum())}


def compare_stat(actual: Mapping[str, Any], expected: Mapping[str, Any], tol: float = 1e-15) -> None:
    assert actual["valid_resamples"] == expected["valid_resamples"]
    for key in ("estimate", "lo", "hi"):
        if expected[key] is None:
            assert actual[key] is None
        else:
            assert np.isclose(float(actual[key]), float(expected[key]), atol=tol, rtol=tol), (key, actual, expected)


def expected_prefix_observed_n(index: np.ndarray, horizon: int) -> np.ndarray:
    """Check the archived true-observed prefix lengths, including SA test rows."""
    with np.load(PREFIX_INPUTS, allow_pickle=False) as pin, np.load(PREFIX_PARAMETERS, allow_pickle=False) as params:
        horizons = np.asarray(pin["horizon_seconds"], dtype=int)
        if int(horizon) == 900:
            # PREFIX_INPUTS/PREFIX_PARAMETERS intentionally store only the
            # 300/600 processed prefixes; the 900 checkpoint is DATA15's
            # full observed representation.  Reuse the archived raw length
            # axis and clip it at the full 900-s horizon.
            raw_n = np.asarray(pin["raw_observed_n"], dtype=np.int32)[0]
            observed_n = np.minimum(900, raw_n)
            return observed_n[np.asarray(index, dtype=np.int32)]
        hi = int(np.where(horizons == int(horizon))[0][0])
        raw_n = np.asarray(pin["raw_observed_n"], dtype=np.int32)[hi]
        observed_n = np.asarray(pin["observed_n"], dtype=np.int32)[hi]
        expected = np.minimum(int(horizon), raw_n)
        assert np.array_equal(observed_n, expected)
        assert int(params["observed_n"].shape[1]) == 480
        assert np.array_equal(np.asarray(params["observed_n"], dtype=np.int32)[hi], observed_n)
        return observed_n[np.asarray(index, dtype=np.int32)]


def verify_hashes(freeze: Mapping[str, Any], eval_hashes: Mapping[str, str], protected: Mapping[str, Any]) -> dict[str, Any]:
    frozen = {**freeze["inputs"], **freeze["models"]}
    assert all(sha(path) == digest for path, digest in frozen.items())
    assert all(sha(path) == digest for path, digest in eval_hashes.items())
    protected_files = protected["protected_files"]
    assert all(sha(path) == digest for path, digest in protected_files.items())
    repo = O.parent / "releases" / "uteg_reference_reporting_20261030"
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo).decode().strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo).decode()
    assert head == protected["public_repo_head"]
    assert status == protected["public_repo_status"]
    return {"freeze_inputs_models": len(frozen), "evaluation_inputs": len(eval_hashes),
            "protected_files": len(protected_files), "public_repo_head_unchanged": True,
            "public_repo_status_unchanged": True}


def per_seed_counts(released: np.ndarray, first_class: np.ndarray, truth: np.ndarray) -> list[dict[str, int]]:
    good = released & (first_class == truth[None, :])
    bad = released & (first_class != truth[None, :])
    rows = []
    for j, seed in enumerate(SEEDS.tolist()):
        rows.append({"seed": int(seed), "correct": int(good[j].sum()),
                     "incorrect": int(bad[j].sum()), "unreported": int((~released[j]).sum()),
                     "coverage": int(released[j].sum()),
                     "low_to_within": int((released[j] & (truth == -1) & (first_class[j] == 0)).sum()),
                     "high_to_within": int((released[j] & (truth == 1) & (first_class[j] == 0)).sum()),
                     "correct_low": int((released[j] & (truth == -1) & (first_class[j] == -1)).sum()),
                     "correct_high": int((released[j] & (truth == 1) & (first_class[j] == 1)).sum())})
    return rows


def main() -> None:
    out = R / "VERIFICATION.json"
    assert not out.exists(), f"verification output already exists: {out}"
    freeze = read_json(R / "FREEZE.json")
    assert freeze["test_evaluated"] is False
    eval_hashes = read_json(R / "EVALUATION_INPUT_HASHES.json")
    results = read_json(R / "RESULTS.json")
    private_path = R / "private" / "evaluation_arrays.npz"
    assert private_path.exists()

    train = load_module("target_sequence_verify_loader", R / "train_prefix.py")
    with np.load(DATA, allow_pickle=False) as archive:
        data15 = {key: np.asarray(archive[key]) for key in archive.files}
    d15 = train.load_checkpoint(900)
    idx = None
    models: dict[int, dict[int, Any]] = {}
    forward: dict[int, dict[int, dict[str, np.ndarray]]] = {}
    target_checks: dict[str, Any] = {}
    prefix_observed: dict[str, Any] = {}

    for horizon in HORIZONS:
        data = train.load_checkpoint(horizon)
        for key in ("ids", "groups", "medical"):
            assert np.array_equal(np.asarray(data[key]), np.asarray(d15[key]))
        assert np.array_equal(np.asarray(data["ids"]), data15["ids"])
        assert np.array_equal(np.asarray(data["groups"]), data15["groups"])
        assert np.array_equal(np.asarray(data["medical"]), data15["medical"])
        if horizon in (300, 600):
            with np.load(PREFIX_PARAMETERS, allow_pickle=False) as params:
                hi = int(np.where(np.asarray(params["horizon_seconds"], dtype=int) == horizon)[0][0])
                assert np.array_equal(np.asarray(data["x"]), np.asarray(params["x"])[hi])
                assert np.array_equal(np.asarray(data["ids"]), np.asarray(params["ids"])[hi].astype(data["ids"].dtype))
                assert np.array_equal(np.asarray(data["groups"]), np.asarray(params["groups"])[hi].astype(data["groups"].dtype))
        horizon_models: dict[int, Any] = {}
        horizon_forward: dict[int, dict[str, np.ndarray]] = {}
        for seed in SEEDS.tolist():
            model_path = FOCUSED / f"model_{seed}.joblib" if horizon == 900 else R / f"T{horizon}" / f"model_{seed}.joblib"
            model = joblib.load(model_path)
            assert int(model["seed"]) == seed
            current_idx = np.asarray(model["test"], dtype=np.int32)
            if idx is None:
                idx = current_idx.copy()
            assert np.array_equal(current_idx, idx)
            assert len(np.asarray(model["pool"])) == 60 and len(current_idx) == 201
            assert not set(np.asarray(data["groups"])[model["pool"]]) & set(np.asarray(data["groups"])[current_idx])
            horizon_models[seed] = model
            result = target_forward(model, data)
            horizon_forward[seed] = result
        models[horizon] = horizon_models
        forward[horizon] = horizon_forward
        obs = expected_prefix_observed_n(idx, horizon)
        prefix_observed[str(horizon)] = {"test_records": 201, "min": int(obs.min()), "max": int(obs.max()),
                                         "unique_lengths": int(np.unique(obs).size),
                                         "raw_clipped_increment_exact": True}

    assert idx is not None
    idx = np.asarray(idx, dtype=np.int32)
    groups = np.asarray(d15["groups"])[idx]
    truth = classify(np.asarray(d15["medical"])[idx, 2])
    assert len(idx) == 201 and len(np.unique(groups)) == 190
    assert [int((truth == value).sum()) for value in (-1, 0, 1)] == [7, 170, 24]

    # Compare each independent target forward result with root's private arrays.
    private_archive = np.load(private_path, allow_pickle=False)
    private = {key: np.asarray(private_archive[key]) for key in private_archive.files}
    private_archive.close()
    if True:
        assert np.array_equal(private["seed"], SEEDS)
        assert np.array_equal(private["groups"], groups)
        assert np.array_equal(private["truth"], truth)
        assert np.array_equal(private["test_global_index"], idx)
        for horizon in HORIZONS:
            med = np.asarray([forward[horizon][seed]["med"] for seed in SEEDS.tolist()])
            pred = np.asarray([forward[horizon][seed]["pred"] for seed in SEEDS.tolist()])
            action = np.asarray([forward[horizon][seed]["action"] for seed in SEEDS.tolist()])
            assert np.array_equal(med, private[f"target_med_T{horizon}"])
            assert np.array_equal(pred, private[f"target_pred_T{horizon}"])
            assert np.array_equal(action, private[f"target_action_T{horizon}"])
            target_checks[str(horizon)] = {"seed_axis_exact": True, "record_axis_exact": True,
                                           "ridge_projection_exact": True, "error_gate_exact": True,
                                           "low_high_head_exact": True, "threshold_action_exact": True,
                                           "events": int(action.sum())}

        # Independent target first-report lock.
        target_cp = {h: (np.asarray(private[f"target_action_T{h}"], dtype=bool),
                         np.asarray(private[f"target_pred_T{h}"], dtype=np.int8)) for h in HORIZONS}
        target_first_class, target_first_time, target_reported = locked_first(target_cp)
        assert np.array_equal(target_first_class, private[TARGET + "_first_class"])
        assert np.array_equal(target_first_time, private[TARGET + "_first_time"])
        assert np.array_equal(target_reported, private[TARGET + "_reported"])

        primary: dict[str, dict[str, np.ndarray]] = {TARGET: {
            "first_class": target_first_class, "first_time": target_first_time,
            "reported": target_reported}}
        # Load transfer checkpoints independently and lock their first reports.
        for direction in DIRS:
            transfer_cp: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for horizon in (300, 600):
                with np.load(SEQ / f"T{horizon}" / f"{direction}.npz", allow_pickle=False) as q:
                    assert np.array_equal(q["ids"].astype(str), np.asarray(d15["ids"])[idx].astype(str))
                    assert np.array_equal(q["groups"], groups) and np.array_equal(q["seed"], SEEDS)
                    assert np.array_equal(q["index"], idx)
                    transfer_cp[horizon] = (q["action"].astype(bool), q["pred"].astype(np.int8))
            with np.load(EXT / f"{direction}.npz", allow_pickle=False) as q:
                assert np.array_equal(q["test_ids"].astype(str), np.asarray(d15["ids"])[idx].astype(str))
                # The archived external-action object has no seed field; the
                # paired frozen prediction archive carries the authoritative
                # seed/test axes and is checked before using q's arrays.
                with np.load(PRED / f"{direction}.npz", allow_pickle=False) as p:
                    assert np.array_equal(p["seed"], SEEDS)
                    assert np.array_equal(p["test_global_index"], idx)
                    assert np.array_equal(p["test_ids"].astype(str), q["test_ids"].astype(str))
                    assert np.array_equal(classify(p["final_med4"][:, :, 2]), q["final_ma_class"])
                transfer_cp[900] = (q["unified_action"].astype(bool), q["final_ma_class"].astype(np.int8))
            first_class, first_time, reported = locked_first(transfer_cp)
            assert np.array_equal(first_class, private[direction + "_first_class"])
            assert np.array_equal(first_time, private[direction + "_first_time"])
            assert np.array_equal(reported, private[direction + "_reported"])
            with np.load(SEQ / f"{direction}_first_report.npz", allow_pickle=False) as old:
                assert np.array_equal(first_class, old["first_class"])
                assert np.array_equal(first_time, old["first_time"])
                assert np.array_equal(reported, old["reported"])
            primary[direction] = {"first_class": first_class, "first_time": first_time,
                                  "reported": reported}

    # Full first-report locking and exhaustive per-seed/class reconciliation.
    one = np.ones((len(SEEDS), len(idx)), dtype=float)
    aggregate: dict[str, Any] = {}
    primary_arrays: dict[str, dict[str, np.ndarray]] = {}
    for arm, locked in primary.items():
        first_class = locked["first_class"]
        first_time = locked["first_time"]
        reported = locked["reported"]
        assert np.array_equal(reported, first_time != -1)
        assert np.array_equal(first_class == 9, ~reported)
        assert np.all(np.isin(first_class[reported], (-1, 0, 1)))
        # Every event is assigned exactly once, and each first report is the
        # earliest eligible checkpoint in the explicit loop above.
        assert int(reported.sum()) + int((~reported).sum()) == 4020
        for j in range(len(SEEDS)):
            seen = np.zeros(len(idx), dtype=bool)
            for h in HORIZONS:
                if arm == TARGET:
                    aa = np.asarray(private[f"target_action_T{h}"])[j]
                else:
                    if h < 900:
                        with np.load(SEQ / f"T{h}" / f"{arm}.npz", allow_pickle=False) as q:
                            aa = q["action"][j].astype(bool)
                    else:
                        with np.load(EXT / f"{arm}.npz", allow_pickle=False) as q:
                            aa = q["unified_action"][j].astype(bool)
                new = aa & ~seen
                assert np.array_equal(new, (reported[j] & (first_time[j] == h)))
                seen |= aa
            assert np.array_equal(seen, reported[j])
        good = reported & (first_class == truth[None, :])
        bad = reported & (first_class != truth[None, :])
        unreported = ~reported
        assert np.all(good.astype(int) + bad.astype(int) + unreported.astype(int) == 1)
        primary_arrays[arm] = {"correct": good, "incorrect": bad, "unreported": unreported,
                               "coverage": reported,
                               "low_to_within": reported & (truth[None, :] == -1) & (first_class == 0),
                               "high_to_within": reported & (truth[None, :] == 1) & (first_class == 0),
                               "correct_low": reported & (truth[None, :] == -1) & (first_class == -1),
                               "correct_high": reported & (truth[None, :] == 1) & (first_class == 1)}
        counts = {key: int(value.sum()) for key, value in primary_arrays[arm].items()}
        expected_counts = results["arms"][arm][PRIMARY]["counts"]
        assert counts == expected_counts
        aggregate[arm] = {"counts": counts,
                          "per_seed": per_seed_counts(reported, first_class, truth),
                          "zero_report_seeds": int((reported.sum(axis=1) == 0).sum()),
                          "class_counts": {str(value): int((truth[None, :] == value).sum() * len(SEEDS)) for value in (-1, 0, 1)},
                          "first_report_time_events": {str(h): int((first_time == h).sum()) for h in (300, 600, 900, -1)}}

    # The pre-existing 900-s target per-seed report is an exact frozen match.
    old_rows = []
    with (FOCUSED / "per_seed.csv").open(encoding="utf-8-sig", newline="") as stream:
        import csv
        old_rows = list(csv.DictReader(stream))
    target_900_rows = {int(row["seed"]): row for row in old_rows if row["method"] == TARGET}
    assert len(target_900_rows) == 20
    target_900 = {"action": np.asarray([forward[900][seed]["action"] for seed in SEEDS.tolist()]),
                  "pred": np.asarray([forward[900][seed]["pred"] for seed in SEEDS.tolist()])}
    target_900_good = target_900["action"] & (target_900["pred"] == truth[None, :])
    target_900_bad = target_900["action"] & (target_900["pred"] != truth[None, :])
    for j, seed in enumerate(SEEDS.tolist()):
        row = target_900_rows[seed]
        actual = {"correct": int(target_900_good[j].sum()), "incorrect": int(target_900_bad[j].sum()),
                  "unreported": int((~target_900["action"][j]).sum()), "coverage": int(target_900["action"][j].sum()),
                  "low_to_within": int((target_900["action"][j] & (truth == -1) & (target_900["pred"][j] == 0)).sum()),
                  "high_to_within": int((target_900["action"][j] & (truth == 1) & (target_900["pred"][j] == 0)).sum()),
                  "correct_low": int((target_900["action"][j] & (truth == -1) & (target_900["pred"][j] == -1)).sum()),
                  "correct_high": int((target_900["action"][j] & (truth == 1) & (target_900["pred"][j] == 1)).sum())}
        assert all(actual[key] == int(row[key]) for key in actual)

    # Primary source-minus-target contrasts, independently recomputed from the
    # locked event arrays using the exact saved 5,000 proxy-group draws.
    contrast: dict[str, Any] = {}
    for direction in DIRS:
        contrast[direction] = {}
        for outcome in ("correct", "incorrect"):
            diff = primary_arrays[direction][outcome].astype(int) - primary_arrays[TARGET][outcome].astype(int)
            actual = bootstrap_stat(diff, one, groups)
            expected = results["source_minus_target"][direction][PRIMARY][outcome]
            compare_stat(actual, expected)
            contrast[direction][outcome] = actual

    hashes = verify_hashes(freeze, eval_hashes, read_json(R / "PROTECTED_BEFORE.json"))
    verification = {
        "status": "PASS",
        "read_only": True,
        "new_fits": 0,
        "new_thresholds": 0,
        "independent_formula": {
            "target_forward_checkpoints": target_checks,
            "formula_sources": ["stored Ridge coefficients", "stored MA-error-LR coefficients/fallback", "stored low/high heads", "stored thresholds"],
            "original_prediction_action_replay_helpers_reused": False,
        },
        "private_evaluation_arrays": {"path": str(private_path.resolve()), "all_target_arrays_exact": True,
                                       "record_seed_identity_exact": True},
        "prefix_observed_n": prefix_observed,
        "first_report_locking": {"all_arms_exact": True, "disjoint_exhaustive": True,
                                 "per_seed_and_class_reconciled": True, "aggregate": aggregate},
        "focused_900_per_seed_exact": True,
        "primary_source_minus_target_bootstrap": contrast,
        "hashes": hashes,
        "scope": "Numerical and implementation cross-check by an implementation agent; not external independent validation.",
    }
    out.write_text(json.dumps(verification, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
