"""Independent evaluator for the frozen Ridge-10 numeric-MA actions.

The default/preflight path is label blind.  It verifies the model, threshold,
prediction and baseline identities without opening ``DATA15.npz`` medical/CI
arrays.  The approved one-shot path is deliberately separate and is the only
path that may read target MA references.  No fitting, threshold search,
candidate selection, B/WITHIN action, or test-side model choice is present.

This file is intentionally independent of the historical evaluator.  It uses
only the already frozen ``eval/predictions/*.npz`` actions and, after explicit
approval, recomputes metrics from those saved actions and the target references.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import runtime as r  # establishes all task-local cache/TMP locations first
import numpy as np


ROOT = r.ROOT
BASE_DATA = r.BASE / "DATA15.npz"
ROLE_PATH = r.BASE / "ROLE_FREEZE.json"
OLD_RUN = r.NINE
OLD_PRED = OLD_RUN / "eval" / "predictions"
PRED_MANIFEST = ROOT / "PREDICTIONS_ACTIONS_FROZEN.json"
MODEL_MANIFEST = ROOT / "FULL_MODELS_FROZEN.json"
THRESHOLD_MANIFEST = ROOT / "THRESHOLD_FREEZE.json"
EVALUATION_PROTOCOL = ROOT / "EVALUATION_PROTOCOL.md"
PREFLIGHT_REQUEST = ROOT / "FINAL_EVALUATION_PREFLIGHT_REQUEST.json"
RUNNER_CODE = ROOT / "run_ridge10_fullchain.py"
RUNTIME_CODE = ROOT / "runtime.py"
PROTOCOL_PATH = ROOT / "PROTOCOL.md"
RUNNER_FREEZE = ROOT / "RUNNER_FREEZE.json"
STATE_PATH = ROOT / "RIDGE10_FULLCHAIN_STATE.json"
CURRENT_LEDGER = ROOT / "CURRENT_FIT_LEDGER.json"
ROUND_LEDGER = ROOT / "ROUND_START_LEDGER.json"
THRESHOLD_CODE = ROOT / "freeze_ma_thresholds.py"
PREDICTION_SCHEMA = ROOT / "PREDICTION_ACTION_SCHEMA.json"
SOURCE_CONTEXTS = ROOT.parent / "source_revision" / "SOURCE_REVISION_CONTEXTS.csv"
SEEDS = tuple(range(1000, 1020))
MA_CLASSES = (-1, 0, 1)
TARGETS = ("Huzhou", "SAHZU", "Hubei")
EXPECTED_TEST_N = {"Huzhou": 55, "SAHZU": 201, "Hubei": 41}
TARGET_BY_DIRECTION = {
    "HZ+HB_to_SA": "SAHZU", "SA+HB_to_HZ": "Huzhou", "SA+HZ_to_HB": "Hubei",
    "HZ_to_SA": "SAHZU", "HB_to_SA": "SAHZU", "SA_to_HZ": "Huzhou",
    "HB_to_HZ": "Huzhou", "SA_to_HB": "Hubei", "HZ_to_HB": "Hubei",
}
DIRECTIONS = tuple(TARGET_BY_DIRECTION)
DUAL_DIRECTIONS = ("HZ+HB_to_SA", "SA+HB_to_HZ", "SA+HZ_to_HB")
SINGLE_DIRECTIONS = tuple(direction for direction in DIRECTIONS if direction not in DUAL_DIRECTIONS)
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20_261_026


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(safe(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def ma_class(value: Any) -> np.ndarray:
    x = np.asarray(value, dtype=float)
    return np.where(~np.isfinite(x), 9, np.where(x < 50, -1, np.where(x <= 70, 0, 1))).astype(np.int8)


def _scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def _record_flow(mask: np.ndarray, truth: np.ndarray, ids: np.ndarray,
                 source_class: int | None = None) -> dict[str, Any]:
    """Count record-seed events and unique record IDs separately."""
    mask = np.asarray(mask, dtype=bool)
    if source_class is None:
        denominator = int(len(truth) * len(SEEDS))
        support = None
    else:
        support = int(np.sum(truth == source_class))
        denominator = int(support * len(SEEDS))
    event_n = int(mask.sum())
    affected = np.unique(ids[np.any(mask, axis=0)]) if mask.ndim == 2 and mask.shape[1] else np.asarray([], dtype=str)
    return {
        "event_n_record_seed": event_n,
        "denominator_record_seed": denominator,
        "unique_affected_records": int(len(affected)),
        "event_rate": (float(event_n / denominator) if denominator else None),
        "support_records": support,
        "support_record_seed": (int(denominator) if source_class is not None else None),
        "support_status": ("NA_no_tail_support" if source_class is not None and support == 0 else "SUPPORTED"),
    }


def _align_baseline(current_ids: np.ndarray, old: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Align the immutable old baseline by record ID, never by position."""
    old_ids = np.asarray(old["test_ids"]).astype(str)
    if len(np.unique(old_ids)) != len(old_ids):
        raise RuntimeError("old baseline test IDs are not unique")
    lookup = {value: index for index, value in enumerate(old_ids.tolist())}
    try:
        order = np.asarray([lookup[value] for value in current_ids.astype(str).tolist()], dtype=np.int64)
    except KeyError as exc:
        raise RuntimeError(f"old baseline missing current test ID: {exc}") from exc
    if set(old_ids.tolist()) != set(current_ids.astype(str).tolist()):
        raise RuntimeError("old baseline/current test ID sets differ")
    result = dict(old)
    for key in ("input_qc", "final_ma", "final_ma_class", "lr_action", "lr_score"):
        if key in old:
            value = np.asarray(old[key])
            if value.ndim >= 1 and value.shape[-1] == len(old_ids):
                result[key] = value[..., order]
    result["test_ids"] = old_ids[order]
    result["test_global_index"] = np.asarray(old["test_global_index"])[order]
    return result


def _arm_arrays(arrays: Mapping[str, np.ndarray], arm: str) -> tuple[np.ndarray, np.ndarray]:
    if arm == "RIDGE10_MA":
        return np.asarray(arrays["final_ma_class"], dtype=np.int8), np.asarray(arrays["lr_action"], dtype=bool)
    if arm == "ORIGINAL_MA":
        return np.asarray(arrays["final_ma_class"], dtype=np.int8), np.asarray(arrays["lr_action"], dtype=bool)
    raise ValueError(arm)


def metrics(arrays: Mapping[str, np.ndarray], truth: np.ndarray, ids: np.ndarray,
            arm: str) -> dict[str, Any]:
    """Recompute all fixed-pool MA metrics from saved predictions/actions."""
    pred, action = _arm_arrays(arrays, arm)
    if arm == "ORIGINAL_MA":
        # The caller supplies an old-baseline view using the same field names.
        pass
    pred = np.asarray(pred, dtype=np.int8)
    truth = np.asarray(truth, dtype=np.int8)
    action = np.asarray(action, dtype=bool)
    if pred.shape != action.shape or pred.shape != (len(SEEDS), len(truth)):
        raise RuntimeError(f"metric shape mismatch for {arm}: pred={pred.shape} action={action.shape} truth={truth.shape}")
    valid_pred = np.isin(pred, MA_CLASSES)
    known = np.isin(truth, MA_CLASSES)[None, :]
    release = action & valid_pred
    release_known = release & known
    correct = release_known & (pred == truth[None, :])
    wrong = release_known & ~correct
    low_to_within = release & (truth[None, :] == -1) & (pred == 0)
    high_to_within = release & (truth[None, :] == 1) & (pred == 0)
    normal_to_low = release & (truth[None, :] == 0) & (pred == -1)
    normal_to_high = release & (truth[None, :] == 0) & (pred == 1)
    low_to_high = release & (truth[None, :] == -1) & (pred == 1)
    high_to_low = release & (truth[None, :] == 1) & (pred == -1)
    correct_low = release_known & (truth[None, :] == -1) & (pred == -1)
    correct_within = release_known & (truth[None, :] == 0) & (pred == 0)
    correct_high = release_known & (truth[None, :] == 1) & (pred == 1)
    reject = known & ~release
    total_released = int(release.sum())
    total_correct = int(correct.sum())
    total_wrong = int(wrong.sum())
    evidence = {
        "low_distinct_records": int(np.sum(truth == -1)),
        "high_distinct_records": int(np.sum(truth == 1)),
        "within_report_distinct_records": int(np.unique(ids[np.any(release & (pred == 0), axis=0)]).size),
        # A reported within record is not evidence of a valid normal report
        # when a true low/high record was released as within.  Keep the broad
        # report count and the correctly supported count separate.
        "correct_within_distinct_records": int(
            np.unique(ids[np.any(correct_within, axis=0)]).size
        ),
    }
    evidence_ok = bool(evidence["low_distinct_records"] >= 5
                       and evidence["high_distinct_records"] >= 5
                       and evidence["correct_within_distinct_records"] >= 5)
    per_seed: list[dict[str, Any]] = []
    for j, seed in enumerate(SEEDS):
        released = int(release[j].sum())
        correct_n = int(correct[j].sum())
        wrong_n = int(wrong[j].sum())
        per_seed.append({
            "seed": int(seed), "released": released, "correct": correct_n, "wrong": wrong_n,
            "agreement": (float(correct_n / released) if released else None),
            "coverage": (float(released / len(truth)) if len(truth) else 0.0),
            "risk": (float(wrong_n / released) if released else None),
            "low_to_within_record_seed": int(low_to_within[j].sum()),
            "high_to_within_record_seed": int(high_to_within[j].sum()),
            "active": bool(released > 0),
            "threshold_eligible": (bool(arrays["lr_threshold_eligible"][j])
                                   if "lr_threshold_eligible" in arrays else None),
        })
    main_pass = bool(total_released > 0 and total_correct / total_released >= 0.80
                     and total_released / (len(SEEDS) * len(truth)) >= 0.20
                     and int(low_to_within.sum()) == 0 and int(high_to_within.sum()) == 0
                     and evidence_ok)
    return {
        "arm": arm, "n_seed": len(SEEDS), "n_test": int(len(truth)),
        "total_record_seed": int(len(SEEDS) * len(truth)),
        "total_released": total_released, "total_correct": total_correct, "total_wrong": total_wrong,
        "agreement": (float(total_correct / total_released) if total_released else None),
        "coverage": float(total_released / (len(SEEDS) * len(truth))) if len(truth) else 0.0,
        "risk": (float(total_wrong / total_released) if total_released else None),
        "low_to_within": _record_flow(low_to_within, truth, ids, -1),
        "high_to_within": _record_flow(high_to_within, truth, ids, 1),
        "normal_to_low": _record_flow(normal_to_low, truth, ids, 0),
        "normal_to_high": _record_flow(normal_to_high, truth, ids, 0),
        "low_to_high": _record_flow(low_to_high, truth, ids, -1),
        "high_to_low": _record_flow(high_to_low, truth, ids, 1),
        "correct_low": _record_flow(correct_low, truth, ids, -1),
        "correct_within": _record_flow(correct_within, truth, ids, 0),
        "correct_high": _record_flow(correct_high, truth, ids, 1),
        "normal_to_abnormal": {
            "event_n_record_seed": int(normal_to_low.sum() + normal_to_high.sum()),
            "denominator_record_seed": int(np.sum(truth == 0) * len(SEEDS)),
            "unique_affected_records": int(np.unique(ids[np.any(normal_to_low | normal_to_high, axis=0)]).size),
            "low_event_n_record_seed": int(normal_to_low.sum()),
            "high_event_n_record_seed": int(normal_to_high.sum()),
        },
        "low_high_cross": {
            "event_n_record_seed": int(low_to_high.sum() + high_to_low.sum()),
            "denominator_record_seed": int((np.sum(truth == -1) + np.sum(truth == 1)) * len(SEEDS)),
            "unique_affected_records": int(np.unique(ids[np.any(low_to_high | high_to_low, axis=0)]).size),
            "low_to_high_n": int(low_to_high.sum()), "high_to_low_n": int(high_to_low.sum()),
        },
        "correct_tail_release": {"low": int(correct_low.sum()), "high": int(correct_high.sum())},
        "rejection_by_predicted_class": {
            "low": int(np.sum(reject & (pred == -1))),
            "within": int(np.sum(reject & (pred == 0))),
            "high": int(np.sum(reject & (pred == 1))),
            "invalid_or_qc": int(np.sum(reject & ~np.isin(pred, MA_CLASSES))),
        },
        "rejection_by_true_class": {
            "low": int(np.sum(reject & (truth[None, :] == -1))),
            "within": int(np.sum(reject & (truth[None, :] == 0))),
            "high": int(np.sum(reject & (truth[None, :] == 1))),
        },
        "active_seed_count": int(sum(row["active"] for row in per_seed)),
        "support_evidence": evidence,
        "support_rule": {
            "minimum_distinct_low_records": 5,
            "minimum_distinct_high_records": 5,
            "minimum_distinct_correct_within_records": 5,
            "within_report_distinct_records_is_not_correct_support": True,
        },
        "evidence_qualification_pass": evidence_ok,
        "main_objective_pass": main_pass,
        "per_seed": per_seed,
    }


def _stage_diagnostics(arrays: Mapping[str, np.ndarray], truth: np.ndarray,
                       ids: np.ndarray) -> dict[str, Any]:
    """Class-flow diagnostics before the final MA-ERROR-LR action.

    These are descriptive fixed-test-pool diagnostics, not alternative
    selectors.  Raw Ridge-10, pure-Huber, and final calibrated classes are counted
    on QC-passing finite outputs; the final-report row uses the frozen LR
    action.  Keeping the stages separate makes mapping/calibration/gate loss
    visible without changing the published arm.
    """
    truth = np.asarray(truth, dtype=np.int8)
    ids = np.asarray(ids).astype(str)
    qc = np.asarray(arrays["input_qc"], dtype=bool)

    def class_stage(med4: np.ndarray) -> dict[str, Any]:
        med4 = np.asarray(med4, dtype=float)
        pred = ma_class(med4[:, :, 2])
        valid = qc[None, :] & np.isfinite(med4).all(axis=2) & np.isin(pred, MA_CLASSES)
        low_to_within = valid & (truth[None, :] == -1) & (pred == 0)
        high_to_within = valid & (truth[None, :] == 1) & (pred == 0)
        correct_low = valid & (truth[None, :] == -1) & (pred == -1)
        correct_high = valid & (truth[None, :] == 1) & (pred == 1)
        normal_to_low = valid & (truth[None, :] == 0) & (pred == -1)
        normal_to_high = valid & (truth[None, :] == 0) & (pred == 1)
        return {
            "denominator_record_seed": int(len(SEEDS) * len(truth)),
            "qc_passing_record_seed": int(valid.sum()),
            "predicted_class_record_seed": {
                "low": int(np.sum(valid & (pred == -1))),
                "within": int(np.sum(valid & (pred == 0))),
                "high": int(np.sum(valid & (pred == 1))),
            },
            "low_to_within": _record_flow(low_to_within, truth, ids, -1),
            "high_to_within": _record_flow(high_to_within, truth, ids, 1),
            "correct_low": _record_flow(correct_low, truth, ids, -1),
            "correct_high": _record_flow(correct_high, truth, ids, 1),
            "normal_to_low": _record_flow(normal_to_low, truth, ids, 0),
            "normal_to_high": _record_flow(normal_to_high, truth, ids, 0),
        }

    final_report = metrics(arrays, truth, ids, "RIDGE10_MA")
    return {
        "raw_mapping": class_stage(np.asarray(arrays["raw_med4"], dtype=float)),
        "pure_huber": class_stage(np.asarray(arrays["pure_huber_med4"], dtype=float)),
        "final_calibrated": class_stage(np.asarray(arrays["final_med4"], dtype=float)),
        "final_report": {
            "released_record_seed": int(final_report["total_released"]),
            "low_to_within": final_report["low_to_within"],
            "high_to_within": final_report["high_to_within"],
            "correct_low": final_report["correct_low"],
            "correct_high": final_report["correct_high"],
            "normal_to_low": final_report["normal_to_low"],
            "normal_to_high": final_report["normal_to_high"],
        },
        "interpretation": "raw/pure/final rows are descriptive class flows; final_report is the frozen MA-ERROR-LR released action",
    }


def _bootstrap_weights(truth: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Class-stratified record bootstrap; the seed axis is never sampled."""
    weights = np.zeros((BOOTSTRAP_DRAWS, len(truth)), dtype=np.float64)
    for cls in MA_CLASSES:
        indices = np.flatnonzero(truth == cls)
        if not len(indices):
            continue
        draws = rng.integers(0, len(indices), size=(BOOTSTRAP_DRAWS, len(indices)))
        for draw in range(BOOTSTRAP_DRAWS):
            weights[draw, indices] = np.bincount(draws[draw], minlength=len(indices))
    return weights


def _bootstrap_arm(arrays: Mapping[str, np.ndarray], truth: np.ndarray,
                   weights: np.ndarray, arm: str) -> dict[str, np.ndarray]:
    pred, action = _arm_arrays(arrays, arm)
    valid_pred = np.isin(pred, MA_CLASSES)
    release = action & valid_pred
    known = np.isin(truth, MA_CLASSES)[None, :]
    release_known = release & known
    correct = release_known & (pred == truth[None, :])
    wrong = release_known & ~correct
    denominator = np.full(len(weights), float(len(truth) * len(SEEDS)))
    release_n = weights @ release.sum(axis=0)
    correct_n = weights @ correct.sum(axis=0)
    wrong_n = weights @ wrong.sum(axis=0)
    output = {
        "coverage": np.divide(release_n, denominator, out=np.zeros_like(release_n), where=denominator > 0),
        "agreement": np.divide(correct_n, release_n, out=np.full(len(denominator), np.nan), where=release_n > 0),
        "risk": np.divide(wrong_n, release_n, out=np.full(len(denominator), np.nan), where=release_n > 0),
    }
    for cls, name in ((-1, "low_to_within"), (1, "high_to_within")):
        event = release & (truth[None, :] == cls) & (pred == 0)
        support = weights @ (truth == cls).astype(float)
        output[name] = np.divide(weights @ event.sum(axis=0), support * len(SEEDS),
                                  out=np.full(len(denominator), np.nan), where=support > 0)
    return output


def _quantile_ci(values: np.ndarray) -> list[float | None]:
    finite = np.isfinite(values)
    if not finite.any():
        return [None, None]
    return [float(x) for x in np.nanquantile(values, [0.025, 0.975])]


def _bootstrap_all(rows: Mapping[str, Mapping[str, Any]], refs: Mapping[str, Mapping[str, np.ndarray]],
                   metrics_by_direction: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    arrays_out: dict[str, np.ndarray] = {}
    summary: dict[str, Any] = {
        "schema": "ridge10_fullchain_record_category_bootstrap_v1", "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED, "seed_is_not_patient": True, "shared_target_weights": {}, "directions": {},
    }
    by_target: dict[str, list[str]] = {}
    for direction, row in rows.items():
        by_target.setdefault(str(row["target"]), []).append(direction)
    for target, directions in by_target.items():
        truth = np.asarray(refs[target]["ma_reference_class"], dtype=np.int8)
        weights = _bootstrap_weights(truth, rng)
        arrays_out[f"{target}_record_weights"] = weights
        summary["shared_target_weights"][target] = {
            "directions": directions, "record_count": int(len(truth)), "record_level": True,
            "category_stratified": True, "paired_across_directions": True,
        }
        for direction in directions:
            row = rows[direction]
            summary["directions"].setdefault(direction, {})
            draw_cache: dict[str, dict[str, np.ndarray]] = {}
            for arm in ("RIDGE10_MA", "ORIGINAL_MA"):
                draws = _bootstrap_arm(row["arrays"][arm], truth, weights, arm)
                draw_cache[arm] = draws
                entry: dict[str, Any] = {"point": {}, "ci95": {}}
                for key, values in draws.items():
                    values = np.asarray(values, dtype=float)
                    point = metrics_by_direction[direction][arm].get(key)
                    if isinstance(point, Mapping):
                        point = point.get("event_rate")
                    entry["point"][key] = point
                    entry["ci95"][key] = _quantile_ci(values)
                    arrays_out[f"{direction}_{arm}_{key}"] = values
                summary["directions"][direction][arm] = entry
            delta_label = "RIDGE10_MA_minus_ORIGINAL_MA"
            delta: dict[str, Any] = {"point": {}, "ci95": {}}
            for key in ("agreement", "coverage", "risk", "low_to_within", "high_to_within"):
                values = np.asarray(draw_cache["RIDGE10_MA"][key]) - np.asarray(draw_cache["ORIGINAL_MA"][key])
                left = metrics_by_direction[direction]["RIDGE10_MA"].get(key)
                right = metrics_by_direction[direction]["ORIGINAL_MA"].get(key)
                if isinstance(left, Mapping):
                    left = left.get("event_rate")
                if isinstance(right, Mapping):
                    right = right.get("event_rate")
                delta["point"][key] = float(left - right) if left is not None and right is not None else None
                delta["ci95"][key] = _quantile_ci(values)
                arrays_out[f"{direction}_{delta_label}_{key}"] = values
            summary["directions"][direction][delta_label] = delta
    path = ROOT / "eval" / "BOOTSTRAP_20K.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays_out)
    summary["npz_sha256"] = sha(path)
    return summary


def _load_prediction_rows() -> dict[str, dict[str, Any]]:
    manifest = read_json(PRED_MANIFEST)
    if manifest.get("status") != "PRE_EVALUATION_LABELS_CLOSED" or manifest.get("test_labels_read") is not False:
        raise RuntimeError("prediction/action manifest is not label-closed")
    if manifest.get("target_evaluation") is not False:
        raise RuntimeError("prediction/action manifest already evaluated")
    rows: dict[str, dict[str, Any]] = {}
    for direction in DIRECTIONS:
        item = manifest.get("directions", {}).get(direction)
        if not isinstance(item, Mapping):
            raise RuntimeError(f"prediction manifest missing {direction}")
        path = Path(str(item["path"]))
        if not path.is_file() or sha(path) != item.get("sha256"):
            raise RuntimeError(f"prediction hash mismatch: {path}")
        arrays = load_npz(path)
        forbidden = {"medical", "ci", "reference", "truth"}.intersection(arrays)
        if forbidden:
            raise RuntimeError(f"label-bearing keys in frozen predictions: {direction}:{forbidden}")
        target = str(_scalar(arrays["target"]))
        if target != TARGET_BY_DIRECTION[direction] or len(arrays["test_ids"]) != EXPECTED_TEST_N[target]:
            raise RuntimeError(f"prediction identity mismatch: {direction}")
        if np.asarray(arrays["seed"]).tolist() != list(SEEDS):
            raise RuntimeError(f"seed order mismatch: {direction}")
        rows[direction] = {"target": target, "path": path, "arrays": arrays, "manifest": item}
    return rows


def _fmt(value: Any, digits: int = 4) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def _fmt_flow(flow: Mapping[str, Any]) -> str:
    return f"{flow.get('event_n_record_seed', 0)}/{flow.get('denominator_record_seed', 0)}"


def _write_report(result: Mapping[str, Any], bootstrap: Mapping[str, Any]) -> Path:
    path = ROOT / "REPORT_ZH.md"
    lines = [
        "# Ridge-10 九方向冻结数值 MA 评价",
        "",
        "主候选为固定 Ridge-10 四参数映射 + 原校准/adapter + 原 MA-ERROR-LR；原九方向 MA 结果作为只读对照。",
        "评价使用冻结的 prediction/action，一次性读取固定 test reference；没有按 test 选方向、seed、模型或回调阈值。",
        "一致率为 pooled `correct / released`，覆盖为 `released / (20 × 固定 test n)`；seed 重复不是独立患者。",
        "低/高→within、正常→异常、低↔高、正确尾放行和拒答均按 record-seed 次数报告，另列 unique record 数。",
        "",
        "| 方向 | 目标 | arm | correct/released | agreement | coverage | low→within | high→within | normal→abnormal | low↔high | correct low/high | correct-within support | active seeds | main80/20/0+support |",
        "|---|---|---|---:|---:|---:|---|---|---|---|---|---|---:|---|",
    ]
    main_passes = 0
    for direction, row in result["directions"].items():
        for arm in ("RIDGE10_MA", "ORIGINAL_MA"):
            metric = row["metrics"][arm]
            if arm == "RIDGE10_MA" and metric.get("main_objective_pass"):
                main_passes += 1
            low = metric["low_to_within"]
            high = metric["high_to_within"]
            normal = metric["normal_to_abnormal"]
            cross = metric["low_high_cross"]
            tails = metric["correct_tail_release"]
            status = "PASS" if metric.get("main_objective_pass") else ("NA" if metric.get("agreement") is None else "NOT_MET")
            lines.append(
                f"| {direction} | {row['target']} | {arm} | "
                f"{metric['total_correct']}/{metric['total_released']} | {_fmt(metric['agreement'])} | "
                f"{_fmt(metric['coverage'])} | {_fmt_flow(low)} (u={low['unique_affected_records']}) | "
                f"{_fmt_flow(high)} (u={high['unique_affected_records']}) | "
                f"{normal['event_n_record_seed']}/{normal['denominator_record_seed']} (u={normal['unique_affected_records']}) | "
                f"{cross['event_n_record_seed']}/{cross['denominator_record_seed']} (u={cross['unique_affected_records']}) | "
                f"{tails['low']}/{tails['high']} | "
                f"{metric['support_evidence']['correct_within_distinct_records']} distinct | "
                f"{metric['active_seed_count']} | {status} |"
            )
    aggregate = result.get("aggregate", {})
    aggregate_status = "PASS" if aggregate.get("main_objective_pass") else "NOT_MET"
    lines.extend([
        "",
        f"Ridge-10 主 arm 达到完整单向+双向研究目标：{aggregate_status}；"
        f"单向通过 {aggregate.get('single_pass_n', 0)}/6，双向通过 {aggregate.get('dual_pass_n', 0)}/3。",
        f"逐方向完整 80%/20%/双尾零误放/支持条件通过数：{main_passes}/9；"
        "aggregate 不能由全部拒答或仅一类方向通过伪造。",
        "支持证据要求 low/high 各至少 5 个独立参考记录，且至少 5 个独立的正确 within（true within 且 released as within）；所有 within 报告数另列，错误 tail→within 不计入正确支持。",
        "零 release 时 coverage=0、agreement=NA；无尾部支持时相应事件率为 NA，不能把它解释为零错误。",
        "",
        "## 映射—校准—门阶段诊断",
        "",
        "下表仅定位固定 test 池中的已冻结链路差异，不构成额外候选或 test 选型。raw/pure/final 为 QC 通过且数值有限样本的预测类别流；final report 才应用 MA-ERROR-LR action。",
        "",
        "| 方向 | stage | low→within | high→within | correct low | correct high | normal→low/high |",
        "|---|---|---|---|---|---|---|",
    ])
    for direction, row in result["directions"].items():
        diag = row.get("stage_diagnostics", {})
        for stage in ("raw_mapping", "pure_huber", "final_calibrated"):
            value = diag.get(stage, {})
            lines.append(
                f"| {direction} | {stage} | {_fmt_flow(value.get('low_to_within', {}))} | "
                f"{_fmt_flow(value.get('high_to_within', {}))} | "
                f"{_fmt_flow(value.get('correct_low', {}))} | {_fmt_flow(value.get('correct_high', {}))} | "
                f"{value.get('normal_to_low', {}).get('event_n_record_seed', 0)}/"
                f"{value.get('normal_to_high', {}).get('event_n_record_seed', 0)} |"
            )
    lines.extend([
        "",
        "## 可复核文件",
        "",
        "- `FULL_METRICS.json`：两 arm 的 pooled/逐 seed 指标、完整分母、unique record 数和支持证据。",
        "- `STAGE_DIAGNOSTICS.json`：raw Ridge-10→pure Huber→final calibration→最终 report 的固定链路诊断。",
        "- `BOOTSTRAP_SUMMARY.json` 与 `eval/BOOTSTRAP_20K.npz`：同 target 共享、按 MA 类分层的 20,000 次 record-level bootstrap 及 Ridge-10−原 MA 配对差值。",
        "- 区间条件于冻结模型与本 test 池；零事件 [0,0] 仅为重采样区间，不是总体错误率或临床安全保证。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _load_old_baseline(direction: str, current_ids: np.ndarray, current_qc: np.ndarray) -> dict[str, np.ndarray]:
    path = OLD_PRED / f"{direction}.npz"
    if not path.is_file():
        raise RuntimeError(f"original MA baseline missing: {path}")
    old = load_npz(path)
    aligned = _align_baseline(current_ids, old)
    if not np.array_equal(np.asarray(aligned["input_qc"], dtype=bool), current_qc):
        raise RuntimeError(f"baseline QC differs from frozen current denominator: {direction}")
    if np.any(np.asarray(aligned["lr_action"], dtype=bool) & ~current_qc[None, :]):
        raise RuntimeError(f"original MA baseline releases a QC-failed record: {direction}")
    if np.asarray(aligned["seed"]).tolist() != list(SEEDS):
        raise RuntimeError(f"baseline seed order mismatch: {direction}")
    return aligned


def preflight_no_labels() -> dict[str, Any]:
    """Check all frozen identities without opening target reference arrays."""
    failures: list[str] = []
    for path in (PRED_MANIFEST, MODEL_MANIFEST, THRESHOLD_MANIFEST, ROLE_PATH):
        if not path.is_file():
            failures.append(f"missing:{path.name}")
    for marker in (ROOT / "TARGET_LABEL_ACCESS.json", ROOT / "EVALUATION_START.json", ROOT / "EVALUATION_COMPLETE.json"):
        if marker.exists():
            failures.append(f"label_marker_exists:{marker.name}")
    try:
        model_manifest = read_json(MODEL_MANIFEST)
        if model_manifest.get("status") != "ALL_MODELS_AND_THRESHOLDS_FROZEN_TEST_BLIND":
            failures.append("model_manifest_status")
        if model_manifest.get("test_labels_read") is not False:
            failures.append("model_manifest_test_labels")
        if model_manifest.get("test_evaluated") is not False:
            failures.append("model_manifest_test_evaluated")
        target_hashes = model_manifest.get("target_object_hashes", {})
        if not isinstance(target_hashes, Mapping) or len(target_hashes) != 180:
            failures.append("model_manifest_counts")
        else:
            for rel, expected_sha in target_hashes.items():
                path = ROOT / str(rel)
                if not path.is_file() or sha(path) != str(expected_sha):
                    failures.append(f"model_manifest_object_hash:{rel}")
        threshold_manifest = read_json(THRESHOLD_MANIFEST)
        if threshold_manifest.get("test_labels_read") is not False or threshold_manifest.get("test_evaluated") is not False:
            failures.append("threshold_manifest_test_firewall")
        if threshold_manifest.get("max_risk_caps") != {"Huzhou": 0.1, "SAHZU": 0.15, "Hubei": 0.15}:
            failures.append("threshold_manifest_caps")
        role = read_json(ROLE_PATH)
        for target, expected in EXPECTED_TEST_N.items():
            if len(role.get("roles", {}).get(target, {}).get("test", [])) != expected:
                failures.append(f"role_test_denominator:{target}")
    except Exception as exc:
        failures.append(f"freeze_metadata_exception:{type(exc).__name__}:{exc}")
    try:
        rows = _load_prediction_rows()
        baseline_hashes = {}
        for direction, row in rows.items():
            arrays = row["arrays"]
            old_path = OLD_PRED / f"{direction}.npz"
            if not old_path.is_file():
                failures.append(f"missing_old_baseline:{direction}")
                continue
            old = load_npz(old_path)
            current_ids = np.asarray(arrays["test_ids"]).astype(str)
            aligned = _align_baseline(current_ids, old)
            if not np.array_equal(np.asarray(aligned["input_qc"], dtype=bool), np.asarray(arrays["input_qc"], dtype=bool)):
                failures.append(f"baseline_qc_mismatch:{direction}")
            baseline_hashes[direction] = {"path": str(old_path), "sha256": sha(old_path),
                                          "ids_sha256": digest(current_ids.tolist())}
            if np.asarray(arrays["lr_action"]).shape != (len(SEEDS), len(current_ids)):
                failures.append(f"action_shape:{direction}")
    except Exception as exc:
        failures.append(f"exception:{type(exc).__name__}:{exc}")
        rows = {}
        baseline_hashes = {}
    result = {
        "schema": "ridge10_fullchain_final_evaluation_preflight_v1",
        "status": "PASS_NO_LABELS" if not failures else "FAIL_NO_LABELS",
        "target_evaluation": False, "test_labels_read": False,
        "prediction_actions_frozen": True, "label_dependent_actions_written": False,
        "test_actions_written": False,
        "fit_units": 0, "directions": len(rows), "baseline_hashes": baseline_hashes,
        "failures": failures,
        "prediction_manifest_sha256": sha(PRED_MANIFEST) if PRED_MANIFEST.is_file() else None,
        "model_manifest_sha256": sha(MODEL_MANIFEST) if MODEL_MANIFEST.is_file() else None,
        "threshold_manifest_sha256": sha(THRESHOLD_MANIFEST) if THRESHOLD_MANIFEST.is_file() else None,
        # Keep the persisted local preflight digest deterministic.  A fresh
        # invocation may recheck the same frozen inputs without invalidating
        # the approval binding merely because a wall-clock field changed.
        "evaluator_sha256": sha(Path(__file__)),
    }
    result["preflight_sha256"] = digest(result)
    write_json(ROOT / "FINAL_EVALUATION_PREFLIGHT_LOCAL.json", result)
    return result


def _frozen_identity_bindings() -> dict[str, str]:
    """Return every immutable identity required before medical access.

    This deliberately hashes both algorithm sources and the round/state
    ledgers.  The approved path must not rely on an approval that binds only
    the six user-facing manifests while a runner, protocol, or ledger has
    drifted underneath it.
    """
    paths = {
        "runner_sha256": RUNNER_CODE,
        "runtime_sha256": RUNTIME_CODE,
        "protocol_sha256": PROTOCOL_PATH,
        "runner_freeze_sha256": RUNNER_FREEZE,
        "threshold_freeze_code_sha256": THRESHOLD_CODE,
        "threshold_manifest_sha256": THRESHOLD_MANIFEST,
        "model_manifest_sha256": MODEL_MANIFEST,
        "round_start_ledger_sha256": ROUND_LEDGER,
        "current_fit_ledger_sha256": CURRENT_LEDGER,
        "state_sha256": STATE_PATH,
        "prediction_schema_sha256": PREDICTION_SCHEMA,
        "evaluation_protocol_sha256": EVALUATION_PROTOCOL,
        "prediction_action_manifest_sha256": PRED_MANIFEST,
        "data15_sha256": BASE_DATA,
        "role_freeze_sha256": ROLE_PATH,
        "source_context_sha256": SOURCE_CONTEXTS,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError("REFUSED: frozen identity file missing: " + ", ".join(missing))
    return {key: sha(path) for key, path in paths.items()}


def _require_approval() -> dict[str, Any]:
    local = preflight_no_labels()
    if local.get("status") != "PASS_NO_LABELS":
        raise RuntimeError("REFUSED: no-label preflight failed")
    identity = _frozen_identity_bindings()
    request = read_json(PREFLIGHT_REQUEST)
    request_bindings = request.get("bindings", {})
    if not isinstance(request_bindings, Mapping):
        raise RuntimeError("REFUSED: preflight request bindings are missing")
    for key, expected in identity.items():
        if request_bindings.get(key) != expected:
            raise RuntimeError(f"REFUSED: request frozen-identity mismatch: {key}")
    if request_bindings.get("evaluator_sha256") != sha(Path(__file__)):
        raise RuntimeError("REFUSED: request evaluator identity mismatch")
    if request_bindings.get("local_preflight_sha256") != local.get("preflight_sha256"):
        raise RuntimeError("REFUSED: request local-preflight digest mismatch")
    local_path = ROOT / "FINAL_EVALUATION_PREFLIGHT_LOCAL.json"
    if request_bindings.get("local_preflight_file_sha256") != sha(local_path):
        raise RuntimeError("REFUSED: request local-preflight file mismatch")
    approval_path = ROOT / "TEST_EVALUATION_APPROVAL.json"
    pass_path = ROOT / "FINAL_INDEPENDENT_PREFLIGHT_PASS.json"
    if not pass_path.is_file():
        raise RuntimeError("REFUSED: independent final preflight pass is missing")
    independent = read_json(pass_path)
    if independent.get("status") not in {"PASS", "PASS_FINAL_NO_LABELS"} or independent.get("test_labels_read") is not False:
        raise RuntimeError("REFUSED: independent preflight is not label-closed PASS")
    if independent.get("request_sha256") != sha(PREFLIGHT_REQUEST):
        raise RuntimeError("REFUSED: independent preflight request binding mismatch")
    if independent.get("evaluator_sha256") != sha(Path(__file__)):
        raise RuntimeError("REFUSED: independent preflight evaluator binding mismatch")
    if independent.get("prediction_manifest_sha256") != sha(PRED_MANIFEST):
        raise RuntimeError("REFUSED: independent preflight prediction binding mismatch")
    for key, expected in identity.items():
        if independent.get(key) != expected:
            raise RuntimeError(f"REFUSED: independent preflight frozen-identity mismatch: {key}")
    if not approval_path.is_file():
        raise RuntimeError("REFUSED: one-shot approval is missing")
    approval = read_json(approval_path)
    required = {
        "approved": True, "target_evaluation": True, "one_shot_count": 1,
        "prediction_manifest_sha256": sha(PRED_MANIFEST),
        "model_manifest_sha256": sha(MODEL_MANIFEST),
        "threshold_manifest_sha256": sha(THRESHOLD_MANIFEST),
        "evaluation_protocol_sha256": sha(EVALUATION_PROTOCOL),
        "data15_sha256": sha(BASE_DATA),
        "role_freeze_sha256": sha(ROLE_PATH),
        "runner_sha256": identity["runner_sha256"],
        "runtime_sha256": identity["runtime_sha256"],
        "protocol_sha256": identity["protocol_sha256"],
        "runner_freeze_sha256": identity["runner_freeze_sha256"],
        "threshold_freeze_code_sha256": identity["threshold_freeze_code_sha256"],
        "round_start_ledger_sha256": identity["round_start_ledger_sha256"],
        "current_fit_ledger_sha256": identity["current_fit_ledger_sha256"],
        "state_sha256": identity["state_sha256"],
        "source_context_sha256": identity["source_context_sha256"],
        "prediction_schema_sha256": identity["prediction_schema_sha256"],
        "evaluator_sha256": sha(Path(__file__)),
        "local_preflight_sha256": local.get("preflight_sha256"),
        "local_preflight_file_sha256": sha(local_path),
        "independent_preflight_sha256": sha(pass_path),
        "preflight_request_sha256": sha(PREFLIGHT_REQUEST),
    }
    for key, expected in required.items():
        if approval.get(key) != expected:
            raise RuntimeError(f"REFUSED: approval binding mismatch: {key}")
    for marker in (ROOT / "TARGET_LABEL_ACCESS.json", ROOT / "EVALUATION_START.json", ROOT / "EVALUATION_COMPLETE.json"):
        if marker.exists():
            raise RuntimeError(f"REFUSED: one-shot marker already exists: {marker.name}")
    return approval


def _read_references_after_approval() -> dict[str, dict[str, np.ndarray]]:
    """The first function permitted to request DATA15 medical values."""
    role = read_json(ROLE_PATH)
    with np.load(BASE_DATA, allow_pickle=False) as archive:
        ids = np.asarray(archive["ids"]).astype(str)
        medical = np.asarray(archive["medical"], dtype=float)
    refs: dict[str, dict[str, np.ndarray]] = {}
    for target in TARGETS:
        index = np.asarray(role["roles"][target]["test"], dtype=np.int64)
        ma = medical[index, 2]
        classes = ma_class(ma)
        if not np.isin(classes, MA_CLASSES).all():
            raise RuntimeError(f"unknown/nonfinite MA reference class in approved test pool: {target}")
        refs[target] = {
            "test_ids": ids[index], "test_global_index": index, "ma_reference": ma,
            "ma_reference_class": classes,
        }
    write_json(ROOT / "TARGET_LABEL_ACCESS.json", {
        "schema": "ridge10_fullchain_target_reference_access_v1", "target_evaluation": True,
        "test_reference_read": True, "ci_labels_read": False,
        "targets": {target: {"n": int(len(refs[target]["test_ids"])),
                              "ids_sha256": digest(refs[target]["test_ids"].tolist())}
                     for target in TARGETS},
    })
    return refs


def _save_references(refs: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    for target, ref in refs.items():
        path = ROOT / "eval" / "references" / f"{target}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, target=np.asarray(target), test_global_index=ref["test_global_index"],
                            test_ids=ref["test_ids"], ma_reference=ref["ma_reference"],
                            ma_reference_class=ref["ma_reference_class"])


def evaluate_once() -> dict[str, Any]:
    approval = _require_approval()
    write_json(ROOT / "EVALUATION_START.json", {
        "schema": "ridge10_fullchain_one_shot_evaluation_start_v1", "target_evaluation": True,
        "test_labels_read": False, "approval_sha256": sha(ROOT / "TEST_EVALUATION_APPROVAL.json"),
        "prediction_manifest_sha256": sha(PRED_MANIFEST), "evaluator_sha256": sha(Path(__file__)),
        "started": time.time(),
    })
    rows = _load_prediction_rows()
    refs = _read_references_after_approval()
    _save_references(refs)
    result: dict[str, Any] = {
        "schema": "ridge10_fullchain_frozen_ma_metrics_v1", "target_evaluation": True,
        "test_evaluation_count": 1, "selection_from_test": False, "threshold_refit": False,
        "approval_sha256": sha(ROOT / "TEST_EVALUATION_APPROVAL.json"),
        "prediction_manifest_sha256": sha(PRED_MANIFEST), "directions": {},
    }
    metric_cache: dict[str, dict[str, Any]] = {}
    stage_cache: dict[str, Any] = {}
    for direction, row in rows.items():
        target = row["target"]
        arrays = row["arrays"]
        current_ids = np.asarray(arrays["test_ids"]).astype(str)
        old = _load_old_baseline(direction, current_ids, np.asarray(arrays["input_qc"], dtype=bool))
        krr_view = {key: value for key, value in arrays.items()}
        baseline_view = {
            "final_ma_class": np.asarray(old["final_ma_class"], dtype=np.int8),
            "lr_action": np.asarray(old["lr_action"], dtype=bool),
            "lr_threshold_eligible": np.asarray(old.get("lr_threshold_eligible", np.zeros(len(SEEDS))), dtype=bool),
        }
        truth = np.asarray(refs[target]["ma_reference_class"], dtype=np.int8)
        if not np.array_equal(current_ids, np.asarray(refs[target]["test_ids"]).astype(str)):
            raise RuntimeError(f"reference/prediction ID order mismatch: {direction}")
        ridge_metrics = metrics(krr_view, truth, current_ids, "RIDGE10_MA")
        old_metrics = metrics(baseline_view, truth, current_ids, "ORIGINAL_MA")
        metric_cache[direction] = {"RIDGE10_MA": ridge_metrics, "ORIGINAL_MA": old_metrics}
        stage_cache[direction] = _stage_diagnostics(arrays, truth, current_ids)
        result["directions"][direction] = {
            "target": target, "test_n": int(len(truth)),
            "prediction_sha256": sha(row["path"]), "original_baseline_sha256": sha(OLD_PRED / f"{direction}.npz"),
            "metrics": metric_cache[direction], "stage_diagnostics": stage_cache[direction],
        }
        row["arrays"] = {"RIDGE10_MA": krr_view, "ORIGINAL_MA": baseline_view}
    ridge_pass_directions = [
        direction for direction in DIRECTIONS
        if bool(result["directions"][direction]["metrics"]["RIDGE10_MA"].get("main_objective_pass"))
    ]
    dual_pass = [direction for direction in ridge_pass_directions if direction in DUAL_DIRECTIONS]
    single_pass = [direction for direction in ridge_pass_directions if direction in SINGLE_DIRECTIONS]
    result["aggregate"] = {
        "arm": "RIDGE10_MA",
        "main_objective_pass": bool(dual_pass and single_pass),
        "dual_pass_directions": dual_pass,
        "single_pass_directions": single_pass,
        "dual_pass_n": int(len(dual_pass)),
        "single_pass_n": int(len(single_pass)),
        "required": "at least one single direction and one dual direction must each satisfy 80% agreement, 20% pooled coverage, zero low/high-to-within, and support evidence",
        "support_rule": {
            "low_distinct_records": 5,
            "high_distinct_records": 5,
            "correct_within_distinct_records": 5,
            "within_report_distinct_records_does_not_count_as_correct_support": True,
            "missing_tail_support_is_not_a_pass": True,
        },
    }
    write_json(ROOT / "STAGE_DIAGNOSTICS.json", {
        "schema": "ridge10_fullchain_stage_diagnostics_v1", "target_evaluation": True,
        "directions": stage_cache,
    })
    result["stage_diagnostics_sha256"] = sha(ROOT / "STAGE_DIAGNOSTICS.json")
    write_json(ROOT / "FULL_METRICS.json", result)
    bootstrap = _bootstrap_all(rows, refs, metric_cache)
    write_json(ROOT / "BOOTSTRAP_SUMMARY.json", bootstrap)
    report_path = _write_report(result, bootstrap)
    result["report_sha256"] = sha(report_path)
    result["bootstrap_summary_sha256"] = sha(ROOT / "BOOTSTRAP_SUMMARY.json")
    write_json(ROOT / "FULL_METRICS.json", result)
    write_json(ROOT / "EVALUATION_COMPLETE.json", {
        "schema": "ridge10_fullchain_evaluation_complete_v1", "target_evaluation": True,
        "test_evaluation_count": 1, "approval_sha256": sha(ROOT / "TEST_EVALUATION_APPROVAL.json"),
        "prediction_manifest_sha256": sha(PRED_MANIFEST), "metrics_sha256": sha(ROOT / "FULL_METRICS.json"),
        "bootstrap_sha256": sha(ROOT / "BOOTSTRAP_SUMMARY.json"),
        "stage_diagnostics_sha256": sha(ROOT / "STAGE_DIAGNOSTICS.json"), "selection_from_test": False,
        "threshold_refit": False, "test_labels_read_once": True,
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true", help="validate frozen artifacts without labels")
    parser.add_argument("--evaluate", action="store_true", help="approved one-shot evaluation; reads labels")
    args = parser.parse_args()
    if args.preflight == args.evaluate:
        raise SystemExit("choose exactly one of --preflight or --evaluate")
    if args.preflight:
        result = preflight_no_labels()
        print(json.dumps({"status": result["status"], "directions": result["directions"],
                          "test_labels_read": False, "failures": result["failures"]}, ensure_ascii=False))
        return
    result = evaluate_once()
    print(json.dumps({"status": "EVALUATION_COMPLETE", "directions": len(result["directions"]),
                      "test_labels_read": True, "test_evaluation_count": 1}, ensure_ascii=False))


if __name__ == "__main__":
    main()
