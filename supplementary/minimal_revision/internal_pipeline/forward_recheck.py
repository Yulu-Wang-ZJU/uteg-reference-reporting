"""Read-only forward recheck for the frozen early-report pipeline.

This script does not fit, select thresholds, open reference labels, or write
to the historical experiment directories.  It loads the already frozen
processed prefix features (300/600 s), frozen target objects and dual risk
heads, and the frozen thresholds.  The 900 s checkpoint is replayed from the
frozen 15-min feature/model objects.  It compares the resulting predictions,
scores, validity masks, and actions with the archived action arrays.

The 60/141 partition is the pre-existing hash split from the reporting
increment analysis.  It is a sensitivity partition, not an independent
validation set.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
OUT = PROJECT / "outputs"
SEQ = OUT / "uteg_current_ma_sequence_20261026"
OLD = OUT / "uteg_signal_model_goal_20261026_000929" / "ridge10_fullchain_20261026_122551"
BASE = OUT / "uteg_legacy_aligned_threecenter_20261022_155614"
EXT = OUT / "uteg_ridge_dual_head_external_eval_20261026_191117"
RUNNER = OUT / "uteg_ridge_dual_head_runner_20261026_173951" / "models"
SELECTOR = OUT / "uteg_ridge_dual_head_unified_selector_20261026_190142" / "THRESHOLDS_FREEZE.json"
PARTITION = OUT / "uteg_reporting_increment_20261030" / "subset_mask.json"

DIRECTIONS = ("HZ_to_SA", "HZ+HB_to_SA")
SEEDS = tuple(range(1000, 1020))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    def clean(x: Any) -> Any:
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        return x

    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_partition() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    part = read_json(PARTITION)
    by_global = {int(k): bool(v) for k, v in part["record_mask_by_global_index"].items()}
    data = np.load(BASE / "DATA15.npz", allow_pickle=False)
    # The test order is the frozen order used by the checkpoint action files.
    test_index = np.asarray(sorted(int(x) for x in part["record_mask_by_global_index"]), dtype=int)
    mask = np.asarray([by_global[int(i)] for i in test_index], dtype=bool)
    if int(mask.sum()) != 60 or int((~mask).sum()) != 141:
        raise AssertionError((int(mask.sum()), int((~mask).sum())))
    return test_index, mask, part


def project(med: Any) -> np.ndarray:
    a = np.asarray(med, dtype=float).copy()
    if a.ndim == 1:
        a = a[None, :]
        one = True
    else:
        one = False
    a[..., :3] = np.maximum(a[..., :3], 0.0)
    a[..., 3] = np.clip(a[..., 3], 0.0, 89.999999)
    a[~np.isfinite(a).all(axis=-1)] = np.nan
    return a[0] if one else a


def source_predict(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """Ridge-10 prediction from the stored numerical coefficients."""
    z = (np.asarray(x, dtype=float) - np.asarray(model["xm"])) / np.asarray(model["xs"])
    y = z @ np.asarray(model["coef"]) + np.asarray(model["intercept"])
    return project(np.asarray(model["ym"])[None, :] + np.asarray(model["ys"])[None, :] * y)


def calibrated(cal: dict[str, Any], raw: np.ndarray) -> np.ndarray:
    if not (bool(cal.get("success", False)) and bool(cal.get("finite", False))):
        return np.full_like(raw, np.nan)
    p = project(raw)
    v = (p - np.asarray(cal["mu"])[None, :]) / np.asarray(cal["scale"])[None, :]
    out = np.asarray(cal["mu"])[None, :] + np.asarray(cal["scale"])[None, :] * (
        np.asarray(cal["ubar"])[None, :]
        + (v - np.asarray(cal["vbar"])[None, :]) @ np.asarray(cal["matrix"])
        + np.asarray(cal["intercept"])[None, :]
    )
    return project(out)


def adapter_cond(x: np.ndarray, raw: np.ndarray, source: dict[str, Any]) -> np.ndarray:
    return np.column_stack([np.asarray(x, dtype=float),
                            (project(raw) - np.asarray(source["ym"])[None, :]) / np.asarray(source["ys"])[None, :]])


def adapter_kernel(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.exp(-np.maximum(((x[:, None, :] - z[None, :, :]) ** 2).sum(-1), 0.0) / 16.0)


def adapt(model: dict[str, Any] | None, x: np.ndarray, raw: np.ndarray,
          huber: np.ndarray, source: dict[str, Any]) -> np.ndarray:
    if model is None or bool(model.get("fallback", False)) or model.get("name") == "H":
        return np.asarray(huber, dtype=float).copy()
    z = np.clip((adapter_cond(x, raw, source) - np.asarray(model["mean"])[None, :]) /
                np.asarray(model["sd"])[None, :], -10.0, 10.0)
    if str(model["name"])[0] == "L":
        pred = z @ np.asarray(model["coef"])
    else:
        pred = adapter_kernel(z, np.asarray(model["train_z"])) @ np.asarray(model["coef"])
    delta = 3.0 * (float(model["rb"]) + pred)
    return project(np.asarray(huber) + delta[:, None] * np.asarray(model["w"])[None, :])


def ma_class(value: Any) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    return np.where(~np.isfinite(a), 9, np.where(a < 50.0, -1, np.where(a <= 70.0, 0, 1))).astype(np.int8)


def ma_features(x: np.ndarray, med: np.ndarray, ym: np.ndarray, ys: np.ndarray) -> np.ndarray:
    pred_ma = np.asarray(med)[:, 2]
    pc = ma_class(pred_ma)
    return np.column_stack((
        x, (med - np.asarray(ym)[None, :]) / np.asarray(ys)[None, :],
        (pred_ma - 60.0) / 10.0,
        np.minimum(np.abs(pred_ma - 50.0), np.abs(pred_ma - 70.0)) / 10.0,
        (pc == -1).astype(float), (pc == 0).astype(float), (pc == 1).astype(float),
    ))


def score_gate(model: dict[str, Any], features: np.ndarray, pred_class: np.ndarray) -> np.ndarray:
    pre = model["pre"]
    finite = np.isfinite(features)
    imputed = np.where(finite, features, np.asarray(pre["median"])[None, :])
    z = np.clip((imputed - np.asarray(pre["mean"])[None, :]) /
                np.asarray(pre["sd"])[None, :], -10.0, 10.0)
    if bool(model.get("fallback", False)):
        score = np.full(len(features), float(model["constant_score"]))
    else:
        eta = z @ np.asarray(model["coef"]) + float(model["intercept"])
        score = 1.0 / (1.0 + np.exp(-eta))
    valid = np.isin(np.asarray(pred_class), (-1, 0, 1)) & np.isfinite(features).all(axis=1)
    return np.where(valid, score, np.inf)


def score_head(head: dict[str, Any] | None, x: np.ndarray, pred_ma: np.ndarray) -> np.ndarray:
    if head is None:
        return np.full(len(pred_ma), np.inf)
    pre = head["pre"]
    features = np.column_stack([x[:, 1], pred_ma])
    z = np.clip((np.where(np.isfinite(features), features, np.asarray(pre["median"])[None, :])
                 - np.asarray(pre["mean"])[None, :]) / np.asarray(pre["sd"])[None, :], -10.0, 10.0)
    theta = np.asarray(head["theta"])
    eta = z @ theta[:-1] + theta[-1]
    return np.where(np.isfinite(features).all(axis=1), 1.0 / (1.0 + np.exp(-eta)), np.inf)


def infer(obj: dict[str, Any], heads: dict[str, Any], x_all: np.ndarray,
          input_qc_all: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source = obj["source_model"]
    x = np.asarray(x_all, dtype=float)[idx]
    raw = source_predict(source, x)
    token = str(obj["positions"][3]["base"])
    cache = obj["cache"][token]
    pure = calibrated(cache["cal"], raw)
    adapter = str(obj["gate_models"][3]["adapter"])
    final = pure if adapter == "H" else adapt(cache["adapters"][adapter], x, raw, pure, source)
    pred_class = ma_class(final[:, 2])
    base_score = score_gate(obj["gate_models"][3]["model"], ma_features(x, final, source["ym"], source["ys"]), pred_class)
    dual_score = np.maximum(score_head(heads.get("low"), x, final[:, 2]),
                            score_head(heads.get("high"), x, final[:, 2]))
    valid = np.asarray(input_qc_all, dtype=bool)[idx] & np.isfinite(final).all(axis=1) & np.isin(pred_class, (-1, 0, 1))
    return final, pred_class, base_score, dual_score, valid


def actions(pred: np.ndarray, base_score: np.ndarray, dual_score: np.ndarray,
            valid: np.ndarray, threshold: dict[str, Any]) -> np.ndarray:
    out = np.zeros(len(pred), dtype=bool)
    if bool(threshold.get("all_withhold", False)):
        return out
    normal = threshold.get("normal")
    tail = threshold.get("tail")
    if normal is not None:
        out |= valid & (pred == 0) & np.isfinite(dual_score) & (dual_score <= float(normal))
    if tail is not None:
        out |= valid & (pred != 0) & np.isfinite(base_score) & (base_score <= float(tail))
    return out


def frozen_900_heads() -> dict[tuple[str, int, int, str], dict[str, Any]]:
    out: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for path in sorted(RUNNER.glob("*.json")):
        obj = read_json(path)
        payload = obj["payload"]
        slot = str(payload["slot_key"])
        direction, seed_text, pos_text, head = slot.split("|")
        seed = int(seed_text[1:])
        pos = int(pos_text[1:])
        out[(direction, seed, pos, head)] = {
            "theta": np.asarray(payload["theta"], dtype=float),
            "pre": {k: np.asarray(v, dtype=float) for k, v in payload["preprocess"].items()},
        }
    return out


def thresholds_900() -> dict[tuple[str, int], dict[str, Any]]:
    entries = read_json(SELECTOR)["entries"]
    return {(str(e["direction"]), int(e["seed"])): e for e in entries}


def run() -> dict[str, Any]:
    test_index, subset_mask, part = load_partition()
    # We use only x/input_qc/ids during this forward computation.  DATA15
    # medical values are not passed to infer() or actions().
    forward_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    exact_all = True
    contexts = 0
    input_hashes: dict[str, str] = {}
    for path in [BASE / "DATA15.npz", BASE / "PREFIX_X.npz", PARTITION,
                 SEQ / "run.py", SEQ / "evaluate.py", SELECTOR]:
        input_hashes[str(path)] = sha(path)
    # Bind the recheck to every frozen object/action file actually consumed.
    # This is an input manifest, not a claim that the historical fitting
    # process was rerun.
    frozen_paths = []
    for horizon in (300, 600):
        for direction in DIRECTIONS:
            frozen_paths.append(SEQ / "actions" / f"T{horizon}" / f"{direction}.npz")
            for seed in SEEDS:
                frozen_paths.extend([
                    SEQ / f"T{horizon}" / "models" / direction / "target15" / f"SAHZU_CFULL_seed{seed}.joblib",
                    SEQ / f"T{horizon}" / "dual" / f"{direction}_{seed}.joblib",
                    SEQ / f"T{horizon}" / "frozen" / f"{direction}_{seed}.json",
                ])
    for direction in DIRECTIONS:
        frozen_paths.extend([
            EXT / "actions" / f"{direction}.npz",
        ])
        for seed in SEEDS:
            frozen_paths.append(OLD / "models" / direction / "target15" / f"SAHZU_CFULL_seed{seed}.joblib")
    frozen_paths.extend(RUNNER.glob("*.json"))
    for path in frozen_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        input_hashes[str(path)] = sha(path)

    with np.load(BASE / "DATA15.npz", allow_pickle=False) as archive:
        data15 = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(BASE / "PREFIX_X.npz", allow_pickle=False) as archive:
        horizons = np.asarray(archive["horizon_seconds"], dtype=int)
        prefix_x = np.asarray(archive["x"], dtype=float)
    x_by_horizon = {
        300: prefix_x[int(np.where(horizons == 300)[0][0])],
        600: prefix_x[int(np.where(horizons == 600)[0][0])],
        900: np.asarray(data15["x"], dtype=float),
    }
    qc_by_horizon = {
        300: np.isfinite(x_by_horizon[300]).all(axis=1),
        600: np.isfinite(x_by_horizon[600]).all(axis=1),
        900: np.asarray(data15["input_qc"], dtype=bool),
    }

    heads900 = frozen_900_heads()
    tau900 = thresholds_900()
    # 300/600 objects carry their own test order and frozen per-seed thresholds.
    for horizon in (300, 600):
        for direction in DIRECTIONS:
            saved = np.load(SEQ / "actions" / f"T{horizon}" / f"{direction}.npz", allow_pickle=False)
            for j, seed in enumerate(SEEDS):
                obj = joblib.load(SEQ / f"T{horizon}" / "models" / direction / "target15" / f"SAHZU_CFULL_seed{seed}.joblib")
                dual = joblib.load(SEQ / f"T{horizon}" / "dual" / f"{direction}_{seed}.joblib")
                frozen = read_json(SEQ / f"T{horizon}" / "frozen" / f"{direction}_{seed}.json")
                idx = np.asarray(obj["test"], dtype=int)
                if not np.array_equal(idx, saved["index"]):
                    raise AssertionError((horizon, direction, seed, "test order"))
                # infer() uses only frozen feature inputs and QC; no
                # medical/reference values are used in this call.
                med, pred, base_score, dual_score, valid = infer(
                    obj, dual["heads"][3], x_by_horizon[horizon], qc_by_horizon[horizon], idx
                )
                action = actions(pred, base_score, dual_score, valid, frozen["threshold"])
                checks = {
                    "pred": bool(np.array_equal(pred, saved["pred"][j])),
                    "base_score": bool(np.allclose(base_score, saved["base_score"][j], equal_nan=True, atol=1e-12, rtol=1e-12)),
                    "dual_score": bool(np.allclose(dual_score, saved["dual_score"][j], equal_nan=True, atol=1e-12, rtol=1e-12)),
                    "valid": bool(np.array_equal(valid, saved["valid"][j])),
                    "action": bool(np.array_equal(action, saved["action"][j])),
                }
                exact = all(checks.values())
                exact_all &= exact
                contexts += 1
                for partition_name, local_mask in (("selected60", subset_mask), ("complement141", ~subset_mask)):
                    sel = np.asarray(local_mask, dtype=bool)
                    forward_rows.append({
                        "horizon": horizon, "direction": direction, "seed": seed,
                        "partition": partition_name, "records": int(sel.sum()),
                        "released": int(action[sel].sum()),
                        "valid": int(valid[sel].sum()),
                        "low_pred": int((pred[sel] == -1).sum()),
                        "within_pred": int((pred[sel] == 0).sum()),
                        "high_pred": int((pred[sel] == 1).sum()),
                        "exact_pred": checks["pred"], "exact_base_score": checks["base_score"],
                        "exact_dual_score": checks["dual_score"], "exact_valid": checks["valid"],
                        "exact_action": checks["action"], "exact_all": exact,
                    })
                comparison_rows.append({"horizon": horizon, "direction": direction, "seed": seed,
                                        "exact_pred": checks["pred"], "exact_base_score": checks["base_score"],
                                        "exact_dual_score": checks["dual_score"], "exact_valid": checks["valid"],
                                        "exact_action": checks["action"], "exact_all": exact})

    # The 900-s checkpoint uses the already frozen 15-min source/target
    # objects and external dual heads.  It is a forward recheck from DATA15 x,
    # not a new signal fit.
    for direction in DIRECTIONS:
        saved = np.load(EXT / "actions" / f"{direction}.npz", allow_pickle=False)
        for j, seed in enumerate(SEEDS):
            obj = joblib.load(OLD / "models" / direction / "target15" / f"SAHZU_CFULL_seed{seed}.joblib")
            heads = {}
            for head in ("low", "high"):
                heads[head] = heads900.get((direction, seed, 3, head))
            med, pred, base_score, dual_score, valid = infer(
                obj, heads, x_by_horizon[900], qc_by_horizon[900], np.asarray(obj["test"], dtype=int)
            )
            threshold_row = tau900[(direction, seed)]
            threshold = {
                "normal": threshold_row["normal_threshold"],
                "tail": threshold_row["common_tail_threshold"],
                "all_withhold": threshold_row["common_tail_status"] == "NO_FEASIBLE_COMMON_TAIL_THRESHOLD",
            }
            action = actions(pred, base_score, dual_score, valid, threshold)
            # The 900-s external archive intentionally stores the dual score
            # only for rows eligible for the within-class dual head; tail rows
            # are NaN there.  Compare both the eligibility mask and the finite
            # score values, rather than treating that storage mask as a
            # numerical disagreement.
            archived_dual_valid = np.asarray(saved["dual_valid"][j], dtype=bool)
            # ``infer`` returns the two-head score for every row.  The 900-s
            # archive masks it to the within-prediction rows only.
            dual_mask_match = bool(np.array_equal(archived_dual_valid, pred == 0))
            if dual_mask_match and np.any(archived_dual_valid):
                dual_values_match = bool(np.allclose(
                    dual_score[archived_dual_valid], saved["dual_score"][j][archived_dual_valid],
                    equal_nan=False, atol=1e-12, rtol=1e-12,
                ))
            else:
                dual_values_match = not np.any(archived_dual_valid)
            if dual_mask_match:
                dual_archive_match = dual_values_match and bool(np.isnan(saved["dual_score"][j][~archived_dual_valid]).all())
            else:
                dual_archive_match = False
            checks = {
                "pred": bool(np.array_equal(pred, saved["final_ma_class"][j])),
                "base_score": bool(np.allclose(base_score, saved["lr_score"][j], equal_nan=True, atol=1e-12, rtol=1e-12)),
                "dual_score": dual_archive_match,
                "action": bool(np.array_equal(action, saved["unified_action"][j])),
            }
            exact = all(checks.values())
            exact_all &= exact
            contexts += 1
            for partition_name, local_mask in (("selected60", subset_mask), ("complement141", ~subset_mask)):
                sel = np.asarray(local_mask, dtype=bool)
                forward_rows.append({
                    "horizon": 900, "direction": direction, "seed": seed,
                    "partition": partition_name, "records": int(sel.sum()),
                    "released": int(action[sel].sum()), "valid": int(valid[sel].sum()),
                    "low_pred": int((pred[sel] == -1).sum()), "within_pred": int((pred[sel] == 0).sum()),
                    "high_pred": int((pred[sel] == 1).sum()),
                    "exact_pred": checks["pred"], "exact_base_score": checks["base_score"],
                    "exact_dual_score": checks["dual_score"], "exact_valid": True,
                    "exact_action": checks["action"], "exact_all": exact,
                })
            comparison_rows.append({"horizon": 900, "direction": direction, "seed": seed,
                                    "exact_pred": checks["pred"], "exact_base_score": checks["base_score"],
                                    "exact_dual_score": checks["dual_score"], "exact_valid": True,
                                    "exact_action": checks["action"], "exact_all": exact})

    write_csv(ROOT / "forward_checkpoint_metrics.csv", forward_rows)
    write_csv(ROOT / "forward_exact_comparison.csv", comparison_rows)
    input_hashes_unchanged = all(sha(Path(path)) == digest for path, digest in input_hashes.items())
    result = {
        "schema": "internal_prefix_feature_forward_recheck_v1",
        "status": "PASS" if exact_all else "FAIL",
        "scope": "300/600 saved processed PREFIX_X -> frozen source/target/dual heads -> frozen thresholds -> actions; 900 DATA15 x -> frozen source/target/dual heads -> frozen thresholds -> actions",
        "directions": list(DIRECTIONS), "seeds": list(SEEDS), "horizons": [300, 600, 900],
        "contexts": contexts, "records_per_context": 201,
        "partition": {"selected_records": int(subset_mask.sum()), "complement_records": int((~subset_mask).sum()),
                       "source": str(PARTITION), "uses_labels_or_scores": part["uses_labels_or_scores"]},
        "exact_all_contexts": bool(exact_all),
        "all_prediction_score_validity_action_fields_match_archived": bool(exact_all),
        "input_hashes_unchanged": bool(input_hashes_unchanged),
        "new_fits": 0, "new_thresholds": 0,
        "raw_signal_fitting_repeated": False, "sensor_acquisition_replayed": False,
        "input_hashes": input_hashes,
    }
    write_json(ROOT / "FORWARD_RECHECK.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))
