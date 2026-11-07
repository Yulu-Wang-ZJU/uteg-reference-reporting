"""Resumable fixed Ridge-10 source/target fullchain runner.

The default command is plan-only.  ``--fit`` is intentionally required and
is expected to be used only after the independent preflight has reviewed the
source identity and this runner hash.  No test labels or test evaluator are
reachable from this module.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

import runtime as r  # cache/thread setup must happen before numerical imports
import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


SEEDS = tuple(range(1000, 1020))
DIRECTIONS = (
    "HZ+HB_to_SA", "SA+HB_to_HZ", "SA+HZ_to_HB",
    "HZ_to_SA", "HB_to_SA", "SA_to_HZ", "HB_to_HZ", "SA_to_HB", "HZ_to_HB",
)
TARGET_BY_DIRECTION = {
    "HZ+HB_to_SA": "SAHZU", "SA+HB_to_HZ": "Huzhou", "SA+HZ_to_HB": "Hubei",
    "HZ_to_SA": "SAHZU", "HB_to_SA": "SAHZU", "SA_to_HZ": "Huzhou",
    "HB_to_HZ": "Huzhou", "SA_to_HB": "Hubei", "HZ_to_HB": "Hubei",
}
SOURCE_KEY_BY_DIRECTION = {
    "HZ+HB_to_SA": "dual_SA", "SA+HB_to_HZ": "dual_HZ", "SA+HZ_to_HB": "dual_HB",
    "HZ_to_SA": "exp_D2_HZ", "HB_to_SA": "single_HB", "SA_to_HZ": "exp_D1_SA",
    "HB_to_HZ": "single_HB", "SA_to_HB": "exp_D1_SA", "HZ_to_HB": "exp_D2_HZ",
}
SOURCE_CENTERS = {
    "dual_SA": ("Huzhou", "Hubei"), "dual_HZ": ("SAHZU", "Hubei"),
    "dual_HB": ("SAHZU", "Huzhou"), "exp_D2_HZ": ("Huzhou",),
    "exp_D1_SA": ("SAHZU",), "single_HB": ("Hubei",),
}
SOURCE_REPRESENTATIVE = {
    "dual_SA": "HZ+HB_to_SA", "dual_HZ": "SA+HB_to_HZ", "dual_HB": "SA+HZ_to_HB",
    "exp_D2_HZ": "HZ_to_SA", "exp_D1_SA": "SA_to_HZ", "single_HB": "HB_to_SA",
}
# This is a new round.  KRR-round wrappers are not admissible; existing
# Ridge-10 numerical objects are accepted only through this round's source
# identity preflight and are wrapped afresh here.
SOURCE_WRAPPER_PRIOR_FREEZE_SHA256S: set[str] = set()
# Original MA outer-B limits are fixed by target and are not re-selected in
# this round.  They are empirical release-error caps, not score cutoffs.
# ``freeze_ma_thresholds.py`` applies the old pure ``summary``/
# ``select_threshold`` contract to finite unique outer-B scores after all
# target objects have been materialized.  Position 3 deploys the one score
# threshold selected from the merged positions 0--2 pool.
OUTER_B_RISK_THRESHOLDS = {"Huzhou": 0.10, "SAHZU": 0.15, "Hubei": 0.15}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _load_data(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: np.asarray(z[key]) for key in z.files}


def _hash_object(obj: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    for key in ("name", "family", "config", "train", "validation", "xm", "xs", "ym", "ys", "coef", "intercept", "train_z", "dual"):
        h.update(key.encode())
        x = obj.get(key)
        if isinstance(x, np.ndarray):
            a = np.ascontiguousarray(x)
            h.update(str(a.dtype).encode()); h.update(repr(a.shape).encode()); h.update(a.tobytes())
        else:
            h.update(json.dumps(x, sort_keys=True, default=str).encode())
    return h.hexdigest()


def _source_rows(data: Mapping[str, np.ndarray]) -> dict[str, list[int]]:
    c = data["centers"].astype(str); qc = np.asarray(data["input_qc"], dtype=bool)
    return {
        "Huzhou": sorted(np.flatnonzero(c == "Huzhou").astype(int).tolist()),
        "SAHZU": sorted(np.flatnonzero(c == "SAHZU").astype(int).tolist()),
        "Hubei": sorted(np.flatnonzero((c == "Hubei") & qc).astype(int).tolist()),
    }


def _expected_source_rows(key: str, rows: Mapping[str, list[int]]) -> list[int]:
    return sorted(sum((list(rows[c]) for c in SOURCE_CENTERS[key]), []))


def _preflight() -> dict[str, Any]:
    path = r.ROOT / "SOURCE_FULL_IDENTITY_PREFLIGHT.json"
    if not path.exists():
        raise RuntimeError("run preflight_ridge10_fullchain.py before this runner")
    result = r.read(path)
    if result.get("status") not in {"PASS", "PASS_SOURCE_REFIT_REQUIRED"}:
        raise RuntimeError(f"source/full preflight not admissible: {result.get('status')}")
    if result.get("test_labels_read") or result.get("test_evaluated"):
        raise RuntimeError("preflight has test access/evaluation marker")
    if int(result.get("source_full_refit_contexts", 0)) > 120:
        raise RuntimeError("source full refit cap exceeded")
    return result


def _write_runner_freeze(preflight: Mapping[str, Any]) -> dict[str, Any]:
    round_start_ledger = r.ROOT / "ROUND_START_LEDGER.json"
    if not round_start_ledger.exists():
        # This snapshot is immutable across pilot/continuation processes;
        # per-attempt rows live separately below ledger/.
        r.write(round_start_ledger.relative_to(r.ROOT), r.read(r.ROOT / "FIT_LEDGER.json"))
    round_start_ledger_sha = r.sha(round_start_ledger)
    threshold_selector_path = r.ROOT / "freeze_ma_thresholds.py"
    old_threshold_source = (r.ROOT.parents[2] / "outputs" / "uteg_nine_direction_ma_20261025_195548"
                            / "code" / "fit_ma_gates.py")
    if not threshold_selector_path.exists() or not old_threshold_source.exists():
        raise RuntimeError("threshold selector/source is missing; refusing to freeze runner")
    payload = {
        "version": "ridge10_fullchain_runner_freeze_v1",
        "created": time.time(),
        "protocol_sha256": r.sha(r.ROOT / "PROTOCOL.md"),
        "prompt_sha256": r.sha(r.ROOT / "EXECUTION_PROMPT_ZH.md"),
        "authorization_sha256": r.sha(r.ROOT / "AUTHORIZATION_RECORD_ZH.md"),
        "outer_threshold_contract_sha256": r.sha(r.ROOT / "OUTER_B_THRESHOLD_CONTRACT.json"),
        "runner_sha256": r.sha(Path(__file__)),
        "runtime_sha256": r.sha(r.ROOT / "runtime.py"),
        "preflight_sha256": r.sha(r.ROOT / "SOURCE_FULL_IDENTITY_PREFLIGHT.json"),
        "threshold_selector_sha256": r.sha(threshold_selector_path),
        "old_threshold_source_sha256": r.sha(old_threshold_source),
        "ledger_sha256": round_start_ledger_sha,
        "round_start_ledger_sha256": round_start_ledger_sha,
        "base_data_sha256": preflight["base_data_sha256"],
        "role_sha256": preflight["role_sha256"],
        "nested_split_sha256": preflight["nested_split_sha256"],
        "source_full_refit_contexts": int(preflight.get("source_full_refit_contexts", 0)),
        "source_full_fit_cap": 120,
        "target_chain_fit_budget": 12600,
        "repair_cap": 280,
        "total_cap": 13000,
        "test_labels_read": False,
        "test_evaluated": False,
    }
    freeze_path = r.ROOT / "RUNNER_FREEZE.json"
    if freeze_path.exists():
        existing = r.read(freeze_path)
        immutable = {k: v for k, v in existing.items() if k != "created"}
        expected = {k: v for k, v in payload.items() if k != "created"}
        if immutable == expected:
            return existing
        # A source/runtime repair may refresh the pre-fit design freeze once.
        # Once any numerical attempt exists, the freeze is immutable.
        if list((r.ROOT / "ledger").glob("*.json")) or existing.get("fit_started"):
            # Keep the original freeze immutable after the first attempt.  A
            # repair-only runner hash is recorded separately; the scientific
            # inputs/contracts below must remain byte-identical.
            existing_no_runner = {k: v for k, v in immutable.items() if k != "runner_sha256"}
            expected_no_runner = {k: v for k, v in expected.items() if k != "runner_sha256"}
            if existing_no_runner == expected_no_runner:
                repair_path = r.ROOT / "REPAIR_RUNTIME_BINDINGS.json"
                history = r.read(repair_path) if repair_path.exists() else {
                    "version": "ridge10_fullchain_repair_runtime_bindings_v1",
                    "test_labels_read": False, "test_evaluated": False, "rows": []}
                history.setdefault("rows", []).append({
                    "validated_runner_sha256": expected["runner_sha256"],
                    "frozen_runner_sha256": existing.get("runner_sha256"),
                    "freeze_sha256": r.sha(freeze_path), "fit_units": 0,
                    "reason": "post-attempt bug repair; freeze scientific contract immutable",
                })
                r.write(repair_path.relative_to(r.ROOT), history)
                return existing
            raise RuntimeError("RUNNER_FREEZE.json drift after fit start; refusing overwrite")
        prior_name = r.ROOT / "RUNNER_FREEZE_PREFIT_SUPERSEDED.json"
        if not prior_name.exists():
            r.write(prior_name.relative_to(r.ROOT), existing)
    r.write("RUNNER_FREEZE.json", payload)
    return payload


def _source_path_from_preflight(direction: str, seed: int, pf: Mapping[str, Any]) -> dict[str, Any]:
    for row in pf["records"]:
        if row["direction"] == direction and int(row["seed"]) == int(seed):
            return row
    raise KeyError((direction, seed))


def _source_payload_path(key: str, seed: int) -> Path:
    return r.ROOT / "source_payloads" / key / f"seed{seed}.joblib"


def _position_outer_prediction(position: Mapping[str, Any], adapter: str) -> np.ndarray:
    """Return the outer-B prediction belonging to exactly one position."""
    outer_pred = position.get("vv")
    if outer_pred is None or adapter not in outer_pred:
        raise RuntimeError(f"missing position-specific outer prediction: adapter={adapter}")
    return np.asarray(outer_pred[adapter], dtype=float)


def _migrate_source_payload_wrapper(out: Path, key: str, seed: int,
                                    current_freeze_sha: str, ledger_sha: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Rebind an existing source wrapper without changing its numerical model.

    The old wrapper is copied byte-for-byte into a versioned directory before
    its run-local replacement is written.  Only wrapper metadata is changed;
    the embedded Ridge-10 object and source OOF arrays are checked by digest.  This
    is deliberately narrower than accepting arbitrary hash drift.
    """
    old = joblib.load(out)
    old_freeze_sha = str(old.get("runner_freeze_sha256", ""))
    if old_freeze_sha == current_freeze_sha:
        return old, None
    if old_freeze_sha not in SOURCE_WRAPPER_PRIOR_FREEZE_SHA256S:
        raise RuntimeError(f"source payload wrapper hash drift: {out}")
    if old.get("ledger_initial_sha256") != ledger_sha:
        raise RuntimeError(f"source payload ledger contract drift: {out}")
    model_digest = _hash_object(old["families"]["RIDGE"]["model"])
    legacy_dir = r.ROOT / "source_payloads_legacy" / old_freeze_sha[:16] / key
    legacy = legacy_dir / out.name
    legacy_dir.mkdir(parents=True, exist_ok=True)
    if not legacy.exists():
        shutil.copy2(out, legacy)
    legacy_obj = joblib.load(legacy)
    if _hash_object(legacy_obj["families"]["RIDGE"]["model"]) != model_digest:
        raise RuntimeError(f"source payload legacy backup/model drift: {out}")
    migrated = copy.deepcopy(old)
    migrated["legacy_runner_freeze_sha256"] = old_freeze_sha
    migrated["runner_freeze_sha256"] = current_freeze_sha
    migrated["validated_with_runner_freeze_sha256"] = current_freeze_sha
    migrated["metadata_migration"] = "zero_fit_source_wrapper_rebind_v1"
    migrated["source_model_digest"] = model_digest
    joblib.dump(migrated, out, compress=3)
    row = {
        "key": key, "seed": int(seed), "path": str(out),
        "legacy_backup": str(legacy),
        "legacy_freeze_sha256": old_freeze_sha,
        "validated_freeze_sha256": current_freeze_sha,
        "model_digest": model_digest,
        "fit_units": 0, "test_labels_read": False,
    }
    return migrated, row


def _source_fit_data(key: str) -> tuple[dict[str, np.ndarray], list[int]]:
    if key in {"exp_D1_SA", "exp_D2_HZ"}:
        data = _load_data(r.EXP / "EXP_DATA.npz")
        # D1 is rows 0:304; D2 is rows 304:399 in the frozen EXP_DATA.
        return data, list(range(304)) if key == "exp_D1_SA" else list(range(304, 399))
    data = _load_data(r.BASE / "DATA15.npz")
    return data, _expected_source_rows(key, _source_rows(data))


def _assert_ridge10_full_model(model: Mapping[str, Any], expected_rows: list[int]) -> None:
    """Fail closed if a source object is not the frozen Ridge-10 contract."""
    if model.get("name") != "RIDGE-10" or model.get("family") != "RIDGE":
        raise RuntimeError(f"source refit returned non-Ridge-10 identity: {model.get('name')!r}/{model.get('family')!r}")
    if model.get("config", {}).get("alpha") != 10:
        raise RuntimeError(f"source refit alpha mismatch: {model.get('config')!r}")
    for key in ("xm", "xs", "ym", "ys", "coef", "intercept"):
        if key not in model or not np.isfinite(np.asarray(model[key], dtype=float)).all():
            raise RuntimeError(f"source refit missing/nonfinite {key}")
    if np.asarray(model["xm"]).shape != (4,) or np.asarray(model["xs"]).shape != (4,):
        raise RuntimeError("source refit source scaler shape mismatch")
    if np.asarray(model["ym"]).shape != (4,) or np.asarray(model["ys"]).shape != (4,):
        raise RuntimeError("source refit target scaler shape mismatch")
    if np.asarray(model["coef"]).shape != (4, 4) or np.asarray(model["intercept"]).shape != (4,):
        raise RuntimeError("source refit Ridge coefficient shape mismatch")
    train = np.asarray(model.get("train", []), dtype=int).tolist()
    if train != list(expected_rows):
        raise RuntimeError("source refit train-row identity mismatch")
    if len(np.asarray(model.get("validation", []))) != 0 or not bool(model.get("success", False)):
        raise RuntimeError("source refit full object is not successful/validation-free")


def _fit_missing_source(key: str, seed: int, bundle: Mapping[str, Any], model_module: Any,
                        expected_rows: list[int]) -> tuple[dict[str, Any], int]:
    data, default_rows = _source_fit_data(key)
    source_folds = bundle.get("source_folds", [])
    fold_rows = sorted({int(i) for fold in source_folds for i in (fold.get("T", []) + fold.get("V", []))})
    train_rows = fold_rows if fold_rows else default_rows
    if len(train_rows) != len(default_rows):
        raise RuntimeError(f"source fold rows incomplete for {key} seed={seed}: {len(train_rows)} != {len(default_rows)}")
    scope = f"SOURCE_RIDGE10_{key}_seed{seed}"
    base_attempt_key = f"source_{key}_seed{seed}_RIDGE10_full"
    prior = sorted((r.ROOT / "ledger").glob(f"{scope}__{base_attempt_key}*.json"))
    if prior:
        retry = 1
        while (r.ROOT / "ledger" / f"{scope}__{base_attempt_key}_retry{retry}.json").exists():
            retry += 1
        attempt_key = f"{base_attempt_key}_retry{retry}"
        r.set_scope(scope, repair=True)
    else:
        attempt_key = base_attempt_key
        r.set_scope(scope)
    # ``key`` is the positional ledger-attempt identifier; keep the source
    # identity in a distinct metadata field so the strict ledger API cannot
    # silently receive the same argument twice.
    attempt = r.register_attempt(attempt_key, "source", 1, source_key=key, seed=seed, source_rows_n=len(train_rows), mapper="RIDGE-10")
    try:
        model = model_module.fit_source(data, train_rows, [], "RIDGE-10", int(seed))
        if not isinstance(model, dict):
            raise RuntimeError("new source fit did not return a model object")
        _assert_ridge10_full_model(model, train_rows)
        out = r.ROOT / "source_full" / key / f"seed{seed}_RIDGE-10_full.joblib"
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out, compress=3)
        r.finish_attempt(attempt, "COMPLETE", output=str(out), model_digest=_hash_object(model))
        return model, 1
    except Exception as exc:
        r.finish_attempt(attempt, "FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


def _build_source_payloads(pf: Mapping[str, Any], base_data: Mapping[str, np.ndarray], model_module: Any,
                           allow_source_refit: bool) -> dict[str, dict[int, Path]]:
    rows = _source_rows(base_data)
    result: dict[str, dict[int, Path]] = {key: {} for key in SOURCE_CENTERS}
    migrations: list[dict[str, Any]] = []
    for key, representative in SOURCE_REPRESENTATIVE.items():
        for seed in SEEDS:
            existing = _source_path_from_preflight(representative, seed, pf)
            bundle_path = Path(existing["bundle_path"])
            bundle = joblib.load(bundle_path)
            full_path = Path(existing["full_path"])
            out = _source_payload_path(key, seed)
            freeze_sha = r.sha(r.ROOT / "RUNNER_FREEZE.json")
            ledger_sha = r.sha(r.ROOT / "ROUND_START_LEDGER.json")
            # Validate a previously materialized payload before considering
            # any source refit.  This prevents a continuation from spending
            # a new fit unit merely to discover a round-start hash mismatch.
            if out.exists():
                old, migration = _migrate_source_payload_wrapper(
                    out, key, seed, freeze_sha, ledger_sha)
                if migration is not None:
                    migrations.append(migration)
                if (old.get("runner_freeze_sha256") != freeze_sha
                        or old.get("ledger_initial_sha256") != ledger_sha):
                    raise RuntimeError(f"source payload round-start contract drift: {out}")
                candidate_path = full_path if bool(existing.get("reusable")) else (r.ROOT / "source_full" / key / f"seed{seed}_RIDGE-10_full.joblib")
                if not candidate_path.exists():
                    raise RuntimeError(f"source payload exists without its Ridge-10 full object: {candidate_path}")
                candidate = joblib.load(candidate_path)
                if _hash_object(old["families"]["RIDGE"]["model"]) != _hash_object(candidate):
                    raise RuntimeError(f"source payload/model drift: {out}")
                result[key][seed] = out
                continue
            model = None
            reusable = bool(existing.get("reusable"))
            if reusable:
                model = joblib.load(full_path)
            elif allow_source_refit:
                model, _ = _fit_missing_source(key, seed, bundle, model_module, _expected_source_rows(key, rows))
                full_path = r.ROOT / "source_full" / key / f"seed{seed}_RIDGE-10_full.joblib"
            else:
                raise RuntimeError(f"source full Ridge-10 needs refit but --allow-source-refit is absent: {key} seed={seed}")
            if model is None:
                raise RuntimeError(f"source model unavailable: {key} seed={seed}")
            out.parent.mkdir(parents=True, exist_ok=True)
            source_rows = _expected_source_rows(key, rows)
            payload = {
                "version": "ridge10_fullchain_source_payload_v1", "target": key, "seed": int(seed),
                "source": source_rows, "source_centers": list(SOURCE_CENTERS[key]),
                "source_selection": "RIDGE", "families": {"RIDGE": {"config": "RIDGE-10", "model": model}},
                "source_OOF": {"RIDGE-10": np.asarray(bundle["source_OOF"]["RIDGE-10"], dtype=float)},
                "source_folds": bundle.get("source_folds", []),
                "source_split_sha256": bundle.get("source_split_sha256", bundle.get("source_split_sha")),
                "source_model_digest": _hash_object(model), "source_model_file": str(full_path),
                "source_model_reused": bool(reusable), "source_model_refit": bool(not reusable),
                "runner_freeze_sha256": r.sha(r.ROOT / "RUNNER_FREEZE.json"),
                "ledger_initial_sha256": r.sha(r.ROOT / "ROUND_START_LEDGER.json"),
                "source_labels": "source-only; omitted third center excluded; Hubei QC-good only",
                "test_labels_read": False, "external_test_evaluated": False,
            }
            joblib.dump(payload, out, compress=3)
            result[key][seed] = out
    r.write("SOURCE_PAYLOAD_MANIFEST.json", {
        "version": "ridge10_fullchain_source_payload_manifest_v1", "test_labels_read": False,
        "contexts": {key: {str(seed): str(path) for seed, path in by_seed.items()} for key, by_seed in result.items()},
        "metadata_migrations": migrations, "fit_units": 0,
    })
    if migrations:
        r.write("SOURCE_PAYLOAD_METADATA_MIGRATION.json", {
            "version": "ridge10_fullchain_source_payload_metadata_migration_v1",
            "status": "COMPLETE_ZERO_FIT", "fit_units": 0,
            "test_labels_read": False, "test_evaluated": False,
            "previous_freeze_sha256": sorted(SOURCE_WRAPPER_PRIOR_FREEZE_SHA256S),
            "current_freeze_sha256": freeze_sha,
            "rows": migrations,
        })
    return result


# ---- MA-ERROR-LR gate implementation (same frozen 50/70 and nested fit) ----
MA_CLASSES = (-1, 0, 1)


def ma_class(value: Any) -> np.ndarray:
    x = np.asarray(value, dtype=float)
    return np.where(~np.isfinite(x), 9, np.where(x < 50, -1, np.where(x <= 70, 0, 1))).astype(np.int8)


def ma_features(x: Any, med: Any, ym: Any, ys: Any) -> np.ndarray:
    x = np.asarray(x, dtype=float); med = np.asarray(med, dtype=float)
    pred_ma = med[:, 2]; pc = ma_class(pred_ma)
    return np.column_stack((x, (med - np.asarray(ym)[None, :]) / np.asarray(ys)[None, :],
                            (pred_ma - 60.0) / 10.0,
                            np.minimum(np.abs(pred_ma - 50.0), np.abs(pred_ma - 70.0)) / 10.0,
                            (pc == -1).astype(float), (pc == 0).astype(float), (pc == 1).astype(float)))


def _preprocess_fit(f: np.ndarray) -> dict[str, Any]:
    f = np.asarray(f, dtype=float); finite = np.isfinite(f)
    median = np.asarray([np.median(f[finite[:, j], j]) if finite[:, j].any() else 0.0 for j in range(f.shape[1])])
    mean = np.asarray([np.mean(f[finite[:, j], j]) if finite[:, j].any() else 0.0 for j in range(f.shape[1])])
    sd = np.asarray([max(np.std(f[finite[:, j], j]), 1e-6) if finite[:, j].any() else 1.0 for j in range(f.shape[1])])
    raw = (np.where(finite, f, median[None, :]) - mean[None, :]) / sd[None, :]
    return {"mean": mean, "sd": sd, "median": median, "ddof": 0, "floor": 1e-6,
            "clip": [-10.0, 10.0], "n": int(len(f)), "p": int(f.shape[1]),
            "missing_fraction": float(np.mean(~finite)) if f.size else 0.0,
            "clip_fraction": float(np.mean(np.abs(raw) > 10.0)) if f.size else 0.0}


def _empty_gate_pre(p: int) -> dict[str, Any]:
    """Exact original zero-row preprocessor contract (zeros/ones, no NaNs)."""
    return {"mean": np.zeros(p), "sd": np.ones(p), "median": np.zeros(p), "ddof": 0,
            "floor": 1e-6, "clip": [-10.0, 10.0], "n": 0, "p": int(p),
            "missing_fraction": 0.0, "clip_fraction": 0.0}


def _fit_ma_gate(f: np.ndarray, pred: np.ndarray, truth: np.ndarray, fit_id: str, seed: int) -> dict[str, Any]:
    known = np.isin(pred, MA_CLASSES) & np.isin(truth, MA_CLASSES)
    legal = np.asarray(f[known], dtype=float)
    y = (pred[known] != truth[known]).astype(float)
    n = len(y); errors = int(y.sum()); correct = n - errors
    constant = float((errors + 0.5) / (n + 1.0))
    out: dict[str, Any] = {"api_version": "ma-error-lr-v1", "id": "MA-ERROR-LR", "family": "MA-ERROR-LR",
        "fit_id": fit_id, "fit_seed": int(seed), "feature_dim": int(f.shape[1]), "regularizer": .1,
        "n": n, "errors": errors, "correct": correct, "error_rate": float(errors / n) if n else None,
        "legal_count": n, "finite_input_rows": int(np.isfinite(f).all(axis=1).sum()),
        "pre": _preprocess_fit(legal) if n else _empty_gate_pre(f.shape[1]),
        "constant_score": constant, "fallback": False, "success": False, "finite": True,
        "status": "not_started", "iterations": 0, "optimizer_steps": 0,
        "coef": np.zeros(f.shape[1]), "intercept": float(math.log(constant / (1.0 - constant))) if 0 < constant < 1 else 0.0,
        "train_pred_class_counts": {str(k): int(np.sum(pred[known] == k)) for k in MA_CLASSES},
        "train_true_class_counts": {str(k): int(np.sum(truth[known] == k)) for k in MA_CLASSES}}
    if n < 8 or errors < 2 or correct < 2:
        out.update(fallback=True, fallback_reason="small_or_one_sided", success=True, status="fallback_constant")
        return out
    pre = out["pre"]; z = np.clip((np.where(np.isfinite(legal), legal, np.asarray(pre["median"])[None, :]) - np.asarray(pre["mean"])[None, :]) / np.asarray(pre["sd"])[None, :], -10.0, 10.0)
    p = z.shape[1]
    def bce(theta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = z @ theta[:p] + theta[p]; prob = expit(eta)
        loss = float(np.mean(np.logaddexp(0.0, eta) - y * eta) + .1 * np.sum(theta[:p] ** 2))
        residual = prob - y
        return loss, np.r_[z.T @ residual / n + .2 * theta[:p], np.mean(residual)]
    try:
        sol = minimize(bce, np.zeros(p + 1), jac=True, method="L-BFGS-B", options={"maxiter": 200, "ftol": 1e-9, "gtol": 1e-9})
        theta = np.asarray(sol.x, dtype=float)
        finite = bool(np.isfinite(theta).all() and np.isfinite(sol.fun))
        if not finite:
            raise FloatingPointError("nonfinite MA gate optimizer result")
        out.update(success=bool(sol.success), finite=True, status=str(sol.message), iterations=int(getattr(sol, "nit", 0)),
                   optimizer_steps=int(getattr(sol, "nit", 0)), objective=float(sol.fun), coef=theta[:-1], intercept=float(theta[-1]))
    except Exception as exc:
        out.update(fallback=True, fallback_reason="optimization_exception:" + type(exc).__name__, success=True, finite=False, status="fallback_exception")
    return out


def score_ma_gate(model: Mapping[str, Any], f: np.ndarray, pred: np.ndarray) -> np.ndarray:
    f = np.asarray(f, dtype=float); pre = model["pre"]
    z = np.clip((np.where(np.isfinite(f), f, np.asarray(pre["median"])[None, :]) - np.asarray(pre["mean"])[None, :]) / np.asarray(pre["sd"])[None, :], -10.0, 10.0)
    if model.get("fallback"):
        s = np.full(len(f), float(model["constant_score"]))
    else:
        coef = np.asarray(model["coef"]); s = expit(z @ coef + float(model["intercept"]))
    # ``ma_class`` uses finite sentinel 9 for unknown/non-QC values.  A
    # finite-value check alone would therefore release invalid rows.  Keep
    # the gate fail-closed: only the three MA classes and fully finite feature
    # rows can receive a finite risk score.
    valid = np.isin(np.asarray(pred), MA_CLASSES) & np.isfinite(f).all(axis=1)
    return np.where(valid, s, np.inf)


def _outer_b_metadata(D: Mapping[str, np.ndarray], B: list[int], pool_order: list[int], pred_ma: np.ndarray,
                     pred_class: np.ndarray, truth_class: np.ndarray,
                     score: np.ndarray) -> dict[str, Any]:
    """Serialize the exact outer-B rows used by the post-fit threshold freeze."""
    ids = np.asarray(D.get("ids", np.arange(len(D["x"]))))
    return {
        "pool_index": [int(v) for v in B],
        "pool_order": [int(v) for v in pool_order],
        "record_id": [str(v) for v in ids[B].tolist()],
        "pred_ma": [float(v) if np.isfinite(v) else None for v in np.asarray(pred_ma, dtype=float)],
        "pred_class": [int(v) for v in np.asarray(pred_class, dtype=np.int8)],
        "truth_class": [int(v) for v in np.asarray(truth_class, dtype=np.int8)],
        "score": [float(v) if np.isfinite(v) else None for v in np.asarray(score, dtype=float)],
        "known_reference_n": int(np.isin(truth_class, MA_CLASSES).sum()),
        "valid_ranking_n": int((np.isin(truth_class, MA_CLASSES)
                                 & np.isin(pred_class, MA_CLASSES)
                                 & np.isfinite(score)).sum()),
        "threshold_frozen": False,
        "threshold_selector": "freeze_ma_thresholds.py::select_threshold",
    }


def _direction_role(role: Mapping[str, Any], direction: str, target: str, source_rows: list[int]) -> dict[str, Any]:
    out = copy.deepcopy(role)
    z = out["roles"][target]
    z["source"] = list(source_rows)
    z["source_spec"] = "+".join(SOURCE_CENTERS[SOURCE_KEY_BY_DIRECTION[direction]]) + str(len(source_rows))
    z["source_contract"] = {"source_centers": list(SOURCE_CENTERS[SOURCE_KEY_BY_DIRECTION[direction]]),
                             "source_n": len(source_rows), "third_center_excluded": [c for c in ("Huzhou", "SAHZU", "Hubei") if c not in set(SOURCE_CENTERS[SOURCE_KEY_BY_DIRECTION[direction]]) | {target}],
                             "qc_rule": "Hubei QC-good only", "test_labels_read": False}
    return out


def _finish_scope(scope: str, status: str, **meta: Any) -> None:
    r.finish_scope(scope, status, **meta)


def fit_target_one(target_module: Any, direction: str, seed: int, source_path: Path, role: Mapping[str, Any], data: dict[str, np.ndarray], source_rows: list[int], protocol_sha: str) -> Path:
    target = TARGET_BY_DIRECTION[direction]
    outpath = r.ROOT / "models" / direction / "target15" / f"{target}_CFULL_seed{seed}.joblib"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    runner_freeze_sha = r.sha(r.ROOT / "RUNNER_FREEZE.json")
    ledger_initial_sha = r.sha(r.ROOT / "ROUND_START_LEDGER.json")
    if outpath.exists():
        obj = joblib.load(outpath)
        if (obj.get("protocol_sha256") != protocol_sha
                or obj.get("runner_freeze_sha256") != runner_freeze_sha
                or obj.get("ledger_initial_sha256") != ledger_initial_sha
                or obj.get("external_test_evaluated") is not False):
            raise RuntimeError(f"existing target object contract mismatch: {outpath}")
        return outpath
    direction_role = _direction_role(role, direction, target, source_rows)
    target_module._load_role = lambda: direction_role
    target_module._load_data = lambda data_path=None, mode="T900": dict(data)
    target_module._freeze_before_fit = lambda *args, **kwargs: None
    m = target_module.m
    spec = target_module._spec(direction_role, target, int(seed), "FULL")
    D0 = target_module._load_data(None)
    D = target_module._hide_test_labels(D0, spec["test"])
    old = joblib.load(source_path)
    src = old["families"]["RIDGE"]["model"]
    pool = spec["pool"]; loc = {v: k for k, v in enumerate(pool)}
    raw = m.predict(src, D["x"][pool])
    d = spec["d"]; As = ["H"] if d == 1 else list(m.ans["A"])
    cache: dict[str, dict] = {}; positions: dict[int, dict] = {}; scores: dict[str, dict] = {}
    OOF = {a: np.full((len(pool), 4), np.nan) for a in As}; ma_gates: dict[int, dict] = {}
    base_key = f"{direction}_{target}_CFULL_seed{seed}"
    existing_scope = sorted((r.ROOT / "ledger").glob(f"{base_key}__*.json"))
    if existing_scope:
        retry = 1
        while list((r.ROOT / "ledger").glob(f"{base_key}_retry{retry}__*.json")):
            retry += 1
        key = f"{base_key}_retry{retry}"
        r.set_scope(key, repair=True)
    else:
        key = base_key
        r.set_scope(key)
    started = time.time()
    try:
        def get(T: list[int]) -> str:
            T = list(map(int, T)); token = ",".join(map(str, sorted(T)))
            if token not in cache:
                ix = [loc[v] for v in T]; rr = raw[ix]; hkey = hashlib.sha256(token.encode()).hexdigest()[:12]
                r.ledger(f"{key}_H_{hkey}", "Huber", train=T)
                cal = m.fit_cal(rr, D["medical"][T], src["ym"], src["ys"]); H = m.calibrated(cal, rr); adapters = {}
                if d == 2:
                    for a in As[1:]:
                        r.ledger(f"{key}_{a}_{hkey}", "adapter", train=T)
                        adapters[a] = m.ans["fit_adapter"](a, D["x"][T], rr, H, D["ci"][T], src)
                cache[token] = {"train": T, "cal": cal, "adapters": adapters}
            return token
        def preds(token: str, I: list[int]) -> dict[str, np.ndarray]:
            I = list(map(int, I)); ca = cache[token]; rr = raw[[loc[v] for v in I]]; H = m.calibrated(ca["cal"], rr)
            return dict(H=H, **{a: m.ans["adapt"](mod, D["x"][I], rr, H, src) for a, mod in ca["adapters"].items()})

        for pos in spec["positions"]:
            code, A, B = int(pos["code"]), list(pos["A"]), list(pos["B"]); al = {v: k for k, v in enumerate(A)}
            cross = {a: np.full((len(A), 4), np.nan) for a in As}; deps = []
            for f in pos["inner"]:
                token = get(f["T"]); pp = preds(token, f["V"])
                for a in As: cross[a][[al[v] for v in f["V"]]] = pp[a]
                deps.append({"T": list(f["T"]), "V": list(f["V"]), "token": token})
            base = get(A); vv = preds(base, B) if B else None
            # Keep each position's outer-B predictions attached to that
            # position.  The legacy loop's ``vv`` variable is overwritten by
            # position 3 (whose B can be empty); consulting that stale value
            # later can turn a valid adapter prediction into ``None``.
            positions[code] = {"A": A, "B": B, "inner": deps, "base": base, "cross": cross, "vv": vv}
            if d == 1:
                selected = "H"; adapter = "H"; selection = {"selected": "H", "adapter": "H", "auxiliary_ci_ordinal_skipped": True}
            else:
                yt_ci = m.labels(D["ci"][A]); Fs, lrs = {}, {}
                for a in As:
                    F = m.gm.features(D["x"][A], cross[a], src); Fs[a] = F; yp = m.labels(m.ci(cross[a]))
                    r.ledger(f"{key}_LR_{code}_{a}", "adapter-selection-LR", train=A)
                    lrs[a] = target_module._fit_error_lr(F, yp, yt_ci)
                    if B:
                        bi = [loc[v] for v in B]; vp = m.labels(m.ci(vv[a])); VF = m.gm.features(D["x"][B], vv[a], src)
                        # Preserve both legacy adapter-selection score heads:
                        # ERROR-LR is fitted above; MARGIN is deterministic
                        # and has no additional fit.  Their parameter counts
                        # and order are part of the frozen tie contract.
                        for gi, gate in enumerate(("ERROR-LR", "MARGIN")):
                            nm = a + "|" + gate
                            adapter_params = 0 if a == "H" else 9 if a.startswith("L") else len(pool) + 1
                            params = adapter_params + (14 if gate == "ERROR-LR" else 0)
                            scores.setdefault(nm, {"score": np.full(len(pool), np.nan), "adapter": a,
                                                    "gate": gate, "params": params,
                                                    "order": As.index(a) * 2 + gi})
                            if gate == "ERROR-LR":
                                value = target_module._lr_score(lrs[a], VF, vp)
                            else:
                                value = -np.minimum(abs(m.ci(vv[a]) + 3), abs(m.ci(vv[a]) - 3))
                            scores[nm]["score"][bi] = value
                positions[code].update(features=Fs, lrs=lrs)
                # The original adapter-selection contract is retained; the
                # endpoint gate itself is the separate frozen MA-ERROR-LR.
                # Selection is resolved after all OOF adapters are populated.
            if B:
                bi = [loc[v] for v in B]
                for a in As: OOF[a][bi] = vv[a]

        if d == 2:
            selection = m.ans["select"](scores, {a: m.labels(m.ci(p)) for a, p in OOF.items()}, m.labels(D["ci"][pool]))
            selected = selection["selected"]["adapter"]
            adapter_by_code: dict[int, str] = {}
            for code, p in positions.items():
                if code == 3:
                    adapter_by_code[code] = selected
                    continue
                yi = m.labels(D["ci"][p["A"]]); rows = []
                for order, a0 in enumerate(As):
                    j = m.ci(p["cross"][a0])
                    rows.append(dict(m.gs["threshold"](-np.minimum(abs(j + 3), abs(j - 3)), m.labels(j), yi),
                                     name=a0, params=0 if a0 == "H" else 9 if a0.startswith("L") else len(p["A"]) + 1,
                                     order=order))
                adapter_by_code[code] = m.gs["choose"](rows)["name"]
        else:
            selection = {"selected": "H", "adapter": "H", "auxiliary_ci_ordinal_skipped": True}; selected = "H"
            adapter_by_code = {code: "H" for code in positions}
        for code, p in positions.items():
            adapter = adapter_by_code[code]
            # MA gate is intentionally independent of CI auxiliary adapter
            # selection and is fit exactly once per target position.
            A, B = p["A"], p["B"]; med_A = p["cross"][adapter]
            f_A = ma_features(D["x"][A], med_A, src["ym"], src["ys"])
            pred_A = ma_class(med_A[:, 2]); truth_A = ma_class(D["medical"][A, 2])
            gate_id = f"{direction}__{target}_seed{seed}_pos{code}"
            r.ledger(f"{key}_MA_GATE_{code}", "MA-ERROR-LR", train=A)
            gate = _fit_ma_gate(f_A, pred_A, truth_A, gate_id, seed)
            if B:
                med_B = _position_outer_prediction(p, adapter)
                pred_ma_B = med_B[:, 2]
                pred_B = ma_class(pred_ma_B)
                truth_B = ma_class(D["medical"][B, 2])
                f_B = ma_features(D["x"][B], med_B, src["ym"], src["ys"])
                score_B = score_ma_gate(gate, f_B, pred_B)
            else:
                med_B = np.empty((0, 4), dtype=float)
                pred_ma_B = np.empty(0, dtype=float)
                pred_B = np.empty(0, dtype=np.int8)
                truth_B = np.empty(0, dtype=np.int8)
                score_B = np.empty(0, dtype=float)
            outer_b = _outer_b_metadata(D, B, [loc[v] for v in B], pred_ma_B, pred_B, truth_B, score_B)
            ma_gates[code] = {"adapter": adapter, "model": gate, "train_A": list(A), "outer_B": list(B),
                              "outerB_risk_cap": float(OUTER_B_RISK_THRESHOLDS[target]),
                              "outerB_threshold": None,
                              "outerB_threshold_source": "deferred_outer_B_select_threshold",
                              "max_risk_is_empirical_error_cap": True,
                              "threshold_is_not_risk_cap": True,
                              "threshold": None,
                              "threshold_source": "deferred_outer_B_select_threshold",
                              "outer_B_data": outer_b}
        out = {"version": "ridge10_fullchain_target_object_v1", "direction": direction, "target": target,
               "arm": "FULL", "budget": "FULL", "seed": int(seed), "source_file": str(source_path),
               "source_sha256": r.sha(source_path), "source_family": "RIDGE", "source_model": src,
               "source_centers": list(SOURCE_CENTERS[SOURCE_KEY_BY_DIRECTION[direction]]), "source_rows_n": len(source_rows),
               "source_rows": list(source_rows), "source_model_digest": _hash_object(src), "pool": pool,
               "test": spec["test"], "test_labels_hidden": True, "external_test_evaluated": False,
               "runner_freeze_sha256": runner_freeze_sha, "ledger_initial_sha256": ledger_initial_sha,
               "cache": cache, "positions": positions, "gate_models": ma_gates, "selection": selection,
               "outerB_threshold_contract": {
                   "target": target, "max_risk": float(OUTER_B_RISK_THRESHOLDS[target]),
                   "max_risk_is_empirical_error_cap": True,
                   "threshold_is_not_risk_cap": True,
                   "threshold_selector": "freeze_ma_thresholds.py::select_threshold",
                   "threshold": None,
                   "threshold_source": "deferred_outer_B_select_threshold",
                   "position3": {"merged_positions": [0, 1, 2], "threshold": None,
                                  "max_risk": float(OUTER_B_RISK_THRESHOLDS[target])},
                   "selection_from_test": False, "threshold_refit": False,
               },
               "final_selection": {"adapter": selected}, "OOF": OOF, "scores": scores,
               "fit_seconds": float(time.time() - started), "protocol_sha256": protocol_sha}
        joblib.dump(out, outpath, compress=3)
        _finish_scope(key, "COMPLETE", output=str(outpath), protocol_sha256=protocol_sha)
        r.guard(); return outpath
    except Exception as exc:
        _finish_scope(key, "FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


def _load_target_legacy() -> Any:
    """Load the read-only legacy wrapper with its historical path contract.

    The wrapper computes ``OLD_MODELS`` as ``r.ROOT.parent / ...``.  In the
    historical runtime ROOT denoted the surrounding outputs/goal directory,
    while this run's ROOT is the new ridge10_fullchain directory.  Temporarily
    exposing the historical-compatible root only during module import fixes
    that path without changing the run-local ledger/cache root used during
    any numerical operation.
    """
    run_root = r.ROOT
    try:
        r.ROOT = run_root.parent
        return r.load("ridge10_fullchain_target_legacy", r.BASE / "train_target_legacy.py")
    finally:
        r.ROOT = run_root


def _run(args: argparse.Namespace) -> None:
    pf = _preflight(); base_data = _load_data(r.BASE / "DATA15.npz")
    protocol_sha = r.sha(r.ROOT / "PROTOCOL.md")
    freeze = _write_runner_freeze(pf)
    if not args.fit:
        print(json.dumps({"status": "PLAN_ONLY", "preflight": pf["status"], "source_refit_contexts": pf.get("source_full_refit_contexts", 0), "target_chain_budget": 12600, "total_cap": 13000}, ensure_ascii=False))
        return
    # A single process owns the persistent ledger for this run.  The runtime
    # lock is released at normal interpreter shutdown and stale locks are
    # recoverable after an interrupted process; concurrent fit processes are
    # rejected before any numerical attempt is registered.
    r.acquire_run_lock("run_ridge10_fullchain.py --fit")
    role = json.loads((r.BASE / "ROLE_FREEZE.json").read_text(encoding="utf-8-sig"))
    target_module = _load_target_legacy()
    target_module._freeze_before_fit = lambda *args, **kwargs: None
    source_payloads = _build_source_payloads(pf, base_data, target_module.m, args.allow_source_refit)
    selected_dirs = list(args.directions) if args.directions else list(DIRECTIONS)
    selected_seeds = [int(x) for x in args.seeds] if args.seeds else list(SEEDS)
    for direction in selected_dirs:
        if direction not in DIRECTIONS: raise ValueError(direction)
        key = SOURCE_KEY_BY_DIRECTION[direction]
        rows = _expected_source_rows(key, _source_rows(base_data))
        for seed in selected_seeds:
            if seed not in SEEDS: raise ValueError(seed)
            fit_target_one(target_module, direction, seed, source_payloads[key][seed], role, base_data, rows, protocol_sha)
            print("TARGET_COMPLETE", direction, seed, flush=True)
    state = r.read(r.ROOT / "RIDGE10_FULLCHAIN_STATE.json")
    totals = r.current_totals(); state["status"] = "TARGET_FITS_IN_PROGRESS" if len(selected_dirs) * len(selected_seeds) < 180 else "TARGET_MODELS_FROZEN_PRE_GATE_REVIEW"
    state["new_round"].update({"fit_units_completed": totals["total"], "science_units": totals["science"], "repair_units": totals["repair"]})
    state["test_labels_read"] = False; state["test_evaluated"] = False
    r.write("RIDGE10_FULLCHAIN_STATE.json", state)
    print(json.dumps({"status": state["status"], "totals": totals, "freeze_sha256": r.sha(r.ROOT / "RUNNER_FREEZE.json")}, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true", help="perform authorized source/target fits")
    ap.add_argument("--allow-source-refit", action="store_true", help="fit missing source full Ridge-10 objects")
    ap.add_argument("--directions", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    _run(ap.parse_args())


if __name__ == "__main__":
    main()
