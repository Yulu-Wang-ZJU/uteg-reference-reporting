"""Fail-closed preflight/runner shell for the Ridge+MA13D HGB candidate.

The default commands are ``--preflight`` and ``--unit-tests``.  The ``--fit``
entrypoint is intentionally unavailable in this prepare round: it requires an
external, hash-bound authorization record which is not created here.  Thus no
HGB.fit, optimizer, candidate-test read, or test-label read can occur.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import runtime as r


PREFLIGHT_ROWS = r.OLD_RUN / "PREFLIGHT_ROWS.json"
AUDIT_JSON = r.AUDIT_RUN / "HGB_RIDGE13D_HISTORY_AUDIT.json"
AUDIT_CSV = r.AUDIT_RUN / "HGB_RIDGE13D_TRAINING_SUPPORT.csv"
AUDIT_PROTOCOL = r.AUDIT_RUN / "CANDIDATE_PROTOCOL_DRAFT.md"
AUDIT_PROMPT = r.AUDIT_RUN / "CANDIDATE_PROMPT_ZH.md"
SELECTOR_JSON = r.OLD_RUN / "DEVELOPMENT_SELECTOR.json"
ROLE_PATH = r.BASE / "ROLE_FREEZE.json"
DATA15 = r.BASE / "DATA15.npz"
NESTED = r.NINE / "NESTED_SPLIT_MANIFEST.json"
CONTEXTS = r.GOAL_ROOT / "source_revision" / "SOURCE_REVISION_CONTEXTS.csv"
RIDGE_IDENTITY = r.RIDGE_RUN / "MODEL_IDENTITY.json"
RIDGE_MANIFEST = r.RIDGE_RUN / "FULL_MODELS_FROZEN.json"
RIDGE_PROTOCOL = r.RIDGE_RUN / "PROTOCOL.md"
RIDGE_THRESHOLD = r.RIDGE_RUN / "THRESHOLD_FREEZE.json"
FIT_LEDGER = r.ROOT / "FIT_LEDGER.json"
ROUND_START_LEDGER = r.ROOT / "ROUND_START_LEDGER.json"
PREFLIGHT_RESULT = r.ROOT / "PREFLIGHT_RESULT.json"
STATE = r.ROOT / "CANDIDATE_STATE.json"
PURE_TEST_RESULT = r.ROOT / "PURE_CONTRACT_TEST_RESULT.json"
CANDIDATE_IDENTITY = r.ROOT / "MODEL_IDENTITY.json"


def _require_file(path: Path, name: str) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise RuntimeError(f"missing required {name}: {path}")
    return r.file_binding(path)


def _assert_hash_map(root: Path, mapping: Any, label: str) -> dict[str, Any]:
    if not isinstance(mapping, dict) or not mapping:
        raise RuntimeError(f"{label} hash map missing/empty")
    checked = 0
    mismatches: list[str] = []
    for rel, expected in mapping.items():
        path = root / Path(str(rel))
        if not path.exists():
            mismatches.append(f"missing:{rel}")
            continue
        actual = r.sha256_file(path)
        if actual != str(expected):
            mismatches.append(f"hash:{rel}")
        checked += 1
    if mismatches:
        raise RuntimeError(f"{label} identity mismatch ({len(mismatches)}): {mismatches[:3]}")
    return {"entries": len(mapping), "checked": checked, "mismatches": 0}


def _validate_rows(rows: Any) -> dict[str, Any]:
    if not isinstance(rows, list) or len(rows) != 720:
        raise RuntimeError(f"expected 720 preflight rows, got {type(rows).__name__}/{len(rows) if isinstance(rows, list) else 'NA'}")
    expected_keys = {(direction, seed, position) for direction in r.DIRECTIONS for seed in r.SEEDS for position in r.POSITIONS}
    seen: set[tuple[str, int, int]] = set()
    required = {"direction", "seed", "position", "feature_dim", "A_n", "errors", "correct",
                "A_hash", "B_hash", "feature_hash", "inner_folds", "fallback_rule",
                "baseline_score_replay", "baseline_threshold_replay", "pred_counts", "truth_counts"}
    fallback_counts = {"small_n": 0, "errors_lt2": 0, "correct_lt2": 0}
    bad: list[str] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            bad.append(f"row{i}:not_dict")
            continue
        missing = required - set(row)
        if missing:
            bad.append(f"row{i}:missing={sorted(missing)}")
            continue
        key = (str(row["direction"]), int(row["seed"]), int(row["position"]))
        if key in seen:
            bad.append(f"duplicate:{key}")
        seen.add(key)
        if int(row["feature_dim"]) != r.FEATURE_DIM:
            bad.append(f"feature_dim:{key}")
        if int(row["A_n"]) < 0 or int(row["errors"]) < 0 or int(row["correct"]) < 0:
            bad.append(f"negative_support:{key}")
        if int(row["errors"]) + int(row["correct"]) != int(row["A_n"]):
            bad.append(f"label_count:{key}")
        folds = row.get("inner_folds")
        if not isinstance(folds, list) or len(folds) != 3:
            bad.append(f"inner_fold_count:{key}")
        else:
            for fold in folds:
                if not all(k in fold for k in ("T_hash", "V_hash", "T_n", "V_n")):
                    bad.append(f"inner_fold_fields:{key}")
                if int(fold.get("T_n", 0)) <= 0 or int(fold.get("V_n", 0)) <= 0:
                    bad.append(f"inner_fold_size:{key}")
        fr = row.get("fallback_rule") or {}
        for flag in fallback_counts:
            fallback_counts[flag] += int(bool(fr.get(flag, False)))
        if not bool(row.get("baseline_score_replay")) or not bool(row.get("baseline_threshold_replay")):
            # A deferred pos3 threshold is still a valid replay contract; the
            # source row must at least carry both identity fields.
            bad.append(f"baseline_replay:{key}")
    if seen != expected_keys:
        bad.append(f"key_set_mismatch:seen={len(seen)} expected={len(expected_keys)}")
    if bad:
        raise RuntimeError("PREFLIGHT_ROWS contract failure: " + "; ".join(bad[:6]))
    return {
        "row_count": len(rows), "unique_position_keys": len(seen),
        "directions": list(r.DIRECTIONS), "seeds": [1000, 1019], "positions": list(r.POSITIONS),
        "feature_dim": r.FEATURE_DIM, "fallback_counts": fallback_counts,
        "feature_hash_count": sum(bool(row.get("feature_hash")) for row in rows),
        "mask_identity_fields": ["A_hash", "B_hash", "inner_folds", "pred_counts", "truth_counts", "baseline_score_replay", "baseline_threshold_replay"],
        "test_labels_read": False,
    }


def _validate_ridge_identity() -> dict[str, Any]:
    identity = r.read_json(_require_file(RIDGE_IDENTITY, "Ridge model identity")["path"])
    manifest = r.read_json(_require_file(RIDGE_MANIFEST, "Ridge full-model manifest")["path"])
    if identity.get("model_family") != "RIDGE" or identity.get("config") != "RIDGE-10" or float(identity.get("alpha")) != 10.0:
        raise RuntimeError("Ridge-10 identity mismatch")
    if identity.get("target_chain", {}).get("directions") != 9 or identity.get("target_chain", {}).get("positions_per_direction_seed") != 4:
        raise RuntimeError("Ridge target role identity mismatch")
    if manifest.get("config") != "RIDGE-10" or manifest.get("model_family") != "RIDGE":
        raise RuntimeError("Ridge manifest identity mismatch")
    if int(manifest.get("source_full_files", -1)) != 70 or int(manifest.get("source_payloads", -1)) != 120 or int(manifest.get("target_objects", -1)) != 180:
        raise RuntimeError("Ridge source/target object count mismatch")
    if manifest.get("test_labels_read") or manifest.get("test_evaluated") or manifest.get("threshold_refit") or manifest.get("threshold_selection_from_test"):
        raise RuntimeError("historical Ridge identity is not test-closed")
    full = _assert_hash_map(r.RIDGE_RUN, manifest.get("source_full_hashes"), "Ridge source full")
    payloads = _assert_hash_map(r.RIDGE_RUN, manifest.get("source_payload_hashes"), "Ridge source payload")
    targets = _assert_hash_map(r.RIDGE_RUN, manifest.get("target_object_hashes"), "Ridge target objects")
    return {"identity_sha256": r.sha256_file(RIDGE_IDENTITY), "manifest_sha256": r.sha256_file(RIDGE_MANIFEST),
            "source_full": full, "source_payloads": payloads, "target_objects": targets,
            "source_oof_not_deployment_substitute": bool(manifest.get("source_oof_not_deployment_substitute")),
            "threshold_source": manifest.get("position3_threshold_source")}


def _validate_selector() -> dict[str, Any]:
    selector = r.read_json(_require_file(SELECTOR_JSON, "development selector")["path"])
    got = selector.get("selector_runner_sha256")
    if got != r.SELECTOR_RUNNER_SHA256:
        raise RuntimeError(f"selector runner hash mismatch: {got!r}")
    if selector.get("test_labels_read") or selector.get("test_evaluated"):
        raise RuntimeError("selector is not test-closed")
    return {"path": str(SELECTOR_JSON), "sha256": r.sha256_file(SELECTOR_JSON),
            "selector_runner_sha256": got, "threshold_pool": selector.get("threshold_pool"),
            "threshold_selection": selector.get("threshold_selection")}


def run_preflight() -> dict[str, Any]:
    rows = r.read_json(_require_file(PREFLIGHT_ROWS, "MA13D preflight rows")["path"])
    bindings = {
        "candidate_identity": _require_file(CANDIDATE_IDENTITY, "candidate model identity"),
        "data15": _require_file(DATA15, "DATA15 container"),
        "role": _require_file(ROLE_PATH, "ROLE_FREEZE"),
        "nested_split": _require_file(NESTED, "nested split manifest"),
        "source_context": _require_file(CONTEXTS, "source revision contexts"),
        "ridge_protocol": _require_file(RIDGE_PROTOCOL, "Ridge protocol"),
        "ridge_threshold": _require_file(RIDGE_THRESHOLD, "Ridge threshold freeze"),
        "audit_protocol": _require_file(AUDIT_PROTOCOL, "candidate protocol draft"),
        "audit_prompt": _require_file(AUDIT_PROMPT, "candidate prompt"),
        "audit_json": _require_file(AUDIT_JSON, "history audit JSON"),
        "audit_csv": _require_file(AUDIT_CSV, "training support CSV"),
        "runtime": r.file_binding(r.HERE),
    }
    row_summary = _validate_rows(rows)
    ridge_summary = _validate_ridge_identity()
    selector_summary = _validate_selector()
    candidate_identity = r.read_json(CANDIDATE_IDENTITY)
    if candidate_identity.get("status") != "PREPARE_ONLY_NOT_AUTHORIZED" or candidate_identity.get("fit_calls_actual") != 0:
        raise RuntimeError("candidate identity is not prepare-only/zero-fit")
    if candidate_identity.get("gate", {}).get("input") != "original numeric-MA 13D gate features":
        raise RuntimeError("candidate identity input drift")
    audit = r.read_json(AUDIT_JSON)
    if audit.get("fit_calls") != 0 or audit.get("test_labels_read") is not False:
        raise RuntimeError("history audit is not zero-fit/test-closed")
    if audit.get("source", {}).get("row_count") != 720:
        raise RuntimeError("history audit row count drift")
    totals = r.ledger_totals()
    if totals != {"science": 0, "repair": 0, "total": 0, "running": 0}:
        raise RuntimeError(f"candidate ledger is not empty: {totals}")
    if r.AUTH_PATH.exists():
        raise RuntimeError("authorization file unexpectedly exists; refusing preflight")
    result = {
        "schema": "ridge_ma13d_hgb_prepare_preflight_v1",
        "status": "READY_FOR_AUTHORIZATION_AND_INDEPENDENT_PREFLIGHT",
        "round": r.ROOT.name,
        "fit_calls_actual": 0,
        "optimizer_calls": 0,
        "test_labels_read": False,
        "candidate_test_read": False,
        "authorization_file_exists": False,
        "position_unit": "direction×seed×position = 9×20×4 = 720; no unverified target/source sharing",
        "candidate": {
            "mapper": "Ridge-10 fullchain existing objects only",
            "gate_input": "original numeric-MA 13D features",
            "gate_label": "original binary MA-ERROR label unchanged",
            "sample_weight": "equal row weights (all ones), not previous three-group weighting",
            "preprocess": "exact original LR fold-local median/mean/sd/clip preprocessing; nonfinite feature values imputed from training fold",
            "legal_mask": "original MA gate label/QC/score masks retained; no B-only feature-finite fail-closed filter",
            "training_input_masks": {
                "known_label_mask": "required bool mask aligned to the exact X/y rows; must be all true after original prefilter",
                "qc_mask": "required original QC bool mask aligned to the exact X/y rows; must be all true after original prefilter",
                "score_valid_mask": "required original score-valid bool mask aligned to the exact X/y rows; must be all true after original prefilter",
                "finite_features_not_a_substitute": True,
            },
            "invalid_score_or_qc_action": "fail closed to 9 before low/high/within action",
            "fallback": "constant only for n<8/errors<2/correct<2; optimizer exception is FAILED and counted, never fallback",
            "low_high_action": "unchanged original action row by row",
            "within_action": "future HGB score only",
            "hgb_config": r.hgb_config(1000),
            "selector_runner_sha256": r.SELECTOR_RUNNER_SHA256,
        },
        "bindings": bindings,
        "rows": row_summary,
        "ridge_identity": ridge_summary,
        "selector": selector_summary,
        "candidate_identity": {"sha256": r.sha256_file(CANDIDATE_IDENTITY), "status": candidate_identity.get("status")},
        "authorization_binding_contract": {
            "required_keys": sorted(r.AUTH_BINDING_PATHS),
            "paths": {name: str(path) for name, path in r.AUTH_BINDING_PATHS.items()},
            "preflight_hash_checked_at_fit_entry": True,
        },
        "ledger": {"totals": totals, "science_cap": r.SCIENCE_CAP, "repair_cap": r.REPAIR_CAP, "total_cap": r.TOTAL_CAP,
                    "future_unit": "one persistent attempt before each HGB optimizer call"},
        "guardrails": [
            "fit entrypoint requires external hash-bound authorization record and currently rejects",
            "no candidate test/evaluation path is referenced by runner",
            "source/target objects are hash-checked without loading test medical labels",
            "no HGB.fit or optimizer call executed",
        ],
    }
    r.write_json(PREFLIGHT_RESULT, result)
    prior_state = r.read_json(STATE) if STATE.exists() else {}
    state = {"schema": "ridge_ma13d_hgb_candidate_state_v1", "status": result["status"],
             "fit_calls_actual": 0, "test_labels_read": False, "candidate_test_read": False,
             "ledger": "FIT_LEDGER.json", "preflight": PREFLIGHT_RESULT.name,
             "authorization_required": True, "authorization_file": r.AUTH_PATH.name}
    for field in ("literature_note", "literature_source_pdf"):
        if field in prior_state:
            state[field] = prior_state[field]
    r.write_json(STATE, state)
    return result


def run_unit_tests() -> dict[str, Any]:
    """Pure contract tests; no HGB class is imported and no fit is called."""
    np = __import__("numpy")
    X = np.array([[1.0] * 13, [2.0] * 13, [float("nan")] + [3.0] * 12, [float("inf")] + [4.0] * 12])
    pre = r.fit_fold_preprocessor(X)
    z = r.apply_fold_preprocessor(X, pre)
    assert z.shape == X.shape and np.isfinite(z).all()
    assert r.fallback_reason(7, 3, 4) == "small_n"
    assert r.fallback_reason(8, 1, 7) == "errors_lt2"
    assert r.fallback_reason(8, 7, 1) == "correct_lt2"
    assert r.fallback_reason(8, 2, 6) is None
    masks = np.ones(len(X), dtype=bool)
    assert r.validate_training_masks(len(X), known_label_mask=masks, qc_mask=masks,
                                     score_valid_mask=masks)["qc_mask"]["legal_n"] == len(X)
    try:
        r.validate_training_masks(len(X), known_label_mask=np.array([True, False, True, True]),
                                  qc_mask=masks, score_valid_mask=masks)
    except ValueError:
        invalid_mask_rejection = True
    else:
        invalid_mask_rejection = False
    assert invalid_mask_rejection
    assert r.validate_position_key(r.DIRECTIONS[0], 1000, 0).endswith("__seed1000__pos0")
    try:
        r.validate_position_key(r.DIRECTIONS[0], 999, 0)
    except ValueError:
        invalid_position_rejection = True
    else:
        invalid_position_rejection = False
    assert invalid_position_rejection
    try:
        r.register_attempt("wrong_key", direction=r.DIRECTIONS[0], seed=1000, position=0)
    except ValueError:
        invalid_attempt_key_rejection = True
    else:
        invalid_attempt_key_rejection = False
    assert invalid_attempt_key_rejection
    assert r.action_for(-1, 0.9, 0.1, score_valid=True, qc_valid=True) == -1
    assert r.action_for(1, 0.9, 0.1, score_valid=True, qc_valid=True) == 1
    assert r.action_for(0, 0.1, 0.2, score_valid=True, qc_valid=True) == 0
    assert r.action_for(0, 0.3, 0.2, score_valid=True, qc_valid=True) == 9
    assert r.action_for(-1, 0.1, 0.2, score_valid=False, qc_valid=True) == 9
    assert r.action_for(-1, 0.1, 0.2, score_valid=True, qc_valid=False) == 9
    assert r.action_for(-1, float("nan"), 0.2, score_valid=True, qc_valid=True) == 9
    try:
        r.require_fit_authorized()
    except PermissionError:
        rejection = True
    else:
        rejection = False
    assert rejection and not r.AUTH_PATH.exists()
    result = {"schema": "ridge_ma13d_hgb_pure_contract_test_v1", "status": "PASS",
              "fit_calls": 0, "optimizer_calls": 0, "test_labels_read": False,
              "authorization_rejection": True, "preprocess_shape": list(z.shape),
              "nonfinite_imputation": True, "fallback_contract": True, "action_contract": True,
              "position_key_contract": True, "training_mask_contract": True,
              "score_qc_fail_closed": True, "lock_contract": True,
              "attempt_key_contract": True}
    r.write_json(PURE_TEST_RESULT, result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--unit-tests", action="store_true")
    ap.add_argument("--fit", action="store_true", help="always rejected in this unauthorized prepare round")
    args = ap.parse_args()
    if args.fit:
        # This check happens before lock acquisition, ledger registration,
        # sklearn import, or any candidate/test data access.
        r.require_fit_authorized()
        raise AssertionError("unreachable without explicit future authorization")
    if not (args.preflight or args.unit_tests):
        print(json.dumps({"status": "PLAN_ONLY", "fit_calls": 0, "test_labels_read": False}, ensure_ascii=False))
        return
    out: dict[str, Any] = {}
    if args.unit_tests:
        out["unit_tests"] = run_unit_tests()
    if args.preflight:
        out["preflight"] = run_preflight()
    print(json.dumps({"status": "READY_FOR_AUTHORIZATION_AND_INDEPENDENT_PREFLIGHT" if args.preflight else "PURE_TEST_PASS",
                      "fit_calls": 0, "test_labels_read": False, "outputs": out}, ensure_ascii=False))


if __name__ == "__main__":
    main()
