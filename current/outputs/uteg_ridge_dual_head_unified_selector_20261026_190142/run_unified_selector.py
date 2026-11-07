"""Two-stage, development-only selector over frozen outer-B artifacts.

Default execution runs synthetic contract checks only.  Real selection is
fail-closed until a separately reviewed REAL_SELECTION_AUTHORIZATION.json is
present.  No optimizer, model loader, DATA15 medical/test payload, or new
threshold grid is used here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
BRIDGE_DIR = RUN_DIR.parent / "uteg_ridge_dual_head_bridge_20261026_180156"
EVALUATION_PATH = BRIDGE_DIR / "BRIDGE_EVALUATION_RESULT.json"
RECORDS_PATH = BRIDGE_DIR / "BRIDGE_DEVELOPMENT_RECORDS.json"
AUTH_PATH = RUN_DIR / "REAL_SELECTION_AUTHORIZATION.json"
SOURCE_BINDINGS_PATH = RUN_DIR / "SOURCE_BINDINGS.json"
OUT_PATH = RUN_DIR / "UNIFIED_SELECTION_RESULT.json"
RECORD_OUT_PATH = RUN_DIR / "UNIFIED_SELECTION_RECORDS.json"
SUMMARY_OUT_PATH = RUN_DIR / "UNIFIED_SELECTION_SUMMARY.json"
MA_CLASSES = {-1, 0, 1}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def _verify_source_bindings() -> dict[str, Any]:
    bindings = _read_json(SOURCE_BINDINGS_PATH)
    for name, item in bindings.items():
        if not isinstance(item, dict) or "path" not in item or "sha256" not in item:
            if name in {"test_read", "fit_units"}:
                continue
            raise RuntimeError(f"invalid source binding: {name}")
        path = (RUN_DIR / item["path"]).resolve()
        if not path.exists() or _sha256(path) != item["sha256"]:
            raise RuntimeError(f"source binding drift: {name}")
    return bindings


def _metric(action: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, Any]:
    action = np.asarray(action, dtype=bool)
    pred = np.asarray([int(row["pred_class"]) for row in rows], dtype=np.int8)
    truth = np.asarray([int(row["truth_class"]) for row in rows], dtype=np.int8)
    ids = np.asarray([str(row["record_id"]) for row in rows], dtype=str)
    if len(action) != len(rows):
        raise ValueError("action/record length mismatch")
    valid = np.isin(pred, list(MA_CLASSES)) & np.isin(truth, list(MA_CLASSES))
    if np.any(action & ~valid):
        raise ValueError("invalid row released")
    low_to_within = action & (truth == -1) & (pred == 0)
    high_to_within = action & (truth == 1) & (pred == 0)
    correct_within = action & (truth == 0) & (pred == 0)
    normal_to_tail = action & (truth == 0) & (pred != 0)
    low_to_high = action & (truth == -1) & (pred == 1)
    high_to_low = action & (truth == 1) & (pred == -1)
    released = int(action.sum())
    return {
        "n": len(rows),
        "released": released,
        "correct": int(np.sum(action & (pred == truth))),
        "agreement": float(np.sum(action & (pred == truth)) / released) if released else None,
        "coverage": float(released / len(rows)) if rows else 0.0,
        "low_to_within": int(low_to_within.sum()),
        "high_to_within": int(high_to_within.sum()),
        "correct_within": int(correct_within.sum()),
        "correct_within_distinct": int(np.unique(ids[correct_within]).size),
        "normal_to_tail": int(normal_to_tail.sum()),
        "low_to_high": int(low_to_high.sum()),
        "high_to_low": int(high_to_low.sum()),
        "released_distinct": int(np.unique(ids[action]).size),
        "normal_correct_distinct": int(np.unique(ids[correct_within]).size),
    }


def _finite_unique(rows: list[dict[str, Any]], field: str, predicate) -> list[float | None]:
    values = sorted({
        float(row[field])
        for row in rows
        if predicate(row) and row.get(field) is not None and np.isfinite(float(row[field]))
    })
    return [None] + values


def _ids_action(rows: list[dict[str, Any]], record_ids: set[str]) -> np.ndarray:
    return np.asarray([str(row["record_id"]) in record_ids for row in rows], dtype=bool)


def _normal_candidate(
    candidate: dict[str, Any], rows: list[dict[str, Any]], lookup: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    # Existing bridge candidates include frozen low/high actions.  Normal stage
    # deliberately retains only the candidate's predicted-within IDs.
    candidate_ids = {
        str(record_id)
        for record_id in candidate.get("release_record_ids", [])
        if str(record_id) in lookup and int(lookup[str(record_id)]["pred_class"]) == 0
    }
    action = _ids_action(rows, candidate_ids)
    metric = _metric(action, rows)
    low_support = sum(int(row["truth_class"]) == -1 for row in rows)
    high_support = sum(int(row["truth_class"]) == 1 for row in rows)
    available_dual = sum(
        int(row["pred_class"]) == 0
        and row.get("dual_score") is not None
        and np.isfinite(float(row["dual_score"]))
        for row in rows
    )
    eligible = bool(
        low_support > 0
        and high_support > 0
        and available_dual > 0
        and metric["low_to_within"] == 0
        and metric["high_to_within"] == 0
        and metric["correct_within_distinct"] >= 5
    )
    return {
        "threshold": candidate.get("threshold"),
        "candidate_index": candidate.get("candidate_index"),
        "candidate_count": candidate.get("n"),
        "normal_record_ids": sorted(candidate_ids),
        "metric": metric,
        "tail_support": {"low": low_support, "high": high_support},
        "dual_available_within_n": available_dual,
        "eligible": eligible,
        "reason": "eligible" if eligible else "normal_support_or_zero_tail_or_distinct_support_failed",
    }


def choose_normal(
    candidates: list[dict[str, Any]], rows: list[dict[str, Any]], lookup: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    evaluated = [_normal_candidate(candidate, rows, lookup) for candidate in candidates]
    eligible = [item for item in evaluated if item["eligible"]]
    selected = max(
        eligible,
        key=lambda item: (
            int(item["metric"]["released"]),
            -(float(item["threshold"]) if item["threshold"] is not None else float("inf")),
        ),
    ) if eligible else None
    return {
        "selected": selected,
        "candidate_count": len(evaluated),
        "eligible_count": len(eligible),
        "candidates": evaluated,
        "status": "NORMAL_SELECTED" if selected is not None else "NORMAL_UNSUPPORTED",
    }


def _tail_action(rows: list[dict[str, Any]], normal_ids: set[str], threshold: float | None) -> np.ndarray:
    action = np.asarray([str(row["record_id"]) in normal_ids for row in rows], dtype=bool)
    if threshold is None:
        return action
    for index, row in enumerate(rows):
        if int(row["pred_class"]) not in (-1, 1):
            continue
        score = row.get("base_score")
        if score is not None and np.isfinite(float(score)) and float(score) <= float(threshold):
            action[index] = True
    return action


def choose_common_tail(rows: list[dict[str, Any]], normal: dict[str, Any]) -> dict[str, Any]:
    selected_normal = normal.get("selected")
    normal_ids = set(selected_normal["normal_record_ids"]) if selected_normal is not None else set()
    candidates: list[dict[str, Any]] = []
    for threshold in _finite_unique(rows, "base_score", lambda row: int(row["pred_class"]) in (-1, 1)):
        action = _tail_action(rows, normal_ids, threshold)
        metric = _metric(action, rows)
        eligible = bool(metric["agreement"] is not None and metric["agreement"] > 0.75 and metric["released_distinct"] >= 5)
        candidates.append({
            "threshold": threshold,
            "action": action,
            "metric": metric,
            "eligible": eligible,
            "reason": "eligible" if eligible else "agreement_or_distinct_support_failed",
        })
    eligible = [item for item in candidates if item["eligible"]]
    selected = max(
        eligible,
        key=lambda item: (
            float(item["metric"]["coverage"]),
            float(item["metric"]["agreement"]),
            -(float(item["threshold"]) if item["threshold"] is not None else float("inf")),
        ),
    ) if eligible else None
    if selected is None:
        action = np.zeros(len(rows), dtype=bool)
        status = "NO_FEASIBLE_COMMON_TAIL_THRESHOLD"
        selected_payload = None
    else:
        action = selected["action"]
        status = "COMMON_TAIL_SELECTED"
        selected_payload = {
            key: value for key, value in selected.items() if key != "action"
        }
    return {
        "selected": selected_payload,
        "status": status,
        "candidate_count": len(candidates),
        "eligible_count": len(eligible),
        "action": action,
        "candidates": [
            {key: value for key, value in item.items() if key != "action"}
            for item in candidates
        ],
    }


def _synthetic_tests() -> dict[str, Any]:
    rows = []
    # Two tail supports and six correct within records; an eighth normal row
    # verifies that complete tie blocks are not split.
    for index in range(8):
        rows.append({
            "record_id": f"S{index}", "pred_class": 0, "truth_class": 0,
            "dual_score": 0.1 if index < 2 else 0.2, "base_score": 0.9,
        })
    rows[0]["truth_class"] = -1
    rows[1]["truth_class"] = 1
    rows.extend([
        {"record_id": "T0", "pred_class": -1, "truth_class": -1, "dual_score": None, "base_score": 0.1},
        {"record_id": "T1", "pred_class": 1, "truth_class": 1, "dual_score": None, "base_score": 0.1},
    ])
    lookup = {row["record_id"]: row for row in rows}
    candidates = [
        {"threshold": None, "candidate_index": 0, "n": len(rows), "release_record_ids": ["T0", "T1"]},
        {"threshold": 0.1, "candidate_index": 1, "n": len(rows), "release_record_ids": [f"S{i}" for i in range(2, 8)] + ["T0", "T1"]},
        {"threshold": 0.2, "candidate_index": 2, "n": len(rows), "release_record_ids": [f"S{i}" for i in range(8)] + ["T0", "T1"]},
    ]
    normal = choose_normal(candidates, rows, lookup)
    if normal["status"] != "NORMAL_SELECTED" or normal["selected"]["metric"]["correct_within_distinct"] != 6:
        raise AssertionError("normal prefix/support contract failed")
    tail = choose_common_tail(rows, normal)
    if tail["status"] != "COMMON_TAIL_SELECTED" or tail["selected"]["metric"]["agreement"] <= 0.75:
        raise AssertionError("common tail contract failed")
    no_high = [dict(row) for row in rows if row["truth_class"] != 1]
    no_high_lookup = {row["record_id"]: row for row in no_high}
    no_high_normal = choose_normal(candidates, no_high, no_high_lookup)
    if no_high_normal["status"] != "NORMAL_UNSUPPORTED":
        raise AssertionError("missing high support must fail closed")
    return {
        "status": "PASS",
        "normal_tie_and_support": "PASS",
        "common_tail_strict_gt75": "PASS",
        "missing_high_failclosed": "PASS",
        "fit_units": 0,
        "test_read": False,
    }


def _load_groups() -> dict[tuple[str, int], list[dict[str, Any]]]:
    evaluation = _read_json(EVALUATION_PATH)
    records = _read_json(RECORDS_PATH)["records"]
    if evaluation.get("test_read") or evaluation.get("test_labels_read"):
        raise RuntimeError("source artifact marked test read")
    lookup = {(str(row["direction"]), int(row["seed"]), str(row["record_id"])): row for row in records}
    if len(lookup) != len(records):
        raise RuntimeError("duplicate source record key")
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in evaluation["rows"]:
        direction, seed = str(row["direction"]), int(row["seed"])
        group = [
            dict(lookup[(direction, seed, str(record_id))])
            for record_id in sorted({str(item["record_id"]) for item in records if item["direction"] == direction and int(item["seed"]) == seed})
        ]
        groups[(direction, seed)] = group
    return groups


def _source_preflight() -> dict[str, Any]:
    bindings = _verify_source_bindings()
    evaluation = _read_json(EVALUATION_PATH)
    records = _read_json(RECORDS_PATH)["records"]
    if evaluation.get("status") != "DEVELOPMENT_EVALUATION_COMPLETE_NO_TEST":
        raise RuntimeError("bridge source is not complete")
    if evaluation.get("test_read") or evaluation.get("test_labels_read"):
        raise RuntimeError("bridge source is marked test-read")
    if len(evaluation.get("rows", [])) != 180 or len(records) != 8400:
        raise RuntimeError("expected 180 seed rows and 8400 records")
    groups = _load_groups()
    if len(groups) != 180:
        raise RuntimeError("source grouping is not 9 directions x 20 seeds")
    return {
        "status": "PREFLIGHT_PASS_NO_REAL_SELECTION",
        "source_binding_names": sorted(bindings),
        "direction_seed_groups": len(groups),
        "record_count": len(records),
        "fit_units": 0,
        "threshold_search": 0,
        "test_read": False,
        "real_selection_executed": False,
    }


def evaluate_real() -> dict[str, Any]:
    if not AUTH_PATH.exists():
        raise RuntimeError("REAL_SELECTION_AUTHORIZATION.json is required; real selection remains closed")
    auth = _read_json(AUTH_PATH)
    if auth.get("status") != "AUTHORIZED" or not auth.get("real_outer_b_selection_authorized"):
        raise RuntimeError("real selection authorization is not valid")
    bindings = auth.get("bindings", {})
    expected_bindings = {
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "protocol_sha256": _sha256(RUN_DIR / "PROTOCOL.md"),
        "state_sha256": _sha256(RUN_DIR / "STATE.json"),
    }
    if any(bindings.get(key) != value for key, value in expected_bindings.items()):
        raise RuntimeError("real selection authorization binding drift")
    _verify_source_bindings()
    groups = _load_groups()
    evaluation = _read_json(EVALUATION_PATH)
    output_rows = []
    record_rows: list[dict[str, Any]] = []
    for (direction, seed), rows in sorted(groups.items()):
        source_row = next(item for item in evaluation["rows"] if item["direction"] == direction and int(item["seed"]) == seed)
        candidates = source_row["selection"]["candidates"]
        lookup = {str(row["record_id"]): row for row in rows}
        normal = choose_normal(candidates, rows, lookup)
        tail = choose_common_tail(rows, normal)
        final_action = tail["action"]
        saved_records = [
            {
                **row,
                "baseline_base_action": bool(row["base_action"]),
                "prior_dual_action": bool(row["candidate_action"]),
                "unified_action": bool(final_action[index]),
                "test_read": False,
            }
            for index, row in enumerate(rows)
        ]
        record_rows.extend({"direction": direction, "seed": int(seed), **row} for row in saved_records)
        output_rows.append({
            "direction": direction,
            "seed": seed,
            "normal": normal,
            "common_tail": {key: value for key, value in tail.items() if key != "action"},
            "metrics": _metric(final_action, rows),
            "baseline_metrics": _metric(np.asarray([bool(row["base_action"]) for row in rows]), rows),
            "prior_dual_metrics": _metric(np.asarray([bool(row["candidate_action"]) for row in rows]), rows),
            "records": saved_records,
            "test_read": False,
        })

    direction_summary: dict[str, Any] = {}
    for direction in sorted({row["direction"] for row in output_rows}):
        direction_rows = [row for row in output_rows if row["direction"] == direction]
        direction_records = [row for row in record_rows if row["direction"] == direction]
        unified = _metric(np.asarray([bool(row["unified_action"]) for row in direction_records]), direction_records)
        baseline = _metric(np.asarray([bool(row["baseline_base_action"]) for row in direction_records]), direction_records)
        prior_dual = _metric(np.asarray([bool(row["prior_dual_action"]) for row in direction_records]), direction_records)
        low_support = len({row["record_id"] for row in direction_records if int(row["truth_class"]) == -1})
        high_support = len({row["record_id"] for row in direction_records if int(row["truth_class"]) == 1})
        normal_distinct = len({
            row["record_id"] for row in direction_records
            if row["unified_action"] and int(row["truth_class"]) == 0 and int(row["pred_class"]) == 0
        })
        target_main = direction in {"HB_to_SA", "HZ+HB_to_SA"}
        main_qualified = bool(
            target_main
            and unified["agreement"] is not None
            and unified["agreement"] > 0.75
            and unified["coverage"] > 0.10
            and unified["low_to_within"] == 0
            and unified["high_to_within"] == 0
            and low_support > 0
            and high_support > 0
            and normal_distinct >= 5
        )
        direction_summary[direction] = {
            "seed_count": len(direction_rows),
            "unified_metrics": unified,
            "baseline_metrics": baseline,
            "prior_dual_metrics": prior_dual,
            "paired_delta_unified_minus_baseline": {
                key: (None if unified.get(key) is None or baseline.get(key) is None else unified[key] - baseline[key])
                for key in ("agreement", "coverage", "low_to_within", "high_to_within", "normal_to_tail", "low_to_high", "high_to_low")
            },
            "support_unique": {"low": low_support, "high": high_support},
            "normal_correct_within_distinct": normal_distinct,
            "active_seed_count": sum(int(row["metrics"]["released"] > 0) for row in direction_rows),
            "main_target_qualification": main_qualified,
            "historical_ge80_ge20": bool(unified["agreement"] is not None and unified["agreement"] >= 0.80 and unified["coverage"] >= 0.20),
            "supervisor_agreement_gt75": bool(unified["agreement"] is not None and unified["agreement"] > 0.75),
            "test_read": False,
            "seed_repetition_is_not_patient_independence": True,
        }
    record_artifact = RECORD_OUT_PATH.name
    summary_artifact = SUMMARY_OUT_PATH.name
    RECORD_OUT_PATH.write_text(json.dumps({
        "schema": "unified_selector_development_records_v1",
        "records": record_rows,
        "record_count": len(record_rows),
        "test_read": False,
        "fit_units": 0,
        "seed_repetition_is_not_patient_independence": True,
    }, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    SUMMARY_OUT_PATH.write_text(json.dumps({
        "schema": "unified_selector_direction_summary_v1",
        "directions": direction_summary,
        "comparison": {
            "original_frozen_baseline": "available in each row",
            "prior_dual_strict": "available in each row",
            "recent_10pct_normal_plus_all_tails": "UNAVAILABLE_NO_FROZEN_RECORD_ARTIFACT",
        },
        "test_read": False,
        "fit_units": 0,
        "seed_repetition_is_not_patient_independence": True,
    }, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    result = {
        "status": "UNIFIED_SELECTION_COMPLETE_DEVELOPMENT_ONLY",
        "rows": output_rows,
        "record_artifact": record_artifact,
        "summary_artifact": summary_artifact,
        "record_count": len(record_rows),
        "direction_summary": direction_summary,
        "comparison_recent_10pct": "UNAVAILABLE_NO_FROZEN_RECORD_ARTIFACT",
        "fit_units": 0,
        "threshold_search": 360,
        "test_read": False,
        "seed_repetition_is_not_patient_independence": True,
    }
    OUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args(argv)
    synthetic = _synthetic_tests()
    (RUN_DIR / "SYNTHETIC_PREFLIGHT_RESULT.json").write_text(
        json.dumps(synthetic, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.preflight:
        result = _source_preflight()
        (RUN_DIR / "PREFLIGHT_RESULT.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if not args.evaluate:
        print(json.dumps({"status": "SYNTHETIC_PREFLIGHT_PASS_REAL_SELECTION_CLOSED", **synthetic}, ensure_ascii=False))
        return 0
    result = evaluate_real()
    print(json.dumps({"status": result["status"], "fit_units": 0, "test_read": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
