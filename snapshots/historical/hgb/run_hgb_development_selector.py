"""Development-only outer-B selector for the frozen HGB gate artifacts.

This adapter reuses the already reviewed selector pure functions, but loads
the Ridge-10+MA13D-HGB artifacts from this run.  It never opens DATA15
medical/test arrays, never fits, and never changes a threshold after this
development replay.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

import runtime as r
import run_hgb_ma13d_driver as hgb

# The reviewed selector is a pure outer-B replay, but its module is packaged
# beside the old weighted-gate runner and that runner expects a few historical
# runtime aliases at import time.  Bind those aliases to the current Ridge
# development runtime before importing it.  No old training/evaluation entry
# point is called; the aliases only let the already-reviewed pure functions
# load without importing a second runtime module.
if not hasattr(r, "sha256"):
    r.sha256 = r.sha256_file
if not hasattr(r, "FROZEN_ROOT"):
    r.FROZEN_ROOT = r.GOAL_ROOT / "ma_tail_weight_gate_dedup_20261026_140315"
if not hasattr(r, "BASE_ROOT"):
    r.BASE_ROOT = r.RIDGE_RUN
if not hasattr(r, "DATA_ROOT"):
    r.DATA_ROOT = r.BASE
if not hasattr(r, "DATA_PATH"):
    r.DATA_PATH = r.BASE / "DATA15.npz"
if not hasattr(r, "ROLE_PATH"):
    r.ROLE_PATH = r.BASE / "ROLE_FREEZE.json"
if not hasattr(r, "AGREEMENT_MIN_STRICT"):
    r.AGREEMENT_MIN_STRICT = 0.75
if not hasattr(r, "COVERAGE_MIN_STRICT"):
    r.COVERAGE_MIN_STRICT = 0.20
sys.path.insert(0, str(r.OLD_RUN))
_selector_source = r.OLD_RUN / "selector_weighted_gate.py"
_selector_source_sha = r.sha256_file(_selector_source)
if _selector_source_sha.lower() != str(r.SELECTOR_RUNNER_SHA256).lower():
    raise RuntimeError(
        f"reviewed selector source hash drift: {_selector_source_sha} != {r.SELECTOR_RUNNER_SHA256}"
    )
import selector_weighted_gate as selector


ROOT = r.ROOT
ARMS = ("BASE_FROZEN", "BASE_JOINT", "HGB_JOINT")


def _load_context(direction: str, seed: int, manifest: dict[Path, str]) -> dict[str, Any]:
    base_path, base_sha = hgb._object_path(manifest, direction, seed)
    obj = joblib.load(base_path)
    artifact_path = hgb._artifact_path(direction, seed)
    if not artifact_path.exists():
        raise RuntimeError(f"HGB artifact missing: {artifact_path}")
    artifact = joblib.load(artifact_path)
    if artifact.get("candidate") != "RIDGE-10+MA13D-HGB" or artifact.get("base_object_sha256") != base_sha:
        raise RuntimeError(f"HGB artifact identity mismatch: {artifact_path}")
    parts: list[dict[str, Any]] = []
    for pos in (0, 1, 2):
        saved = obj["gate_models"][pos]["outer_B_data"]
        gate = artifact["gates"].get(str(pos))
        if not isinstance(gate, dict):
            raise RuntimeError(f"missing HGB gate pos{pos}")
        base_score = np.asarray(saved["score"], dtype=float)
        candidate_score = np.asarray(gate["candidate_score"], dtype=float)
        pred = np.asarray(saved["pred_class"], dtype=np.int8)
        truth = np.asarray(saved["truth_class"], dtype=np.int8)
        index = np.asarray(saved["pool_index"], dtype=np.int64)
        record_id = np.asarray(saved["record_id"], dtype=str)
        if any(len(x) != len(base_score) for x in (candidate_score, pred, truth, index, record_id)):
            raise RuntimeError(f"HGB outer-B array length mismatch pos{pos}")
        gate_pred = np.asarray(gate["outer_pred"], dtype=np.int8)
        gate_truth = np.asarray(gate["outer_truth"], dtype=np.int8)
        gate_ids = np.asarray(gate["outer_record_id"], dtype=str)
        if not (np.array_equal(gate_pred, pred) and np.array_equal(gate_truth, truth)
                and np.array_equal(gate_ids, record_id)):
            raise RuntimeError(f"HGB outer-B identity mismatch pos{pos}")
        if not np.array_equal(base_score[pred != 0], candidate_score[pred != 0]):
            raise RuntimeError(f"HGB changed non-within score pos{pos}")
        if len(set(index.tolist())) != len(index):
            raise RuntimeError(f"duplicate outer-B index pos{pos}")
        parts.append({"base_score": base_score, "candidate_score": candidate_score,
                      "pred": pred, "truth": truth, "index": index, "record_id": record_id})
    merged = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    if len(set(merged["index"].tolist())) != len(merged["index"]):
        raise RuntimeError("merged HGB outer-B indices are not unique")
    threshold = obj["gate_models"][3].get("threshold")
    if not isinstance(threshold, dict) or "threshold" not in threshold:
        raise RuntimeError("pos3 frozen threshold missing")
    return {"direction": direction, "seed": int(seed), "target": hgb.TARGET_BY_DIRECTION[direction],
            "base_path": str(base_path), "base_sha256": base_sha,
            "base_tau": threshold.get("threshold"), "rows": merged,
            "threshold_source": threshold.get("threshold_source")}


def _arm_row(context: dict[str, Any], arm: str) -> dict[str, Any]:
    rows = context["rows"]
    base_score = rows["base_score"]
    score = base_score if arm in {"BASE_FROZEN", "BASE_JOINT"} else rows["candidate_score"]
    base_tau = context["base_tau"]
    if arm == "BASE_FROZEN":
        selected = {"threshold": base_tau, "strict_eligible": False, "reason": "frozen_base_threshold"}
        action = selector._mixed_action(base_score, score, rows["pred"], base_tau, base_tau)
        selection = {"selected": selected, "reason": "frozen_base_threshold", "candidate_count": 1, "eligible_count": 0}
    else:
        selection = selector._select_within_threshold(
            base_score, score, rows["pred"], rows["truth"], rows["record_id"], base_tau)
        chosen = selection["selected"]
        tau = chosen["threshold"] if chosen is not None else None
        action = selector._mixed_action(base_score, score, rows["pred"], base_tau, tau)
    metrics = selector._metrics(action, rows["pred"], rows["truth"], rows["record_id"])
    return {"arm": arm, "selection": selection, "metrics": metrics,
            "same_coverage": selector._same_coverage_diagnostic(
                base_score, score, rows["pred"], rows["truth"], base_tau),
            "action": np.asarray(action, dtype=bool), "pred": rows["pred"],
            "truth": rows["truth"], "record_id": rows["record_id"],
            "test_labels_read": False}


def _promotion(summaries: dict[str, dict[str, Any]], arm: str) -> dict[str, Any]:
    single = [d for d in r.DIRECTIONS if "+" not in d]
    dual = [d for d in r.DIRECTIONS if "+" in d]
    single_ok = [d for d in single if summaries[d][arm]["strict_eligible"]]
    dual_ok = [d for d in dual if summaries[d][arm]["strict_eligible"]]
    return {"strict_agreement_operator": ">", "strict_coverage_operator": ">",
            "single_eligible_directions": single_ok, "dual_eligible_directions": dual_ok,
            "goal_evidence_ready": bool(single_ok and dual_ok),
            "test_evaluation_started": False}


def run_selector() -> dict[str, Any]:
    selector.synthetic_selector_contract_tests()
    manifest = hgb._manifest_paths()
    rows: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        d: {arm: [] for arm in ARMS} for d in r.DIRECTIONS}
    actions: dict[str, dict[str, list[np.ndarray]]] = {
        d: {arm: [] for arm in ARMS} for d in r.DIRECTIONS}
    for direction in r.DIRECTIONS:
        for seed in r.SEEDS:
            context = _load_context(direction, int(seed), manifest)
            for arm in ARMS:
                row = _arm_row(context, arm)
                row.update({"direction": direction, "seed": int(seed), "target": context["target"],
                            "base_threshold": context["base_tau"],
                            "base_sha256": context["base_sha256"]})
                rows.append(row)
                grouped[direction][arm].append(row)
                actions[direction][arm].append(row["action"])
    summaries: dict[str, dict[str, Any]] = {d: {} for d in r.DIRECTIONS}
    for direction in r.DIRECTIONS:
        for arm in ARMS:
            summaries[direction][arm] = selector._aggregate(grouped[direction][arm])
    promotion = {arm: _promotion(summaries, arm) for arm in ARMS}
    action_path = ROOT / "HGB_DEVELOPMENT_SELECTOR_ACTIONS.npz"
    np.savez_compressed(action_path, **{
        f"{d}__{arm}": np.concatenate(actions[d][arm]) for d in r.DIRECTIONS for arm in ARMS})
    selector_path = ROOT / "HGB_DEVELOPMENT_SELECTOR.json"
    rows_path = ROOT / "HGB_DEVELOPMENT_SELECTOR_ROWS.json"
    fit_report = r.read_json(ROOT / "HGB_FIT_RUN_REPORT.json")
    out = {
        "schema": "ridge_ma13d_hgb_development_selector_v1",
        "candidate": "RIDGE-10+MA13D-HGB",
        "arms": list(ARMS), "directions": list(r.DIRECTIONS), "seeds": list(r.SEEDS),
        "threshold_pool": "merged outer-B positions0-2 only; finite unique predicted-within scores plus None; complete ties",
        "threshold_selection": "eligible first, max pooled coverage, max pooled agreement, smallest numeric threshold",
        "strict_contract": {"agreement": ">0.75", "coverage": ">0.20", "tail_to_within": {"low": 0, "high": 0},
                            "correct_within_distinct_records": 5, "both_tail_support": True},
        "same_coverage": "descriptive only; fractional ties never enter actions or promotion",
        "test_labels_read": False, "test_evaluation_started": False,
        "science_units": int(r.ledger_totals()["science"]), "repair_units": int(r.ledger_totals()["repair"]),
        "total_units": int(r.ledger_totals()["total"]),
        "fit_report_sha256": r.sha256_file(ROOT / "HGB_FIT_RUN_REPORT.json"),
        "candidate_runner_sha256": fit_report.get("driver_sha256"),
        "reviewed_selector_source_sha256": _selector_source_sha,
        "selector_runner_sha256": r.sha256_file(Path(__file__)),
        "summaries": summaries, "promotion": promotion,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    r.write_json(selector_path, out)
    r.write_json(rows_path, {"schema": "ridge_ma13d_hgb_development_selector_rows_v1",
                             "test_labels_read": False, "rows": rows})
    summary_path = ROOT / "HGB_DEVELOPMENT_NINE_DIRECTION_SUMMARY.csv"
    fields = ["direction", "arm", "n", "released", "correct", "agreement", "coverage",
              "low_to_within", "high_to_within", "correct_within", "correct_within_distinct_records",
              "strict_eligible", "seed_count", "released_seed_count"]
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for direction in r.DIRECTIONS:
            for arm in ARMS:
                row = {"direction": direction, "arm": arm, **summaries[direction][arm]}
                writer.writerow({k: row.get(k) for k in fields})
    r.write_json(ROOT / "HGB_DEVELOPMENT_SELECTOR_STATUS.json", {
        "status": "DEVELOPMENT_SELECTOR_COMPLETE_NO_TEST", "test_labels_read": False,
        "test_evaluation_started": False, "selector": str(selector_path.name),
        "rows": str(rows_path.name), "actions": str(action_path.name),
        "summary": str(summary_path.name), "promotion": promotion,
        "science_units": int(r.ledger_totals()["science"]), "repair_units": int(r.ledger_totals()["repair"]),
        "total_units": int(r.ledger_totals()["total"]),
    })
    return out


if __name__ == "__main__":
    print(json.dumps(run_selector(), ensure_ascii=False))
