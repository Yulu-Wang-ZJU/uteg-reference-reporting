"""Development-only replay selector for the weighted MA-error gate.

This file never imports an evaluator and never opens DATA15 ``medical``.  It
uses only the saved target objects, the candidate gate artifacts, and the
held-out outer-B rows.  Thresholds are descriptive development choices; no
test action is produced here.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np

import runtime as r
import run_weighted_gate as w


BASE_MANIFEST = w.BASE_MANIFEST
STRICT_AGREEMENT = float(r.AGREEMENT_MIN_STRICT)
STRICT_COVERAGE = float(r.COVERAGE_MIN_STRICT)


def _valid_mask(score: np.ndarray, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    pred = np.asarray(pred, dtype=np.int8)
    truth = np.asarray(truth, dtype=np.int8)
    return np.isfinite(score) & np.isin(pred, r.MA_CLASSES) & np.isin(truth, r.MA_CLASSES)


def _release_mask(score: np.ndarray, pred: np.ndarray, tau: float | None) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    pred = np.asarray(pred, dtype=np.int8)
    if tau is None:
        return np.zeros(len(score), dtype=bool)
    return np.isfinite(score) & np.isin(pred, r.MA_CLASSES) & (score <= float(tau))


def _mixed_action(base_score: np.ndarray, score: np.ndarray, pred: np.ndarray,
                  base_tau: float | None, within_tau: float | None) -> np.ndarray:
    """Keep original low/high actions; only replace predicted-within action."""
    base_score = np.asarray(base_score, dtype=float)
    score = np.asarray(score, dtype=float)
    pred = np.asarray(pred, dtype=np.int8)
    original_nonwithin = _release_mask(base_score, pred, base_tau) & (pred != 0)
    candidate_within = _release_mask(score, pred, within_tau) & (pred == 0)
    return original_nonwithin | candidate_within


def _threshold_candidates(score: np.ndarray, candidate_mask: np.ndarray | None = None) -> list[float | None]:
    """All-withhold plus finite unique *within* scores; ties use <=."""
    score = np.asarray(score, dtype=float)
    if candidate_mask is None:
        candidate_mask = np.ones(len(score), dtype=bool)
    finite = score[np.asarray(candidate_mask, dtype=bool) & np.isfinite(score)]
    values = sorted({float(v) for v in finite})
    return [None] + values


def _metrics(action: np.ndarray, pred: np.ndarray, truth: np.ndarray,
             record_id: np.ndarray) -> dict[str, Any]:
    action = np.asarray(action, dtype=bool)
    pred = np.asarray(pred, dtype=np.int8)
    truth = np.asarray(truth, dtype=np.int8)
    ids = np.asarray(record_id, dtype=str)
    if not (len(action) == len(pred) == len(truth) == len(ids)):
        raise RuntimeError("selector arrays have inconsistent lengths")
    valid = np.isin(pred, r.MA_CLASSES) & np.isin(truth, r.MA_CLASSES)
    if np.any(action & ~valid):
        raise RuntimeError("invalid row released by selector")
    released = int(action.sum())
    correct = int(np.sum(action & (pred == truth)))
    low_to_within = action & (truth == -1) & (pred == 0)
    high_to_within = action & (truth == 1) & (pred == 0)
    correct_within = action & (truth == 0) & (pred == 0)
    tail_support = {"low": int(np.sum(truth == -1)), "high": int(np.sum(truth == 1))}
    correct_ids = np.unique(ids[correct_within]).tolist()
    agreement = float(correct / released) if released else None
    coverage = float(released / len(truth)) if len(truth) else 0.0
    strict = bool(
        agreement is not None and agreement > STRICT_AGREEMENT and coverage > STRICT_COVERAGE
        and int(low_to_within.sum()) == 0 and int(high_to_within.sum()) == 0
        and len(correct_ids) >= 5 and tail_support["low"] > 0 and tail_support["high"] > 0
    )
    return {
        "n": int(len(truth)), "released": released, "correct": correct,
        "agreement": agreement, "coverage": coverage,
        "low_to_within": int(low_to_within.sum()), "high_to_within": int(high_to_within.sum()),
        "correct_within": int(correct_within.sum()),
        "correct_within_distinct_records": int(len(correct_ids)),
        "tail_support": tail_support, "strict_eligible": strict,
        "release_record_ids": np.unique(ids[action]).tolist(),
        "correct_within_record_ids": correct_ids,
    }


def _select_within_threshold(base_score: np.ndarray, score: np.ndarray, pred: np.ndarray,
                             truth: np.ndarray, record_id: np.ndarray,
                             base_tau: float | None) -> dict[str, Any]:
    candidates = _threshold_candidates(score, np.asarray(pred, dtype=np.int8) == 0)
    evaluated: list[dict[str, Any]] = []
    for tau in candidates:
        action = _mixed_action(base_score, score, pred, base_tau, tau)
        row = _metrics(action, pred, truth, record_id)
        row.update({"threshold": tau, "candidate_index": len(evaluated)})
        evaluated.append(row)
    eligible = [row for row in evaluated if row["strict_eligible"]]
    if eligible:
        # None is deterministic and always sorts after a finite threshold if
        # an otherwise impossible tie ever survives the primary keys.
        selected = sorted(
            eligible,
            key=lambda row: (-float(row["coverage"]), -float(row["agreement"]),
                             float("inf") if row["threshold"] is None else float(row["threshold"])),
        )[0]
        reason = "strict_eligible"
    else:
        selected = None
        reason = "no_feasible_strict_threshold"
    return {
        "selected": selected,
        "reason": reason,
        "candidate_count": len(evaluated),
        "eligible_count": len(eligible),
        "candidates": evaluated,
    }


def _same_coverage_diagnostic(base_score: np.ndarray, score: np.ndarray, pred: np.ndarray,
                              truth: np.ndarray, base_tau: float | None) -> dict[str, Any]:
    """Descriptive same-k comparison; never feeds selection or actions."""
    base_action = _release_mask(base_score, pred, base_tau)
    within = (pred == 0)
    k = int(np.sum(base_action & within))
    if k == 0:
        return {"k": 0, "status": "NOT_EVALUABLE", "base": None, "candidate": None}

    def counts(values: np.ndarray) -> dict[str, Any]:
        values = np.asarray(values, dtype=float)
        keep = within & np.isfinite(values)
        ordered = np.argsort(np.where(keep, values, np.inf), kind="mergesort")
        kept = ordered[keep[ordered]]
        if len(kept) == 0:
            return {"released_expected": None}
        boundary = values[kept[min(k - 1, len(kept) - 1)]]
        below = keep & (values < boundary)
        tie = keep & (values == boundary)
        n_below = int(below.sum())
        n_tie = int(tie.sum())
        tie_weight = float((k - n_below) / n_tie) if n_tie else 0.0
        row_weight = below.astype(float) + tie.astype(float) * tie_weight
        # This is an expected same-k diagnostic only.  It is deliberately
        # fractional at a tie and never feeds an action or qualification.
        return {
            "release_expected": float(row_weight.sum()),
            "low_to_within_expected": float(np.sum(row_weight * (truth == -1))),
            "high_to_within_expected": float(np.sum(row_weight * (truth == 1))),
            "correct_within_expected": float(np.sum(row_weight * (truth == 0))),
            "tie_block_boundary": float(boundary), "below_n": n_below,
            "tie_n": n_tie, "tie_weight": tie_weight,
        }

    return {"k": k, "status": "DESCRIPTIVE_ONLY", "base": counts(base_score), "candidate": counts(score)}


def _load_base_manifest() -> dict[str, str]:
    manifest = r.read_json(BASE_MANIFEST)
    if manifest.get("test_evaluated") or manifest.get("test_labels_read"):
        raise RuntimeError("base manifest is not test blind")
    out = {}
    for rel, expected in manifest.get("target_object_hashes", {}).items():
        path = w._path_from_manifest(rel)
        got = r.sha256(path)
        if got.lower() != str(expected).lower():
            raise RuntimeError(f"base object hash drift: {rel}")
        out[str(path)] = str(expected).lower()
    if len(out) != 180:
        raise RuntimeError("expected 180 frozen base objects")
    return out


def _load_context(direction: str, seed: int, manifest: Mapping[str, str]) -> dict[str, Any]:
    base_path, base_sha = w._base_path_for(direction, seed, manifest)
    obj = joblib.load(base_path)
    artifact_path = w._artifact_path(direction, seed)
    if not artifact_path.exists():
        raise RuntimeError(f"candidate artifact missing: {artifact_path}")
    artifact = joblib.load(artifact_path)
    if artifact.get("candidate") != "MA-TAIL-BALANCED-ERROR-LR" or artifact.get("base_object_sha256") != base_sha:
        raise RuntimeError(f"candidate artifact identity mismatch: {artifact_path}")
    parts: list[dict[str, Any]] = []
    for pos in (0, 1, 2):
        gm = obj["gate_models"][pos]
        saved = gm["outer_B_data"]
        gate = artifact["gates"].get(str(pos))
        if not isinstance(gate, Mapping):
            raise RuntimeError(f"missing candidate gate pos{pos}")
        n = len(saved["score"])
        arrays = {
            "base_score": np.asarray(saved["score"], dtype=float),
            "candidate_score": np.asarray(gate["candidate_score"], dtype=float),
            "pred": np.asarray(saved["pred_class"], dtype=np.int8),
            "truth": np.asarray(saved["truth_class"], dtype=np.int8),
            "index": np.asarray(saved["pool_index"], dtype=np.int64),
            "record_id": np.asarray(saved["record_id"], dtype=str),
        }
        if any(len(value) != n for value in arrays.values()):
            raise RuntimeError(f"outer-B candidate array length mismatch pos{pos}")
        # The candidate is allowed to alter only predicted-within rows.
        pred = arrays["pred"]
        expected_nonwithin = arrays["base_score"][pred != 0]
        actual_nonwithin = arrays["candidate_score"][pred != 0]
        if not np.array_equal(expected_nonwithin, actual_nonwithin):
            raise RuntimeError(f"candidate changed non-within score pos{pos}")
        if len(set(arrays["index"].tolist())) != n:
            raise RuntimeError(f"duplicate outer-B index pos{pos}")
        parts.append(arrays)
    merged = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    if len(set(merged["index"].tolist())) != len(merged["index"]):
        raise RuntimeError("merged outer-B indices are not unique")
    threshold = obj["gate_models"][3].get("threshold")
    if not isinstance(threshold, Mapping) or "threshold" not in threshold:
        raise RuntimeError("pos3 frozen threshold missing")
    return {
        "direction": direction, "seed": int(seed), "target": r.TARGET_BY_DIRECTION[direction],
        "base_path": str(base_path), "base_sha256": base_sha,
        "base_tau": threshold.get("threshold"), "rows": merged,
        "threshold_source": threshold.get("threshold_source"),
    }


def _arm_row(context: Mapping[str, Any], arm: str) -> dict[str, Any]:
    rows = context["rows"]
    base_score = rows["base_score"]
    score = base_score if arm == "BASE_JOINT" or arm == "BASE_FROZEN" else rows["candidate_score"]
    base_tau = context["base_tau"]
    if arm == "BASE_FROZEN":
        selected = {"threshold": base_tau, "strict_eligible": False, "reason": "frozen_base_threshold"}
        action = _mixed_action(base_score, score, rows["pred"], base_tau, base_tau)
        selection = {"selected": selected, "reason": "frozen_base_threshold", "candidate_count": 1, "eligible_count": 0}
    else:
        selection = _select_within_threshold(base_score, score, rows["pred"], rows["truth"], rows["record_id"], base_tau)
        chosen = selection["selected"]
        if chosen is None:
            action = _mixed_action(base_score, score, rows["pred"], base_tau, None)
        else:
            action = _mixed_action(base_score, score, rows["pred"], base_tau, chosen["threshold"])
    metrics = _metrics(action, rows["pred"], rows["truth"], rows["record_id"])
    return {
        "direction": context["direction"], "seed": context["seed"], "target": context["target"],
        "arm": arm, "base_threshold": base_tau, "selection": selection,
        "metrics": metrics,
        "same_coverage": _same_coverage_diagnostic(base_score, score, rows["pred"], rows["truth"], base_tau),
        "test_labels_read": False,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("cannot aggregate empty selector rows")
    # Reconstruct pooled arrays from the metric/action records stored below.
    action = np.concatenate([np.asarray(row["action"], dtype=bool) for row in rows])
    pred = np.concatenate([np.asarray(row["pred"], dtype=np.int8) for row in rows])
    truth = np.concatenate([np.asarray(row["truth"], dtype=np.int8) for row in rows])
    ids = np.concatenate([np.asarray(row["record_id"], dtype=str) for row in rows])
    out = _metrics(action, pred, truth, ids)
    out["seed_count"] = len(rows)
    out["released_seed_count"] = int(sum(bool(np.any(np.asarray(row["action"], dtype=bool))) for row in rows))
    out["selected_model_seed_count"] = int(sum(row["selection"].get("selected") is not None for row in rows))
    return out


def synthetic_selector_contract_tests() -> dict[str, str]:
    """Small no-fit guards for ties, empty candidates, and action retention."""
    score = np.asarray([0.2, 0.2, 0.3], dtype=float)
    pred = np.zeros(3, dtype=np.int8)
    truth = np.zeros(3, dtype=np.int8)
    ids = np.asarray(["a", "b", "c"], dtype=str)
    if _threshold_candidates(score) != [None, 0.2, 0.3]:
        raise RuntimeError("selector finite-unique threshold contract failed")
    tie_action = _mixed_action(score, score, pred, None, 0.2)
    if int(tie_action.sum()) != 2:
        raise RuntimeError("selector complete-tie contract failed")
    same_k = _same_coverage_diagnostic(score, score, pred, truth, 0.2)
    if same_k["status"] != "DESCRIPTIVE_ONLY" or same_k["base"]["release_expected"] != 2.0:
        raise RuntimeError("selector fractional same-k contract failed")
    # Low/high actions are retained even when the within threshold is None.
    mixed_pred = np.asarray([-1, 0, 1], dtype=np.int8)
    retained = _mixed_action(np.asarray([0.1, 0.3, 0.2]), np.asarray([0.9, 0.3, 0.9]),
                             mixed_pred, 0.5, None)
    if retained.tolist() != [True, False, True]:
        raise RuntimeError("selector non-within action retention failed")
    # The minimum finite score is a tail error; no threshold can satisfy the
    # strict zero-tail and >20% coverage rule in this toy pool.
    bad_truth = np.asarray([-1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
    bad_pred = np.zeros(10, dtype=np.int8)
    bad_score = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=float)
    bad_ids = np.asarray([f"r{i}" for i in range(10)], dtype=str)
    selected = _select_within_threshold(bad_score, bad_score, bad_pred, bad_truth, bad_ids, None)
    if selected["selected"] is not None:
        raise RuntimeError("selector no-feasible contract failed")
    return {"finite_unique_scores": "PASS", "complete_ties": "PASS",
            "fractional_same_k_diagnostic": "PASS", "nonwithin_action_retention": "PASS",
            "no_feasible": "PASS"}


def run_selector() -> dict[str, Any]:
    fit_report_path = r.ROOT / "FIT_RUN_REPORT.json"
    if not fit_report_path.exists():
        raise RuntimeError("FIT_RUN_REPORT.json is required before selector replay")
    fit_report = r.read_json(fit_report_path)
    if fit_report.get("failures"):
        raise RuntimeError("selector refuses incomplete fit report; failed units must be repaired first")
    if fit_report.get("test_labels_read") or fit_report.get("test_evaluation_started"):
        raise RuntimeError("fit report is not test blind")
    manifest = _load_base_manifest()
    arms = ("BASE_FROZEN", "BASE_JOINT", "WEIGHTED_JOINT")
    rows: list[dict[str, Any]] = []
    pooled: dict[str, dict[str, list[dict[str, Any]]]] = {direction: {arm: [] for arm in arms} for direction in r.DIRECTIONS}
    action_rows: dict[str, dict[str, list[np.ndarray]]] = {direction: {arm: [] for arm in arms} for direction in r.DIRECTIONS}
    for direction in r.DIRECTIONS:
        for seed in r.SEEDS:
            context = _load_context(direction, seed, manifest)
            for arm in arms:
                row = _arm_row(context, arm)
                row["action"] = _mixed_action(
                    context["rows"]["base_score"],
                    context["rows"]["base_score"] if arm != "WEIGHTED_JOINT" else context["rows"]["candidate_score"],
                    context["rows"]["pred"], context["base_tau"],
                    context["base_tau"] if arm == "BASE_FROZEN" else (
                        row["selection"]["selected"]["threshold"] if row["selection"].get("selected") is not None else None),
                )
                row["pred"] = context["rows"]["pred"].tolist()
                row["truth"] = context["rows"]["truth"].tolist()
                row["record_id"] = context["rows"]["record_id"].tolist()
                # Metrics are recomputed from exactly the stored action.
                row["metrics"] = _metrics(np.asarray(row["action"], dtype=bool), context["rows"]["pred"], context["rows"]["truth"], context["rows"]["record_id"])
                rows.append(row)
                pooled[direction][arm].append(row)
                action_rows[direction][arm].append(np.asarray(row["action"], dtype=bool))
    summaries: dict[str, Any] = {}
    for direction in r.DIRECTIONS:
        summaries[direction] = {}
        for arm in arms:
            summaries[direction][arm] = _aggregate(pooled[direction][arm])
    promotion: dict[str, Any] = {}
    single = [d for d in r.DIRECTIONS if "+" not in d]
    dual = [d for d in r.DIRECTIONS if "+" in d]
    for arm in arms:
        single_ok = [d for d in single if summaries[d][arm]["strict_eligible"]]
        dual_ok = [d for d in dual if summaries[d][arm]["strict_eligible"]]
        promotion[arm] = {
            "strict_agreement_operator": ">",
            "strict_coverage_operator": ">",
            "single_eligible_directions": single_ok,
            "dual_eligible_directions": dual_ok,
            "goal_evidence_ready": bool(single_ok and dual_ok),
            "test_evaluation_started": False,
        }
    action_npz = r.ROOT / "DEVELOPMENT_SELECTOR_ACTIONS.npz"
    np.savez_compressed(action_npz, **{
        f"{direction}__{arm}": np.concatenate(action_rows[direction][arm])
        for direction in r.DIRECTIONS for arm in arms
    })
    out = {
        "schema": "ma_tail_weight_gate_development_selector_v1",
        "candidate": "MA-TAIL-BALANCED-ERROR-LR",
        "arms": list(arms), "directions": list(r.DIRECTIONS), "seeds": list(r.SEEDS),
        "threshold_pool": "pooled outer-B positions0-2 only; finite unique scores plus None; complete ties",
        "threshold_selection": "eligible first, max pooled coverage, max pooled agreement, smallest numeric threshold",
        "strict_contract": {"agreement": ">0.75", "coverage": ">0.20", "tail_to_within": {"low": 0, "high": 0},
                            "correct_within_distinct_records": 5, "both_tail_support": True},
        "same_coverage": "descriptive only; no fractional tie enters actions or promotion",
        "test_labels_read": False, "test_evaluation_started": False,
        "fit_report_sha256": r.sha256(fit_report_path),
        "candidate_runner_sha256": fit_report.get("runner_sha256"),
        "selector_runner_sha256": r.sha256(Path(__file__)),
        "summaries": summaries, "promotion": promotion,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    r.write_json("DEVELOPMENT_SELECTOR.json", out)
    r.write_json("DEVELOPMENT_SELECTOR_ROWS.json", {
        "schema": "ma_tail_weight_gate_development_selector_rows_v1",
        "test_labels_read": False, "rows": rows,
    })
    r.write_json("CURRENT_STATUS.json", {
        "status": "DEVELOPMENT_SELECTOR_COMPLETE_NO_TEST", "test_labels_read": False,
        "test_evaluation_started": False, "science_units": r.ledger_totals()["science"],
        "repair_units": r.ledger_totals()["repair"], "total_units": r.ledger_totals()["total"],
        "selector": "DEVELOPMENT_SELECTOR.json", "selector_runner_sha256": out["selector_runner_sha256"],
    })
    r.update_ledger_summary(status="DEVELOPMENT_SELECTOR_COMPLETE_NO_TEST", current_phase="DEVELOPMENT_SELECTOR_COMPLETE",
                            selector_sha256=r.sha256(r.ROOT / "DEVELOPMENT_SELECTOR.json"))
    return out


if __name__ == "__main__":
    print(json.dumps(run_selector(), ensure_ascii=False))
