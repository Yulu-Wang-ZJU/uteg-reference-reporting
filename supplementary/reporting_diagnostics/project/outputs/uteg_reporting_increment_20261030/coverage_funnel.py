"""Summarise fixed 15-min dual-head coverage by target centre and true MA class.

This is a read-only analysis of already frozen models, development thresholds,
and external action arrays.  It deliberately does not fit a model, search a
threshold, or alter any upstream artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


CLASS_CODES = (-1, 0, 1)
CLASS_NAMES = {-1: "low", 0: "within", 1: "high"}
REPORT_CODES = (-1, 0, 1, 9)
REPORT_NAMES = {-1: "low", 0: "within", 1: "high", 9: "unreported"}
MODEL_NAME_RE = re.compile(r"^(d\d{2})s(\d+)p3(lo|hi)\.json$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def class_name(code: int) -> str:
    return CLASS_NAMES[int(code)]


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def safe_rate(num: int, den: int) -> float | None:
    return None if den == 0 else float(num) / float(den)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def parse_slots_and_models(runner_dir: Path) -> tuple[
    dict[str, str],
    dict[tuple[str, int, str], dict[str, Any]],
    dict[tuple[str, int, str], dict[str, Any]],
    dict[str, str],
]:
    """Read p3 support from slot metadata and p3 model payloads.

    The direction code is not guessed from file ordering: it is resolved from
    the model payload attempt_key, while the support status is cross-checked
    against the current FIT_LEDGER p3 slot metadata.
    """

    ledger_path = runner_dir / "FIT_LEDGER.jsonl"
    slots: dict[tuple[str, int, str], dict[str, Any]] = {}
    with ledger_path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item.get("event") != "slot":
                continue
            metadata = item.get("metadata", {})
            if metadata.get("position") != 3:
                continue
            direction = str(metadata["direction"])
            seed = int(metadata["seed"])
            head = str(metadata["head"])
            support = dict(metadata.get("support") or {})
            slots[(direction, seed, head)] = {
                "status": support.get("status", item.get("support_status")),
                "support": support,
                "slot_key": item.get("slot_key"),
            }

    direction_by_code: dict[str, str] = {}
    models: dict[tuple[str, int, str], dict[str, Any]] = {}
    model_manifest: list[str] = []
    model_dir = runner_dir / "models"
    for path in sorted(model_dir.glob("d??s????p3*.json")):
        match = MODEL_NAME_RE.match(path.name)
        if match is None:
            continue
        dcode, seed_text, head_short = match.groups()
        seed = int(seed_text)
        head = "low" if head_short == "lo" else "high"
        payload_wrapper = json_load(path)
        payload = dict(payload_wrapper.get("payload") or {})
        attempt_key = str(payload.get("attempt_key", ""))
        direction = attempt_key.split("|", 1)[0]
        if not direction or "|" not in attempt_key:
            raise ValueError(f"Cannot resolve direction from {path}")
        if payload.get("head") != head:
            raise ValueError(f"Head mismatch in {path}")
        old = direction_by_code.setdefault(dcode, direction)
        if old != direction:
            raise ValueError(f"Direction code {dcode} maps to both {old} and {direction}")
        support = dict(payload.get("support") or {})
        models[(direction, seed, head)] = {
            "status": support.get("status"),
            "support": support,
            "path": str(path),
            "payload_sha256": payload_wrapper.get("payload_sha256"),
        }
        model_manifest.append(f"{path.name}\t{sha256_file(path)}")

    if not direction_by_code:
        raise ValueError("No p3 model payloads were found")
    slot_directions = {key[0] for key in slots}
    model_directions = set(direction_by_code.values())
    if slot_directions != model_directions:
        raise ValueError(
            "Slot/model direction sets differ: "
            f"slots={sorted(slot_directions)}, models={sorted(model_directions)}"
        )
    return (
        direction_by_code,
        slots,
        models,
        {"count": str(len(model_manifest)), "sha256": sha256_text("\n".join(model_manifest))},
    )


def load_thresholds(path: Path) -> tuple[list[int], dict[tuple[str, int], dict[str, Any]]]:
    frozen = json_load(path)
    entries = list(frozen.get("entries") or [])
    if len(entries) != 180:
        raise ValueError(f"Expected 180 frozen threshold entries, got {len(entries)}")
    table: dict[tuple[str, int], dict[str, Any]] = {}
    for entry in entries:
        key = (str(entry["direction"]), int(entry["seed"]))
        if key in table:
            raise ValueError(f"Duplicate threshold entry: {key}")
        table[key] = entry
    seeds = sorted({seed for _, seed in table})
    if len(seeds) != 20:
        raise ValueError(f"Expected 20 seeds, got {seeds}")
    return seeds, table


def load_data(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        required = {"medical", "ids", "centers", "input_qc"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"DATA15 missing keys: {sorted(missing)}")
        ids = np.asarray(data["ids"]).astype(str)
        centers = np.asarray(data["centers"]).astype(str)
        medical = np.asarray(data["medical"], dtype=float)
        input_qc = np.asarray(data["input_qc"], dtype=bool)
    if medical.ndim != 2 or medical.shape[1] <= 2:
        raise ValueError(f"medical has unexpected shape {medical.shape}")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("DATA15 ids are not unique")
    true_ma = medical[:, 2]
    if not np.isfinite(true_ma).all():
        raise ValueError("DATA15 medical[:,2] contains non-finite MA")
    true_class = np.where(true_ma < 50.0, -1, np.where(true_ma <= 70.0, 0, 1)).astype(np.int8)
    return {
        "ids": ids,
        "centers": centers,
        "true_ma": true_ma,
        "true_class": true_class,
        "input_qc": input_qc,
        "index": {record_id: i for i, record_id in enumerate(ids.tolist())},
    }


def load_actions(actions_dir: Path, data: dict[str, Any], seeds: list[int]) -> dict[str, dict[str, Any]]:
    actions: dict[str, dict[str, Any]] = {}
    for path in sorted(actions_dir.glob("*.npz")):
        direction = path.stem
        with np.load(path, allow_pickle=False) as z:
            required = {"test_ids", "input_qc", "final_ma_class", "dual_valid", "unified_action"}
            missing = required.difference(z.files)
            if missing:
                raise ValueError(f"{path.name} missing keys: {sorted(missing)}")
            test_ids = np.asarray(z["test_ids"]).astype(str)
            input_qc = np.asarray(z["input_qc"], dtype=bool)
            final_class = np.asarray(z["final_ma_class"], dtype=np.int8)
            dual_valid = np.asarray(z["dual_valid"], dtype=bool)
            action = np.asarray(z["unified_action"], dtype=bool)
        if final_class.ndim != 2 or final_class.shape[0] != len(seeds):
            raise ValueError(f"{path.name} final_ma_class shape {final_class.shape} is not (20,n)")
        if final_class.shape[1] != len(test_ids):
            raise ValueError(f"{path.name} array/ID length mismatch")
        if dual_valid.shape != final_class.shape or action.shape != final_class.shape:
            raise ValueError(f"{path.name} action array shapes do not agree")
        if input_qc.shape != (len(test_ids),):
            raise ValueError(f"{path.name} input_qc shape mismatch")
        bad_codes = set(np.unique(final_class).tolist()).difference({-1, 0, 1})
        if bad_codes:
            raise ValueError(f"{path.name} has unexpected final_ma_class values {sorted(bad_codes)}")
        try:
            indices = np.asarray([data["index"][record_id] for record_id in test_ids], dtype=int)
        except KeyError as exc:
            raise ValueError(f"{path.name} contains an ID absent from DATA15: {exc}") from exc
        center_values = sorted(set(data["centers"][indices].tolist()))
        if len(center_values) != 1:
            raise ValueError(f"{path.name} spans multiple target centres: {center_values}")
        target_center = center_values[0]
        actions[direction] = {
            "path": str(path),
            "test_ids": test_ids,
            "indices": indices,
            "input_qc": input_qc,
            "final_class": final_class,
            "dual_valid": dual_valid,
            "action": action,
            "target_center": target_center,
            "sha256": sha256_file(path),
        }
    if len(actions) != 9:
        raise ValueError(f"Expected nine action files, got {len(actions)}")
    return actions


def expected_center_from_direction(direction: str) -> str | None:
    if "_to_" not in direction:
        return None
    suffix = direction.rsplit("_to_", 1)[1]
    return {"SA": "SAHZU", "HZ": "Huzhou", "HB": "Hubei"}.get(suffix)


def support_rows(
    direction: str,
    seeds: list[int],
    slots: dict[tuple[str, int, str], dict[str, Any]],
    models: dict[tuple[str, int, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        row: dict[str, Any] = {"direction": direction, "seed": seed}
        for head in ("low", "high"):
            slot = slots.get((direction, seed, head))
            model = models.get((direction, seed, head))
            slot_status = None if slot is None else slot.get("status")
            model_status = None if model is None else model.get("status")
            row[f"{head}_slot_status"] = slot_status
            row[f"{head}_model_status"] = model_status
            row[f"{head}_supported"] = bool(
                slot_status == "SUPPORTED" and model_status == "SUPPORTED" and model is not None
            )
        row["both_supported"] = bool(row["low_supported"] and row["high_supported"])
        rows.append(row)
    return rows


def build_outputs(
    output_dir: Path,
    data_path: Path,
    threshold_path: Path,
    runner_dir: Path,
    actions_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    data = load_data(data_path)
    seeds, thresholds = load_thresholds(threshold_path)
    direction_by_code, slots, models, model_manifest = parse_slots_and_models(runner_dir)
    actions = load_actions(actions_dir, data, seeds)

    directions = sorted(actions)
    model_directions = set(direction_by_code.values())
    if set(directions) != model_directions:
        raise ValueError(
            f"Action/model direction sets differ: actions={directions}, models={sorted(model_directions)}"
        )
    threshold_directions = {key[0] for key in thresholds}
    if threshold_directions != set(directions):
        raise ValueError("Threshold/action direction sets differ")

    funnel_rows: list[dict[str, Any]] = []
    destination_rows: list[dict[str, Any]] = []
    direction_summaries: dict[str, Any] = {}
    target_center_map: dict[str, str] = {}
    support_status_mismatches: list[dict[str, Any]] = []
    class_flow_checks: list[bool] = []

    for direction in directions:
        item = actions[direction]
        target_center = item["target_center"]
        expected_center = expected_center_from_direction(direction)
        if expected_center != target_center:
            raise ValueError(
                f"Target centre mismatch for {direction}: IDs={target_center}, suffix={expected_center}"
            )
        target_center_map[direction] = target_center
        support = support_rows(direction, seeds, slots, models)
        for sr in support:
            for head in ("low", "high"):
                slot_status = sr[f"{head}_slot_status"]
                model_status = sr[f"{head}_model_status"]
                if slot_status == "SUPPORTED" and model_status != "SUPPORTED":
                    support_status_mismatches.append(
                        {"direction": direction, "seed": sr["seed"], "head": head, "slot": slot_status, "model": model_status}
                    )
                if slot_status != "SUPPORTED" and model_status == "SUPPORTED":
                    support_status_mismatches.append(
                        {"direction": direction, "seed": sr["seed"], "head": head, "slot": slot_status, "model": model_status}
                    )

        normal_qualified = {
            seed: bool(
                thresholds[(direction, seed)].get("normal_status") == "NORMAL_SELECTED"
                and thresholds[(direction, seed)].get("normal_threshold") is not None
            )
            for seed in seeds
        }
        tail_selected = {
            seed: bool(thresholds[(direction, seed)].get("common_tail_status") == "COMMON_TAIL_SELECTED")
            for seed in seeds
        }
        tail_value_available = {
            seed: bool(
                thresholds[(direction, seed)].get("common_tail_status") == "COMMON_TAIL_SELECTED"
                and thresholds[(direction, seed)].get("common_tail_threshold") is not None
            )
            for seed in seeds
        }
        for seed in seeds:
            entry = thresholds[(direction, seed)]
            if (entry.get("normal_status") == "NORMAL_SELECTED") != (
                entry.get("normal_threshold") is not None
            ):
                raise ValueError(f"Normal threshold status/value inconsistency: {direction}, {seed}")
            if entry.get("common_tail_status") != "COMMON_TAIL_SELECTED" and entry.get("common_tail_threshold") is not None:
                raise ValueError(f"Tail threshold status/value inconsistency: {direction}, {seed}")

        final_class = item["final_class"]
        action = item["action"]
        record_indices = item["indices"]
        true_class = data["true_class"][record_indices]
        qc = item["input_qc"]
        n_records = len(record_indices)
        n_seeds = len(seeds)
        direction_report_seed_count = int(np.any(action, axis=1).sum())
        direction_report_events = int(action.sum())
        direction_dual_valid_events = int(item["dual_valid"].sum())
        direction_summaries[direction] = {
            "target_center": target_center,
            "test_record_count": n_records,
            "record_seed_denominator": n_records * n_seeds,
            "low_head_supported_seed_count": int(sum(row["low_supported"] for row in support)),
            "high_head_supported_seed_count": int(sum(row["high_supported"] for row in support)),
            "both_head_supported_seed_count": int(sum(row["both_supported"] for row in support)),
            "normal_threshold_qualified_seed_count": int(sum(normal_qualified.values())),
            "tail_threshold_selected_seed_count": int(sum(tail_selected.values())),
            "tail_threshold_value_seed_count": int(sum(tail_value_available.values())),
            "tail_explicit_withhold_seed_count": int(
                sum(tail_selected[seed] and not tail_value_available[seed] for seed in seeds)
            ),
            "both_threshold_value_seed_count": int(
                sum(normal_qualified[seed] and tail_value_available[seed] for seed in seeds)
            ),
            "actual_report_seed_count": direction_report_seed_count,
            "actual_report_seed_denominator": n_seeds,
            "actual_report_record_seed_events": direction_report_events,
            "test_dual_valid_record_seed_events": direction_dual_valid_events,
        }

        for true_code in CLASS_CODES:
            true_mask = true_class == true_code
            n_true_records = int(true_mask.sum())
            denominator = n_true_records * n_seeds
            class_action = action[:, true_mask]
            class_prediction = final_class[:, true_mask]
            correct_events = int((class_action & (class_prediction == true_code)).sum())
            wrong_events = int((class_action & (class_prediction != true_code)).sum())
            unreported_events = int((~class_action).sum())
            released_events = correct_events + wrong_events
            if correct_events + wrong_events + unreported_events != denominator:
                raise ValueError(f"Class flow does not close for {direction}/{true_code}")
            class_flow_checks.append(correct_events + wrong_events + unreported_events == denominator)
            qc_true = qc[true_mask]
            report_seed_for_class = int(np.any(class_action, axis=1).sum())
            class_row = {
                "direction": direction,
                "target_center": target_center,
                "true_class": class_name(true_code),
                "true_class_code": true_code,
                "class_support_status": "SUPPORTED" if n_true_records else "NO_TRUE_CLASS_SUPPORT",
                "test_record_count": n_records,
                "true_class_record_count": n_true_records,
                "true_class_record_seed_denominator": denominator,
                "qc_eligible_record_count": int(qc_true.sum()),
                "qc_ineligible_record_count": int((~qc_true).sum()),
                "qc_ineligible_record_seed_count": int((~qc_true).sum()) * n_seeds,
                "seed_denominator": n_seeds,
                "low_head_supported_seed_count": direction_summaries[direction]["low_head_supported_seed_count"],
                "high_head_supported_seed_count": direction_summaries[direction]["high_head_supported_seed_count"],
                "both_head_supported_seed_count": direction_summaries[direction]["both_head_supported_seed_count"],
                "normal_threshold_qualified_seed_count": direction_summaries[direction]["normal_threshold_qualified_seed_count"],
                "tail_threshold_selected_seed_count": direction_summaries[direction]["tail_threshold_selected_seed_count"],
                "tail_threshold_value_seed_count": direction_summaries[direction]["tail_threshold_value_seed_count"],
                "tail_explicit_withhold_seed_count": direction_summaries[direction]["tail_explicit_withhold_seed_count"],
                "both_threshold_value_seed_count": direction_summaries[direction]["both_threshold_value_seed_count"],
                "actual_report_seed_count_direction": direction_report_seed_count,
                "actual_report_seed_count_for_class": report_seed_for_class,
                "correct_record_seed_events": correct_events,
                "wrong_record_seed_events": wrong_events,
                "unreported_record_seed_events": unreported_events,
                "released_record_seed_events": released_events,
                "correct_yield": safe_rate(correct_events, denominator),
                "wrong_yield": safe_rate(wrong_events, denominator),
                "coverage": safe_rate(released_events, denominator),
                "agreement": safe_rate(correct_events, released_events),
            }
            funnel_rows.append(class_row)

            for report_code in REPORT_CODES:
                if report_code == 9:
                    destination_mask = ~class_action
                    destination_category = "unreported"
                else:
                    destination_mask = class_action & (class_prediction == report_code)
                    destination_category = "correct" if report_code == true_code else "wrong"
                event_count = int(destination_mask.sum())
                unique_records = int(np.any(destination_mask, axis=0).sum()) if n_true_records else 0
                seed_count = int(np.any(destination_mask, axis=1).sum()) if n_true_records else 0
                destination_rows.append(
                    {
                        "direction": direction,
                        "target_center": target_center,
                        "true_class": class_name(true_code),
                        "true_class_code": true_code,
                        "reported_class": REPORT_NAMES[report_code],
                        "reported_class_code": report_code,
                        "destination_category": destination_category,
                        "class_support_status": "SUPPORTED" if n_true_records else "NO_TRUE_CLASS_SUPPORT",
                        "record_seed_events": event_count,
                        "unique_records": unique_records,
                        "reported_seed_count": seed_count,
                        "record_seed_denominator": denominator,
                        "proportion": safe_rate(event_count, denominator),
                    }
                )

    class_flow_closed = all(class_flow_checks)
    direction_code_mapping = {code: direction_by_code[code] for code in sorted(direction_by_code)}
    # The funnel is a test-cohort summary.  Do not substitute the full 480-row
    # DATA15 centre totals for the 201/55/41 target test pools.
    test_center_ids: dict[str, set[str]] = {}
    test_center_class_records: dict[str, dict[str, int]] = {}
    for item in actions.values():
        center = item["target_center"]
        ids = set(item["test_ids"].tolist())
        if center in test_center_ids and test_center_ids[center] != ids:
            raise ValueError(f"Directions targeting {center} do not share the same test IDs")
        test_center_ids[center] = ids
        indices = item["indices"]
        true_classes = data["true_class"][indices]
        test_center_class_records[center] = {
            class_name(code): int(np.sum(true_classes == code)) for code in CLASS_CODES
        }
    test_center_record_counts = {
        center: int(sum(class_counts.values()))
        for center, class_counts in test_center_class_records.items()
    }

    action_ids_match_data = all(
        all(record_id in data["index"] for record_id in item["test_ids"].tolist()) for item in actions.values()
    )
    target_center_from_ids_matches_suffix = all(
        expected_center_from_direction(direction) == item["target_center"] for direction, item in actions.items()
    )
    data_center_counts = Counter(data["centers"].tolist())
    output_files = [
        "coverage_funnel.py",
        "center_class_funnel.csv",
        "center_class_destinations.csv",
        "COVERAGE_FUNNEL_ZH.md",
        "funnel_validation.json",
    ]
    validation = {
        "schema": "coverage_funnel_v1",
        "status": "VALIDATED_FIXED_15MIN_ACTIONS_NO_NEW_SEARCH",
        "analysis_scope": "fixed 15-min external actions; no model fit and no threshold search",
        "threshold_search_performed": False,
        "new_model_fit": False,
        "test_labels_used_for_threshold_selection": False,
        "true_ma_source": "DATA15.npz: medical[:,2]",
        "true_class_rule": {"low": "MA < 50", "within": "50 <= MA <= 70", "high": "MA > 70"},
        "seed_order": seeds,
        "direction_code_mapping_from_model_attempt_key": direction_code_mapping,
        "target_center_by_direction_from_test_ids": target_center_map,
        "input_hashes": {
            "data15": sha256_file(data_path),
            "thresholds_freeze": sha256_file(threshold_path),
            "fit_ledger": sha256_file(runner_dir / "FIT_LEDGER.jsonl"),
            "actions": {direction: actions[direction]["sha256"] for direction in directions},
            "p3_model_manifest": model_manifest,
        },
        "data15_center_record_counts": dict(sorted(data_center_counts.items())),
        "test_center_record_counts": dict(sorted(test_center_record_counts.items())),
        "test_center_true_class_record_counts": test_center_class_records,
        "direction_summary": direction_summaries,
        "checks": {
            "threshold_entry_count_180": len(thresholds) == 180,
            "threshold_seed_count_20": len(seeds) == 20,
            "action_file_count_9": len(actions) == 9,
            "action_ids_match_data15": action_ids_match_data,
            "direction_mapping_verified_by_action_centres": target_center_from_ids_matches_suffix,
            "slot_model_support_status_mismatches": support_status_mismatches,
            "all_class_flows_close": class_flow_closed,
            "all_destinations_sum_to_class_denominator": class_flow_closed,
            "qc_ineligible_records_remain_in_denominator": True,
            "threshold_qualification_not_called_prediction_failure": True,
            "repeated_seeds_not_patient_independence": True,
        },
        "temporal_evidence_not_recomputed": [
            "outputs/uteg_nc_goal_20261030/analysis/RESULTS.json",
            "outputs/uteg_nc_goal_20261030/analysis/first_report_times.csv",
        ],
        "output_files": output_files,
        "notes": [
            "A missing or unqualified threshold is reported as a gate/eligibility status, not as a prediction failure.",
            "COMMON_TAIL_SELECTED with a null threshold is retained as an explicit tail-withhold decision; it is not counted as a usable numeric tail threshold.",
            "A true class with zero records is marked NO_TRUE_CLASS_SUPPORT; it is not treated as zero error support.",
            "The 15-min funnel does not establish earlier reporting at 5 or 10 min.",
        ],
    }
    return funnel_rows, destination_rows, validation


def make_markdown(
    funnel_rows: list[dict[str, Any]],
    destination_rows: list[dict[str, Any]],
    validation: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# 固定 15 min 覆盖漏斗补充")
    lines.append("")
    lines.append(
        "本附件只汇总当前已冻结的 15 min dual-head actions，按“最终风险头支持 → 冻结阈值资格 → 测试释放”"
        "串联到目标中心和真实 MA 类别。没有训练模型、没有重新搜索阈值，也没有改变任何 action。"
    )
    lines.append("")
    lines.append("## 读取范围与判定")
    lines.append("")
    lines.append(
        "方向映射来自当前 p3 slot/模型 payload 的 `attempt_key`；最终头支持来自 `p3lo/p3hi` 的 "
        "`payload.support`，并与 `FIT_LEDGER.jsonl` 的 p3 slot 状态交叉核对。阈值状态直接读取 "
        "`THRESHOLDS_FREEZE.json`：`NORMAL_SELECTED` 表示有数值阈值；`COMMON_TAIL_SELECTED` 可能是数值阈值，"
        "也可能是明确选择 tail 全部 withholding 的 `None`，因此表中将 tail 的‘状态选择’与‘数值可用’分开。"
    )
    lines.append("")
    lines.append(
        "真实类别使用 `DATA15.npz` 的 `medical[:,2]`：low = MA < 50，within = 50 ≤ MA ≤ 70，"
        "high = MA > 70。计数单位为 record–seed，20 个 seed 不增加患者独立性；QC 不合格记录仍保留在分母。"
    )
    lines.append("")
    lines.append("## 目标中心和真实类别支持（观察事实）")
    lines.append("")
    lines.append("| 中心 | 测试记录 | low | within | high | record–seed 总量 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    center_counts = validation["test_center_record_counts"]
    center_classes = validation["test_center_true_class_record_counts"]
    for center in sorted(center_counts):
        cls = center_classes[center]
        lines.append(
            f"| {center} | {center_counts[center]} | {cls['low']} | {cls['within']} | {cls['high']} | {center_counts[center] * 20} |"
        )
    lines.append("")
    lines.append(
        "Huzhou 测试记录没有真实 high 类，因此相关 high 行标记为 `NO_TRUE_CLASS_SUPPORT`；这不是“零错误”"
        "或模型已验证覆盖 high 类的证据。"
    )
    lines.append("")
    lines.append("## 九个方向的漏斗（观察事实）")
    lines.append("")
    lines.append(
        "下表的支持和阈值列是 20 个 seed 中的 seed 数；`actual_report_seed_count` 是测试中至少释放一条记录的 seed 数。"
    )
    lines.append("")
    lines.append(
        "| direction | target | low 头支持 | high 头支持 | 双头支持 | normal 阈值资格 | tail 阈值选择（可用值） | 有报告 seed | 释放事件 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for direction in sorted(validation["direction_summary"]):
        s = validation["direction_summary"][direction]
        lines.append(
            f"| {direction} | {s['target_center']} | {s['low_head_supported_seed_count']}/20 | "
            f"{s['high_head_supported_seed_count']}/20 | {s['both_head_supported_seed_count']}/20 | "
            f"{s['normal_threshold_qualified_seed_count']}/20 | "
            f"{s['tail_threshold_selected_seed_count']}/20 (value {s['tail_threshold_value_seed_count']}/20) | "
            f"{s['actual_report_seed_count']}/20 | {s['actual_report_record_seed_events']} |"
        )
    lines.append("")
    lines.append("## 真实类别释放结果")
    lines.append("")
    lines.append(
        "`center_class_funnel.csv` 给出每个 direction×center×true-class 的 correct/wrong/unreported，"
        "并同时保留头支持、阈值资格、实际有报告 seed 数和 record–seed 分母。"
    )
    lines.append("")
    lines.append(
        "`center_class_destinations.csv` 进一步把报告目的地拆成 low/within/high/unreported；每个真实类别的四类"
        "目的地之和应等于该类 record–seed 分母。"
    )
    lines.append("")
    lines.append("## 解释边界")
    lines.append("")
    lines.append(
        "这里的漏斗是可用性和释放协议的描述性分解。阈值无资格或风险头不支持会减少释放，但不能直接解释为"
        "预测错误；反过来，释放增加也不自动意味着临床效用增加。最终 release 的 correct/wrong 只说明在固定"
        "50/70 类别下的经验类别一致性。"
    )
    lines.append("")
    lines.append(
        "本附件只覆盖 15 min；没有重做 5/10 min 训练，也不把 15 min 结果称为更早报告证据。已有时序材料仅定位"
        "于当前 `outputs/uteg_nc_goal_20261030/analysis/RESULTS.json` 和 `first_report_times.csv`。"
    )
    lines.append("")
    lines.append(
        "该补充仍使用既有测试队列，因此不是独立外部验证。任何 target-only focused reporting 文件若被引用，"
        "应与本九方向 funnel 分开，不把它们混成新的独立中心/病例。"
    )
    lines.append("")
    lines.append("## 机器核验")
    lines.append("")
    checks = validation["checks"]
    lines.append(f"- 阈值条目：{len(validation['direction_summary']) * 20}；方向：{len(validation['direction_summary'])}；seed：20。")
    lines.append(f"- 类别流闭合：`{checks['all_class_flows_close']}`；目标中心映射核对：`{checks['direction_mapping_verified_by_action_centres']}`。")
    lines.append(f"- p3 slot/model 支持状态不一致数：{len(checks['slot_model_support_status_mismatches'])}。")
    lines.append("- 详细哈希、输入和逐方向计数见 `funnel_validation.json`。")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    repo_root = output_dir.parents[1]
    outputs_root = repo_root / "outputs"
    data_path = outputs_root / "uteg_legacy_aligned_threecenter_20261022_155614" / "DATA15.npz"
    threshold_path = outputs_root / "uteg_ridge_dual_head_unified_selector_20261026_190142" / "THRESHOLDS_FREEZE.json"
    runner_dir = outputs_root / "uteg_ridge_dual_head_runner_20261026_173951"
    actions_dir = outputs_root / "uteg_ridge_dual_head_external_eval_20261026_191117" / "actions"
    for path in (data_path, threshold_path, runner_dir / "FIT_LEDGER.jsonl", actions_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    funnel_rows, destination_rows, validation = build_outputs(
        output_dir, data_path, threshold_path, runner_dir, actions_dir
    )
    funnel_fields = [
        "direction", "target_center", "true_class", "true_class_code", "class_support_status",
        "test_record_count", "true_class_record_count", "true_class_record_seed_denominator",
        "qc_eligible_record_count", "qc_ineligible_record_count", "qc_ineligible_record_seed_count",
        "seed_denominator", "low_head_supported_seed_count", "high_head_supported_seed_count",
        "both_head_supported_seed_count", "normal_threshold_qualified_seed_count",
        "tail_threshold_selected_seed_count", "tail_threshold_value_seed_count",
        "tail_explicit_withhold_seed_count", "both_threshold_value_seed_count",
        "actual_report_seed_count_direction", "actual_report_seed_count_for_class",
        "correct_record_seed_events", "wrong_record_seed_events", "unreported_record_seed_events",
        "released_record_seed_events", "correct_yield", "wrong_yield", "coverage", "agreement",
    ]
    destination_fields = [
        "direction", "target_center", "true_class", "true_class_code", "reported_class",
        "reported_class_code", "destination_category", "class_support_status", "record_seed_events",
        "unique_records", "reported_seed_count", "record_seed_denominator", "proportion",
    ]
    write_csv(output_dir / "center_class_funnel.csv", funnel_rows, funnel_fields)
    write_csv(output_dir / "center_class_destinations.csv", destination_rows, destination_fields)
    (output_dir / "funnel_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "COVERAGE_FUNNEL_ZH.md").write_text(
        make_markdown(funnel_rows, destination_rows, validation) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": validation["status"], "funnel_rows": len(funnel_rows), "destination_rows": len(destination_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
