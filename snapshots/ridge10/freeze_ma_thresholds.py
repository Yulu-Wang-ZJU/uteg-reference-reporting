"""Freeze the original MA outer-B score thresholds after target fitting.

This module deliberately does not import the historical runner.  It extracts
only the two pure threshold functions from the audited ``fit_ma_gates.py`` by
AST, records that source hash, and applies them to the outer-B rows saved by
``run_ridge10_fullchain.py``.  The target risk values (0.10/0.15) are empirical
release-error caps; they are never used as score cutoffs.

The script is post-fit and test-free.  It must be run only after all target
objects for the intended development arm are materialized and before any test
action is generated.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import runtime as r  # set all run-local caches before numerical imports
import joblib
import numpy as np


ROOT = r.ROOT
PROJECT_ROOT = ROOT.parents[2]
OLD_GATE_SOURCE = PROJECT_ROOT / "outputs" / "uteg_nine_direction_ma_20261025_195548" / "code" / "fit_ma_gates.py"
MA_CLASSES = (-1, 0, 1)
TARGET_MAX_RISK = {"Huzhou": 0.10, "SAHZU": 0.15, "Hubei": 0.15}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _extract_old_selectors() -> tuple[Any, Any, str]:
    """Extract old ``summary`` and ``select_threshold`` without module import."""
    source = OLD_GATE_SOURCE.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(OLD_GATE_SOURCE))
    wanted = {"summary", "select_threshold"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if {node.name for node in nodes} != wanted:
        raise RuntimeError("old fit_ma_gates pure selector functions are incomplete")
    ns: dict[str, Any] = {"np": np, "Any": Any, "Mapping": Mapping, "MA_CLASSES": MA_CLASSES}
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(OLD_GATE_SOURCE), "exec"), ns, ns)
    return ns["summary"], ns["select_threshold"], hashlib.sha256(source.encode("utf-8")).hexdigest()


def _merged_outer_b(gates: Mapping[int, Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """Merge positions 0/1/2 in frozen pool order, preserving complete ties."""
    rows: list[tuple[int, str, int, int, float | None, float | None]] = []
    for code in (0, 1, 2):
        data = gates[code].get("outer_B_data")
        if not isinstance(data, Mapping):
            raise RuntimeError(f"missing outer_B_data for position {code}")
        fields = ("pool_index", "pool_order", "record_id", "pred_class", "truth_class", "score")
        if any(field not in data for field in fields):
            raise RuntimeError(f"incomplete outer_B_data for position {code}")
        n = len(data["pool_index"])
        if any(len(data[field]) != n for field in fields):
            raise RuntimeError(f"outer_B_data length mismatch for position {code}")
        for i in range(n):
            rows.append((int(data["pool_order"][i]), str(data["record_id"][i]),
                          int(data["pred_class"][i]), int(data["truth_class"][i]),
                          data["score"][i], data["pool_index"][i]))
    rows.sort(key=lambda row: (row[0], row[1]))
    return {
        "pool_order": np.asarray([row[0] for row in rows], dtype=np.int64),
        "record_id": np.asarray([row[1] for row in rows], dtype=str),
        "pred_class": np.asarray([row[2] for row in rows], dtype=np.int8),
        "truth_class": np.asarray([row[3] for row in rows], dtype=np.int8),
        "score": np.asarray([np.nan if row[4] is None else float(row[4]) for row in rows], dtype=float),
        "pool_index": np.asarray([row[5] for row in rows], dtype=np.int64),
    }


def _position_outer_b(gate: Mapping[str, Any]) -> dict[str, np.ndarray]:
    data = gate.get("outer_B_data")
    if not isinstance(data, Mapping):
        raise RuntimeError("missing outer_B_data")
    return {
        "pool_order": np.asarray(data.get("pool_order", []), dtype=np.int64),
        "record_id": np.asarray(data.get("record_id", []), dtype=str),
        "pred_class": np.asarray(data.get("pred_class", []), dtype=np.int8),
        "truth_class": np.asarray(data.get("truth_class", []), dtype=np.int8),
        "score": np.asarray([np.nan if v is None else float(v) for v in data.get("score", [])], dtype=float),
        "pool_index": np.asarray(data.get("pool_index", []), dtype=np.int64),
    }


def apply_thresholds(obj: dict[str, Any], *, old_source_sha256: str | None = None) -> dict[str, Any]:
    """Add old-contract threshold mappings to one fitted target object."""
    target = str(obj["target"])
    if target not in TARGET_MAX_RISK:
        raise RuntimeError(f"unknown target {target}")
    summary, select_threshold, source_sha = _extract_old_selectors()
    if old_source_sha256 is not None and source_sha != old_source_sha256:
        raise RuntimeError("old threshold source hash mismatch")
    gates = {int(code): obj["gate_models"][code] for code in range(4)}
    merged = _merged_outer_b(gates)
    cap = TARGET_MAX_RISK[target]
    for code in range(4):
        gate = gates[code]
        rows = merged if code == 3 else _position_outer_b(gate)
        selected = select_threshold(rows["score"], rows["pred_class"], rows["truth_class"], cap)
        selected["threshold_source"] = "outer_B_merged_positions_0_1_2" if code == 3 else "outer_B"
        selected["threshold_selector_source_sha256"] = source_sha
        selected["max_risk_is_empirical_error_cap"] = True
        selected["threshold_is_not_risk_cap"] = True
        gate["threshold"] = selected
        gate["outerB_threshold"] = selected.get("threshold")
        gate["outerB_risk_cap"] = float(cap)
        gate["outerB_threshold_source"] = selected["threshold_source"]
        gate["outer_B_threshold_rows"] = {
            "n": int(len(rows["score"])),
            "pool_order_sha256": hashlib.sha256(np.asarray(rows["pool_order"], dtype=np.int64).tobytes()).hexdigest(),
            "record_id_sha256": hashlib.sha256("\n".join(rows["record_id"].tolist()).encode()).hexdigest(),
        }
    obj["outerB_threshold_contract"] = {
        "target": target,
        "max_risk": float(cap),
        "threshold_selector": "old_fit_ma_gates.summary_select_threshold_ast_extract",
        "threshold_selector_source_sha256": source_sha,
        "threshold_is_not_risk_cap": True,
        "position3": {"merged_positions": [0, 1, 2], "threshold": gates[3]["threshold"].get("threshold"),
                       "max_risk": float(cap), "threshold_source": "outer_B_merged_positions_0_1_2"},
        "selection_from_test": False,
        "threshold_refit": False,
    }
    return obj


def freeze_object(path: Path) -> Path:
    obj = joblib.load(path)
    if obj.get("external_test_evaluated") is not False or obj.get("test_labels_hidden") is not True:
        raise RuntimeError(f"test contract violation: {path}")
    out = apply_thresholds(obj)
    joblib.dump(out, path, compress=3)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path, help="fitted target joblib objects")
    ap.add_argument("--manifest", type=Path, help="write selector provenance JSON")
    args = ap.parse_args()
    summary, select_threshold, source_sha = _extract_old_selectors()
    del summary, select_threshold
    paths = [p if p.is_absolute() else ROOT / p for p in args.paths]
    for path in paths:
        freeze_object(path)
    payload = {"version": "outer_b_ma_threshold_freeze_v1", "paths": [str(p) for p in paths],
               "old_fit_ma_gates_sha256": source_sha, "test_labels_read": False,
               "test_evaluated": False, "selector": "summary/select_threshold via AST pure extraction",
               "max_risk_caps": TARGET_MAX_RISK}
    if args.manifest:
        args.manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
