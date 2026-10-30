#!/usr/bin/env python3
"""Validate the four-plus-five direction union without private fits.

The current main sequence runner intentionally contains the four directions
with an existing matched sequence result.  The supplement batch owns the five
single-source directions whose prefix branches were added later.  This check
verifies the source declarations and current aggregate direction coverage,
then exercises first-report retention on synthetic arrays for all nine labels.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path

CURRENT = (
    "HZ_to_SA",
    "HZ+HB_to_SA",
    "SA+HB_to_HZ",
    "SA+HZ_to_HB",
)
MISSING = (
    "HB_to_SA",
    "SA_to_HZ",
    "HB_to_HZ",
    "SA_to_HB",
    "HZ_to_HB",
)
NINE = tuple(
    sorted(
        {
            "HZ_to_SA",
            "HB_to_SA",
            "HZ+HB_to_SA",
            "SA_to_HZ",
            "HB_to_HZ",
            "SA_to_HB",
            "HZ_to_HB",
            "SA+HB_to_HZ",
            "SA+HZ_to_HB",
        }
    )
)


def literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    for node in tree.body:
        targets = []
        value = None
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return ast.literal_eval(value)
    raise ValueError(f"literal assignment {name!r} not found in {path}")


def csv_directions(path: Path) -> set[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        if not rows.fieldnames or "direction" not in rows.fieldnames:
            raise ValueError(f"direction column is missing from {path}")
        return {str(row["direction"]) for row in rows if row.get("direction")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="workspace root containing current sequence/supplement outputs",
    )
    args = parser.parse_args()
    root = args.source_root.expanduser().resolve()
    release = Path(__file__).resolve().parents[1]
    sequence_source = release / "snapshots/sequence/run.py"
    batch_source = release / "snapshots/supplement/batch.py"

    declared_current = tuple(literal_assignment(sequence_source, "DIRECTIONS"))
    declared_missing = tuple(literal_assignment(batch_source, "MISSING_PREFIX"))
    failures: list[str] = []
    if declared_current != CURRENT:
        failures.append(f"sequence DIRECTIONS changed: {declared_current!r}")
    if declared_missing != MISSING:
        failures.append(f"supplement MISSING_PREFIX changed: {declared_missing!r}")
    if set(declared_current) & set(declared_missing):
        failures.append("current and missing direction sets overlap")
    if set(declared_current) | set(declared_missing) != set(NINE):
        failures.append("current plus missing directions do not cover the nine-direction contract")

    results_path = root / "outputs/uteg_current_ma_sequence_20261026/RESULTS.json"
    summary_path = root / "outputs/uteg_supplement_completion_20261027_203140/tables/A2_nine_direction_summary.csv"
    cumulative_path = root / "outputs/uteg_supplement_completion_20261027_203140/tables/A7_pointwise_cumulative.csv"
    if results_path.is_file():
        result_directions = set(json.loads(results_path.read_text(encoding="utf-8-sig")))
        if result_directions != set(CURRENT):
            failures.append(f"current RESULTS.json directions changed: {sorted(result_directions)!r}")
    else:
        failures.append(f"missing current aggregate: {results_path}")
    for path in (summary_path, cumulative_path):
        if path.is_file():
            observed = csv_directions(path)
            if observed != set(NINE):
                failures.append(f"aggregate does not cover nine directions: {path}")
        else:
            failures.append(f"missing supplement aggregate: {path}")

    # The replay is deliberately synthetic and exercises the package contract,
    # not the private records or any fitting routine.
    sys.path.insert(0, str(release / "src"))
    from uteg_reference_reporting.core import cumulative_first_report

    for offset, direction in enumerate(NINE):
        shape = (2, 4)
        actions = []
        predictions = []
        for horizon in (300, 600, 900):
            action = __import__("numpy").zeros(shape, dtype=bool)
            action[:, : min(4, 1 + (horizon // 300))] = True
            predicted = __import__("numpy").full(shape, offset % 3 - 1, dtype=int)
            actions.append(action)
            predictions.append(predicted)
        replay = cumulative_first_report(
            [(horizon, action, predicted) for horizon, action, predicted in zip((300, 600, 900), actions, predictions)]
        )
        if replay["reported"].shape != shape or not replay["reported"].all():
            failures.append(f"synthetic first-report replay failed for {direction}")
        if (replay["first_time"] < 0).any():
            failures.append(f"synthetic replay left an unreported record for {direction}")

    payload = {
        "status": "PASS" if not failures else "FAIL",
        "current_main_runner_directions": list(CURRENT),
        "supplement_added_directions": list(MISSING),
        "nine_direction_union": list(NINE),
        "main_runner_direction_count": len(CURRENT),
        "supplement_direction_count": len(NINE),
        "private_fits_run": False,
        "synthetic_replays": len(NINE),
    }
    print(json.dumps(payload, indent=2))
    if failures:
        print("FAILURES:")
        print("\n".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
