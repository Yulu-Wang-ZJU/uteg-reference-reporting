# Target-only sequential reporting

This directory provides the matched target-only comparator for sequential
reporting at 5, 10 and 15 minutes. It uses the same development/test design as
the focused source-transfer analysis and is intended for controlled re-runs
and post-run numerical checks.

## Contents

- `train_prefix.py`: frozen 300/600-second target-only development procedure.
- `evaluate.py`: matched schedule evaluation using frozen transfer actions and
  target-only objects.
- `verify.py`: read-only formula, first-report, aggregate and hash checks for
  an authorized frozen run.
- `report_results.py`: aggregate result-report generator.
- `PROTOCOL.md`: concise public protocol and restricted-input boundary.

## Required project layout

The scripts resolve their inputs from the archived project layout. After
extracting this archive, copy the `target_only_sequence` directory to

`outputs/uteg_nc_goal_20261030/target_only_sequence_20261031/`

so that it is a sibling of `focused_reporting_20261030` and `target_ridge`
under `outputs/uteg_nc_goal_20261030/`. Running it directly below the ZIP's
`Focused_reporting_code` directory does not provide the required `G`/`OUT`
layout.

From `outputs/uteg_nc_goal_20261030/`, with the restricted archived inputs in
that layout, the intended order is:

1. `python target_only_sequence_20261031/train_prefix.py check`
2. `python target_only_sequence_20261031/train_prefix.py train` in a fresh
   `target_only_sequence_20261031` directory
3. `python target_only_sequence_20261031/evaluate.py`
4. `python target_only_sequence_20261031/report_results.py`

Training and evaluation require the existing focused `target_ridge`, selector,
source-action and frozen-data paths. In addition, `evaluate.py` checks the
archive-integrity manifest `PROTECTED_BEFORE.json`, the existing analysis
script and focused result files, and the unchanged public-repository checkout.
The archive does not contain those restricted inputs and cannot independently
reconstruct the reported results. Keep restricted inputs and generated models,
freeze manifests, private arrays, individual predictions and clinical-data
workbooks in the controlled analysis workspace rather than in a code or data
release.

`verify.py` is an archived-environment audit, not a general fresh-run entry
point. It additionally consumes `FREEZE.json`, `EVALUATION_INPUT_HASHES.json`,
`RESULTS.json`, `private/evaluation_arrays.npz`, `PROTECTED_BEFORE.json`, the
protected source/manuscript files and the unchanged public-repository checkout.
Run it only after the authorized evaluation has produced those local artifacts.
Numerical results from that workspace are conditional descriptive analyses, not
external validation or a claim of clinical utility.

