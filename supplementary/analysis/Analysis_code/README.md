# Additional analyses accompanying Supplementary Data 6

These scripts implement the target-pool median, the target-only Ridge comparator and the three reporting-schedule replays. They supplement the core public workflow at commit `5f32af59907470ced142595625c4b80555fd1d81`; they are not included in that public commit.

## Inputs and execution

Python, NumPy and joblib are required. No patient data, frozen models, split manifests or per-record predictions are included. The authorized archive custodian must supply the restricted evaluation artifacts identified by the relative paths in the scripts. Set `UTEG_ARTIFACT_ROOT` to the parent containing those archived output directories. The two PROTOCOL.md files describe the design and information restrictions.

Use a writable copy of this code directory. Preserve its `analysis` and `target_ridge` subdirectories. Run in this order:

1. `python analysis/round1.py` for the constant reference and frozen-action schedule comparisons.
2. `python target_ridge/run.py train` for target-pool-only alpha selection and model freezing.
3. `python target_ridge/run.py evaluate` for evaluation of those frozen objects.

The scripts refuse to overwrite existing result/freeze files. Outputs include aggregate summaries and per-seed tables. Training also writes fitted objects containing restricted record assignments; these objects must remain private. Do not publish execution folders wholesale. Access to restricted inputs follows the article's Data availability statement.

## Version and verification scope

This distribution changes input-root resolution to an explicit environment variable. Analytical operations, thresholds, cohort roles and bootstrap settings are unchanged from the executed analysis. The distributed scripts have been syntax-checked. That check is not a second experiment or full numerical replay of this portable copy. The executed Ridge calculation was separately compared with scikit-learn SVD Ridge, and its saved aggregates were reconciled with the delivered workbook.

The reporting schedules reuse existing frozen decisions. They do not fit new gates or select thresholds on test outcomes. The target-only comparison evaluates ungated numerical predictions, not equivalent reporting performance. Both focal source configurations share the same target-only comparator.
