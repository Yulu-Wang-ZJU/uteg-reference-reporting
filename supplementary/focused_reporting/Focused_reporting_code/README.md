# Matched target-only reporting analysis (Supplementary Data 7)

This archive contains exact copies of the executed focused-comparison scripts and protocol. It supplements, rather than replaces, Analysis_code.zip and the public core workflow at commit 5f32af59907470ced142595625c4b80555fd1d81. No patient data, fitted objects, record assignments or individual predictions are distributed.

## Archive layout and inputs

For authorized replay, place the `focused_reporting_20261030` folder beside `target_ridge` from Analysis_code.zip, inside a new writable analysis directory under the custodian's archived `outputs` directory. The script's `ROOT` is its own folder, `G` is that analysis directory, and `OUT` is the archive directory one level above G. Preserve this layout; these are archived project-relative scripts, not a standalone package. The run.py constants identify the required restricted data, original fitted objects, risk-contract functions, selectors and saved reference actions. Python, NumPy, SciPy and joblib are required.

The target_ridge module must be present for numerical helper functions; the focused run itself reconstructs the final local predictors and verifies their identity against the already evaluated target-only predictors. No previous reporting gate is transferred to the local learner.

## Execution on an authorized, fresh output directory

1. Read PROTOCOL.md and supply the restricted archived inputs at their recorded paths.
2. Run `python focused_reporting_20261030/run.py train` to fit the finite matched development interface and write immutable models/thresholds and FREEZE.json.
3. Run `python focused_reporting_20261030/evaluate_schema_fix.py` to evaluate the frozen models. Use this wrapper, not `run.py evaluate`: the historical reference-action archive has no explicit seed column. The wrapper checks test IDs, all MA values and classes against the seed-labelled prediction archive without changing fitting or threshold selection.
4. Run `python focused_reporting_20261030/verify.py` to reconcile frozen hashes, actions and aggregate counts without fitting.

Existing output files are not overwritten. Fitted models and freeze manifests contain restricted data-role information and must remain private. Do not publish execution directories wholesale. The code copies were syntax-checked; this distribution was not rerun as another experiment. Access to restricted inputs follows the article's Data availability statement.

## Statistical scope

One shared target-only comparator, two original source workflows, 15-minute inputs and all 20 seeds. Supplementary Data 7 provides aggregate working points, paired descriptive intervals, seed outcomes and class destinations. This analysis uses a previously examined cohort and does not establish independent validation, non-inferiority or clinical utility.
