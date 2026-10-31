# Reporting diagnostics code bundle

This is the cleaned delivery bundle for the frozen 15-min aggregate diagnostics. The retained D1 freeze/evaluate core is AST-equivalent to the original implementation; only the obsolete equal_error_analysis.py D2/D3 functions and dispatch were removed. Use equal_error_postprocess.py for the corrected frontier and cross-calibration post-processing. The deprecated erroneous D2/D3 implementation is not included.

The bundle is source-relative under project/outputs/... and requires the frozen project inputs at their original relative locations: DATA15.npz, frozen joblib models, source action archives, development OOF aggregate rows, and the aggregate result files produced by the locked runs. It also requires NumPy, SciPy, joblib and matplotlib as appropriate. It does not contain patient identifiers, raw signals, row-level predictions, private NPZ archives, group/hash masks, or historical review/scoring files.

The public workbook is aggregate-only and cannot by itself reproduce group bootstrap intervals. Reproduction additionally requires controlled access to the paired data, frozen models and project-relative input tree. The scripts are preserved for audit and source-relative reproduction; this ZIP does not claim that a clean checkout can run without those restricted inputs. The present diagnostic results do not retrain final predictors. However, this provenance bundle includes dependency scripts with training or refit entry points (notably focused_reporting/run.py, target_ridge/run.py and the Ridge-10 fullchain runner), and increment_experiment.py can fit analysis-time Ridge OOF folds if explicitly invoked. Those entry points are included for controlled inspection/source-relative provenance only; they are not invoked automatically and must not be treated as part of the reported diagnostic run.

Analysis sequence using restricted inputs (these commands are not read-only: they may write derived outputs; they are intended to leave frozen input objects unchanged, so run in a controlled copy and verify input hashes):
1. outputs/uteg_reporting_increment_20261030/increment_experiment.py for the prior frozen incremental audit.
2. outputs/uteg_equal_error_20261030/equal_error_analysis.py freeze then evaluate for D1 policy/action materialization.
3. outputs/uteg_equal_error_20261030/equal_error_postprocess.py d2, d3, results for the corrected equal-error diagnostics.
4. outputs/uteg_reporting_increment_20261030/coverage_funnel.py for fixed 15-min center/class eligibility summaries.

The independent checker scripts are included for audit only. They are not a substitute for access to the restricted paired inputs, and their outputs must retain the stated post-hoc and non-independent-validation boundaries.
