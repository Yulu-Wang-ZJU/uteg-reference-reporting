# Current 100-repeat evaluation

The main SAHZU evaluation uses the full 60-record development pool and 220 of 244 test records in each repeat (7 low, 187 normal, 26 high MA). Sensitivity to 90% development sampling uses 54 records and is reported separately. All 100 repeats are retained. Other target-centre and internal evaluations retain their documented assignments and rules.

The Class-specific reporting gate is stored as `common_B` in aggregate tables. In the current SAHZU analysis, normal predictions pass both low- and high-MA risk checks, amplitude bounds and a selected normal-state gate. Each risk cutoff is the 90th percentile of out-of-fold scores among development predictions classified as normal. After fixing normal decisions, the existing development-side selector chooses gates for low/high predictions. No cutoff is selected using test labels.

With the full development pool, all 100 repeats had zero released FN at 15 minutes for the current SAHZU workflows. This result does not apply to every control, the sampled-development sensitivity, early checkpoints or every target centre. FNR uses released abnormal references; FPR uses released normal references. Mean ± sample SD summarizes repeat-level ratios over overlapping subsets, not independent-cohort confidence limits. Undefined ratios retain their valid-repeat counts.

`aggregate_results.zip` contains aggregate repeat outputs and summaries. The reference244 folder holds the fixed full-test-cohort Figure 2 metrics. Original fixed-cohort sheets have an H_ prefix. Representation controls and internal evaluations retain their separate scope.

Current numerical entry: `experiments/uteg_sahzu_locked_completion_20261118/complete.py`; `derive.py` summarizes saved outputs. The fixed split/initial fitting runner is `experiments/uteg_sahzu60_test244_20261118/run.py`. `CURRENT_PATH_RELOCATION.json` records path-only edits to executed dependencies. Restricted inputs, source bundles and saved assignments must be supplied before numerical replay. In particular, `experiments/uteg_sahzu60_test244_20261118/private/SPLITS.json` holds the fixed 200 assignments. These individual-level files are not distributed publicly. The archive-root configurator covers original input files, not these additional assignments.

The source projects are under `project/`; analysis runners are under `experiments/`. Earlier code remains available for the labelled retained analyses. Gate application timing excludes acquisition and model fitting; five timing repetitions are separate from the 100 data-subsampling repeats.

Run `python validate_aggregates.py` to check the current aggregate summaries without individual inputs. Plotting scripts require the restricted inputs for waveform panels; plots based only on aggregate summaries can be regenerated from the JSON archive.
