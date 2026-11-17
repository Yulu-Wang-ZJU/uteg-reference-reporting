# Current 100-repeat evaluation

Main results use 60 local development records; the 54-record sensitivity and the 271-of-301 internal evaluation are separate. All 100 repeats are retained. The fixed composite rule is `common_B`; the separate ensemble extension is not substituted into the nine-configuration comparison.

`aggregate_results.zip` contains complete aggregate repeat records and summaries, component ablations, component timing and the fixed 201-record display check. It contains no individual patient inputs or prediction vectors. Each metric is calculated within a repeat before mean and sample SD are computed; missing denominators remain undefined. These SDs are not confidence limits.

`experiments/uteg_locked_expansion_20261117/experiment.py` is the repeated-evaluation entry point. Companion scripts generate matched controls and summaries. Code snapshots are retained under `experiments/` and `project/`; `PATH_RELOCATION.json` lists path-only portability edits. Training parameters and rule definitions are unchanged. Restricted inputs, pretrained source bundles, saved split assignments and input-window coefficients must be provided in the original relative layout under `project/` before numerical replay. See the archive-root input contract and configurator. No individual inputs are distributed here.

The risk cap is the 90th percentile of development normal-prediction high-risk scores. A normal report must pass its selected base gate, risk cap and amplitude bounds. Low/high predictions use separately selected gates. All selections use development predictions; final test outcomes do not choose a threshold. The separate ensemble has its own implementation and output scope.

The root code outside `repeat100/` reproduces the retained fixed-cohort analyses. H_ worksheet labels identify those results. Current figures use the 100-repeat summaries except Figure 2's prespecified full 201-record illustration and explicitly labelled fixed-cohort supplementary displays.

Timing covers the risk heads and decision step separately. It is not end-to-end acquisition-to-report latency. All 100 subsampling repeats differ from repeated CPU timing calls.

To reproduce aggregate plots, first run `python repeat100/unpack_aggregates.py`. The figure scripts accept `UTEG_WORKSPACE_ROOT` for the manuscript workspace. Figure 2 additionally requires the restricted paired traces and saved reference201 predictions; it does not fit or select models. `validate_aggregates.py` needs no patient data.
