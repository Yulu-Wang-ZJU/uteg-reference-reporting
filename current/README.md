# Current manuscript analysis snapshot

This package contains analysis code, aggregate results and protocols for the
current manuscript. Individual research records are not included.

## Public repository

Analysis code and aggregate results correspond to [release v2026.11.07.6](https://github.com/Yulu-Wang-ZJU/uteg-reference-reporting/tree/v2026.11.07.6)
in [Yulu-Wang-ZJU/uteg-reference-reporting](https://github.com/Yulu-Wang-ZJU/uteg-reference-reporting).
Use [current/](https://github.com/Yulu-Wang-ZJU/uteg-reference-reporting/tree/v2026.11.07.6/current) for analysis sources and
[data/](https://github.com/Yulu-Wang-ZJU/uteg-reference-reporting/tree/v2026.11.07.6/data) for Supplementary Data 1-8.
The submission archive additionally includes the editable Fig. 1 source.

## Input data

The participating hospitals supplied U-TEG MA values and voltage recordings, together with the corresponding instrument-exported TEG readouts.
RESTRICTED_INPUT_HASHES.json identifies the input files. Input preparation and
loading are described in INPUT_CONTRACT.json and Supplementary Methods,
Part A1.

## Included cohort

The included cohort contains 478 records: 304 from SAHZU, 95 from Huzhou and 79 from Hubei. All included records passed the recorded input QC. Hubei has 40 adaptation and 39 test records. The two out-of-cohort imports are excluded from both current inputs and reporting denominators. See COHORT_UPDATE.json for the aggregate correction summary.

## Time windows and numerical processing

We index the U-TEG inputs on a 900-point grid at one-second intervals. The
5- and 10-minute windows cover the first 300 and 600 grid points. Processing
masks record the points retained for fitting.

Early coefficient preparation fills short masked prefixes with their last retained
value, divides each prepared prefix by its maximum absolute amplitude, fits the
model and restores amplitude units. These fits include filled positions.
`current_version.activate()` loads the saved early coefficients and the stored
full-record coefficients; it does not regenerate the early coefficient arrays.
Supplementary Information Part A1 describes this distinction.

For the separate EXP3/EXP4 representation control, `manuscript_signal.py` selects
the retained points and excludes filled positions before fitting. The shared
`reconstruction.shape()` centres model outputs by subtracting the median of their
first ten values. This operation belongs to model evaluation. Both fits use five
starts and at most 300 optimizer function evaluations per start; EXP4 also has a
2,000-residual-call cap per start. Predicted TEG reconstructions and display
extensions are model/plot outputs.

## Setup and entry points

Extract into a new directory. Install the Python dependencies in
requirements-observed.txt. Obtain the restricted files through the authorized
data-access process, then run:

```text
python configure_restricted_inputs.py PATH_TO_AUTHORIZED_INPUT_DIRECTORY
```

The setup checks every restricted-file hash before copying any input.
The original centre-role evaluation and 301-record grouped internal evaluation
remain separate:

| Entry | Purpose |
| --- | --- |
| `current_version evaluate` | 301-record grouped internal evaluation |
| `current_suite primary` | Original centre-role primary comparisons |
| `current_suite mapping --method all` | Mapping and calibration comparisons |
| `current_suite gates` | Reporting-rule comparisons |
| `current_suite sequence` | First-report accounting |
| `manuscript_delivery rescore` | Recalculate metrics from saved predictions |
| `manuscript_figure3`, `manuscript_figure4` | Current reporting figures |

Invoke these modules with `python -m code_submit.paper_inward4_replay.MODULE`.
Additional current analyses remain in their named modules and protocols.
Use the current entry points above. Shared helper modules retain their import
paths and supply numerical methods to these runners.

EVIDENCE.json and WORKBOOK_TABLES.json contain the aggregate results.
build_workbooks.mjs additionally requires @oai/artifact-tool to render workbooks.
The Fig. 4 plotting input is figure4_matched_aggregate.json.

## Packaging

CODE_MANIFEST.json lists the permitted package members and checksums.
`manuscript_delivery code` repacks this snapshot from that allowlist.
Keep the restricted inputs, folds, evaluation roles and numerical results fixed
when comparing this release with the manuscript.

## One-input linear/RBF comparison

Supplementary Data 2 and 8 include the separate 201-record and 301-record comparisons. `LINEAR_RBF_RESULTS.json` contains both fixed-penalty and development-selected results, including class errors, confusion counts and seed ranges. `protocols/LINEAR_RBF_PROTOCOL.json` specifies the matched loss, weights and selection procedure.

After configuring authorized inputs and generating the current primary and internal-301 mapping outputs, run `python -m code_submit.paper_inward4_replay.linear_rbf_comparison smoke`, then `full`, then `score`. These commands read the existing inputs and frozen splits. They save new comparison outputs without changing the primary results.
