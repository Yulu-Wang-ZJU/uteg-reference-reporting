# Target-only sequential comparison: public protocol

This directory specifies the frozen target-only comparator used alongside the
focused source-transfer workflow.

## Matched design

- Use the same 60 target-centre development records, 201 held-out records and
  20 seeds (1000--1019) as the focused comparison. The three checkpoints share
  the 60-record label budget and the frozen record, group and fold identities.
- Evaluate 300, 600 and 900 seconds. The 300- and 600-second predictors use
  the corresponding observed-prefix representations; the 900-second
  target-only objects are the frozen 15-minute objects already used by the
  focused analysis.
- At 300 and 600 seconds, choose Ridge penalties from
  `[0.1, 1, 10, 100]` by grouped nested development folds using MA absolute
  error. Fit the stored MA-error gate and supported low/high heads on the
  development side, then freeze thresholds before any held-out evaluation.
- Replay fixed-5, fixed-10, fixed-15, 5--10, 10--15 and 5--10--15 schedules.
  A record--seed event keeps its first released class and time; later reports
  cannot replace an earlier release. Zero-report events and seeds remain in
  the denominator.
- Report correct, incorrect and unreported events, coverage, agreement,
  class-specific destinations, first-report times and per-seed outcomes.
  Paired descriptive intervals use 5,000 draws of the 190 identity-proxy
  groups with all seeds retained within each sampled group and RNG seed
  `20261030`.

## Input boundary and reporting scope

The implementation consumes the archived restricted inputs, frozen source
objects, selectors and project layout. This code package contains none of the
patient data, record identifiers, partition arrays, fitted models, private
prediction/action arrays or result workbooks needed to run it. Numerical
outputs are therefore conditional descriptive results from the matched cohort,
not external validation or a claim of clinical utility. Generated models,
freeze manifests, private arrays, individual predictions and clinical-data
workbooks remain in the controlled analysis workspace.

