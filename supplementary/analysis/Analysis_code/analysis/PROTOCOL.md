# Bounded evidence augmentation, round 1

Locked before computing new summaries. Two focal configurations HZ_to_SA and HZ+HB_to_SA, all seeds 1000–1019, same 201 SAHZU test records and original 60-record target pool. No thresholds, mapping or reporting models are changed.

## A: target-prior reference
For each frozen run, use the median reference MA in its target adaptation pool as a constant test prediction. Compare MA absolute error with saved raw and final mapped predictions. No test outcome selects the constant. Report paired mean differences and 95% percentile intervals from 5,000 identity-proxy-group resamples, retaining all seeds together. This is a numerical reference, not an equivalent reporting policy. No claim about source-transfer value versus direct target-only learning follows from this comparison alone.

## B: timing policy replay
Replay three prespecified schedules: 300/600/900 seconds, 600/900 seconds, and 900 seconds only. Reuse saved checkpoint-specific actions and predicted classes verbatim. First accepted report locks both class and time. Unreported stays explicit. Report correct/incorrect/unreported yields, agreement, coverage, all 3-by-4 class destinations, first-report times and per-seed results including zero releases. Paired differences use the same 5,000 identity-proxy-group resamples. Primary contrasts are full sequence minus 10/15, and 10/15 minus fixed15; also retain full minus fixed15.

No policy will be relabelled as independently validated after this comparison on previously examined external records. No ranking-based threshold changes, clinical utility thresholds or multiple model search. Bootstrap intervals are descriptive and conditional on the frozen artifacts. Repeated seeds are not patients.

Validate alignment of record IDs, groups, truth, seeds and stored original full sequence. Verify pool/test identity-proxy separation. Hash all input files before/after. Outputs are aggregate only; private record arrays remain in original local archives. Original package and artifacts are read-only.
