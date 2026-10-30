# One nonlinear risk baseline for Table 2

Two manuscript directions, HZ_to_SA and HZ+HB_to_SA; 15 minutes; all 20 seeds 1000–1019; original 60-record calibration pools and 201-record test cohort. No new fitting or model search. Reuse fixed MA13D equal-weight HGB models after exact upstream-object hash checks. Historical within-only hybrid selection is NOT reused: this comparison uses HGB error scores for every predicted class.

Compare against original MA-ERROR-LR with identical mapping, calibration, predictions, QC, 13 features, and held-out error labels. HGB configuration: learning rate .05, 100 iterations, 4 leaves, depth 2, minimum leaf 10, L2=10, 255 bins, no early stopping, outer seed. Use historical pure threshold selector on merged outer-B scores: maximize coverage subject to empirical error <=15%, at least 5 releases and >=20% development coverage, preserving ties. Failure yields no report. Freeze all 40 thresholds before test evaluation. No changes after test inspection.

Report all results including poor performance; no promotion of HGB based on test performance. Original manuscript results and figures remain untouched. Cross-seed events are not independent patients. The earlier HGB development non-promotion remains part of provenance; this is a newly authorized matched baseline evaluation, not a claim that the historical promotion test passed.
