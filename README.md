# U-TEG reference reporting

Current release: **v2026.11.17.1** (17 November 2026).

The main evaluation uses 60 local development records and all 100 stratified subsampling repeats. The 54-record sensitivity and the 271-of-301 internal evaluation are separate. The fixed composite reporting rule is `common_B`; ensemble extensions are reported separately. Mean ± SD describes variation across overlapping repeats, not 95% confidence limits.

Start with [the current repeated evaluation](current/repeat100/README.md). Complete aggregate repeat records are in `current/repeat100/aggregate_results.zip`. Run `python current/repeat100/validate_aggregates.py` to verify the reporting summaries without patient data. The [eight supplementary workbooks](data/) are identical to the submission files; H_ sheets retain the original fixed-cohort analyses.

The included cohort contains 478 records: 304 from SAHZU, 95 from Huzhou and 79 from Hubei. Main transfer comparisons share the original 60-record development pool and 201-record evaluation pool. Each repeat samples 181 evaluation records and designated source records. The additional local internal analysis samples 271 of 301 records and uses grouped outer and inner folds.

Accuracy is correct MA-state reports divided by reports. Reporting rate is reports divided by all sampled test records. FNR and FPR use reported abnormal and normal references, respectively; zero denominators remain undefined. Low/high correct-report counts and withheld outcomes are retained separately.

The U-TEG device uses longitudinal ultrasonic guided waves in a micro-metal tube, as described in US patent 12,487,161 B2 (2025). At SAHZU and Hubei, samples were kept at room temperature and tested within 6 h. At Huzhou, samples were stored at 2–8 °C and tested at 24–48 h. Both assays began simultaneously on aliquots from the same blood draw.

This repository contains code, aggregate tables and manuscript checksums. It does not publish individual patient inputs, prediction vectors, manuscript PDFs or the device video. The video remains in the submission ZIP.

| Location | Contents |
|---|---|
| `current/repeat100/` | Current fixed-rule implementation, 100-repeat summaries, component ablation and timing |
| `current/code_submit/`, `current/protocols/` | Retained fixed-cohort implementation and protocols |
| `current/WORKBOOK_TABLES.json` | Exact source tables for all eight workbooks |
| `data/` | Supplementary Data 1–8 and their ZIP |
| `provenance/CURRENT_CODE.json` | Public code inventory and archive hashes |
| `provenance/PUBLIC_DATA.json` | Aggregate data hashes and source correspondence |
| `provenance/CURRENT_MANUSCRIPT.json` | Current manuscript, figures and video hashes |

Run `python scripts/verify_current_release.py` to check the public inventory. With the authorized local workspace available, add `--source-root PATH` to check manuscript and archive correspondence. No analyses are executed by this verification.
