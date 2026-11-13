# Supplementary Data

These eight workbooks are identical to the current submission files,
verified against the submission files on 13 November 2026. Read each workbook's README and Definitions
sheets for units, denominators and evaluation scope.

| Workbook | Contents |
| --- | --- |
| [Supplementary Data 1](Supplementary_Data_1.xlsx) | Cohort, data roles, nine configurations and evaluation denominators |
| [Supplementary Data 2](Supplementary_Data_2.xlsx) | Mapping, adaptation, Fig. 2 aggregates and linear/RBF comparisons |
| [Supplementary Data 3](Supplementary_Data_3.xlsx) | Reporting rules, class destinations, risk–coverage curves and Fig. 3 counts |
| [Supplementary Data 4](Supplementary_Data_4.xlsx) | Checkpoints, first-report schedules, HGB comparisons and Fig. 4 operating points |
| [Supplementary Data 5](Supplementary_Data_5.xlsx) | Seed summaries, representation comparisons, diagnostic analyses and computational timing |
| [Supplementary Data 6](Supplementary_Data_6.xlsx) | Timing destinations and target-only prediction comparisons |
| [Supplementary Data 7](Supplementary_Data_7.xlsx) | Local-label budgets, paired comparisons, schedule controls and internal checks |
| [Supplementary Data 8](Supplementary_Data_8.xlsx) | Separate 301-record internal evaluation, fold counts and linear/RBF comparisons |

Rows summarize configurations, models, seeds, folds, classes, checkpoints or
rank prefixes. Fold tables contain sample and class counts, not record
assignments. Time-to-first-report tables are count distributions.

The original 60/201 SAHZU evaluation and the separate 301-record internal
evaluation are identified separately. Agreement uses released reports;
coverage uses the complete evaluation cohort. FNR/FPR use released
abnormal/normal references, while full-class yields include withheld events.

All 98 worksheets and their workbook containers were checked for
patient-level fields, hidden content and embedded attachments. No individual
records, patient identifiers, raw signals or individual predictions were
found. Seeds and folds identify experimental runs or partitions.
File hashes, worksheet inventories and the review scope are recorded in
[`PUBLIC_DATA.json`](../provenance/PUBLIC_DATA.json).
