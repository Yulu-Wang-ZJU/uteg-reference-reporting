# U-TEG reference reporting: code and aggregate data

Updated **7 November 2026** to match the current submission materials.
Start with [`current/`](current/) for the current analysis code and
[`data/`](data/) for Supplementary Data 1–8. The workbooks are exact copies
of the submission files. This repository contains no patient-level records,
raw waveforms, individual predictions, fitted models or manuscript PDFs.

The manuscript cites the fixed analysis version [`051be12`](https://github.com/Yulu-Wang-ZJU/uteg-reference-reporting/commit/051be12108fc7f7fd3faaaa689c8a6f87703084c).
Subsequent documentation and checksum updates retain the same analysis code
and aggregate data.

## Current analysis

The included cohort contains 478 records: 304 from SAHZU, 95 from Huzhou and
79 from Hubei. All included records passed the recorded input QC. The two
evaluation protocols remain separate:

- The original centre-role evaluation uses reserved adaptation and held-out
  test cohorts. The main single-source and dual-source comparisons share
  60 SAHZU adaptation records and 201 test records.
- The separate 301-record local evaluation uses grouped five-fold outer and
  three-fold inner splits, with one outer-held-out prediction per record.

The current sources cover hybrid regression, asymmetric correction, mapping
and adaptation controls, target-only RBF and Ridge comparisons, and
boundary-distance, logistic and HGB reporting rules. HGB capability
comparisons at matched coverage or observed agreement are distinct from the
supplementary B95 analyses that retain the first report.

MA states are low (`<50 mm`), normal (`50–70 mm`, inclusive) and high
(`>70 mm`). Coverage uses the complete evaluation denominator; agreement
uses released reports. FNR and FPR use released abnormal and normal
reference events, respectively. Full-class yields also include withheld
events. Repeated seed outcomes are retained within the same record group
when estimating intervals; they are not independent patients.

## Files and entry points

| Location | Contents |
| --- | --- |
| [`current/README.md`](current/README.md) | Input definitions, provenance, dependencies and execution instructions |
| [`current/code_submit/`](current/code_submit/) | Current runners and their shared dependencies |
| [`current/protocols/`](current/protocols/) | Frozen analysis settings and comparison protocols |
| [`current/EVIDENCE.json`](current/EVIDENCE.json) | Aggregate analysis results |
| [`current/WORKBOOK_TABLES.json`](current/WORKBOOK_TABLES.json) | Aggregate tables used to build the eight workbooks |
| [`data/`](data/) | Supplementary Data 1–8 and their index |
| [`provenance/CURRENT_CODE.json`](provenance/CURRENT_CODE.json) | Source-archive member hashes and the public-file inventory |
| [`provenance/PUBLIC_DATA.json`](provenance/PUBLIC_DATA.json) | Workbook hashes, worksheet inventory and content-review scope |
| [`provenance/CURRENT_MANUSCRIPT.json`](provenance/CURRENT_MANUSCRIPT.json) | Hashes linking this release to the local manuscript and figures |

The files under `current/` retain the submission code package's paths and
bytes. One editable manuscript-figure asset is omitted from the public copy;
the exclusion is recorded in `CURRENT_CODE.json`. The original
`current/CODE_MANIFEST.json` describes the submission archive, including that
asset. Use `provenance/CURRENT_CODE.json` to verify the public subset.

From `current/`, the principal analysis commands use
`python -m code_submit.paper_inward4_replay.MODULE`:

| Module and arguments | Analysis |
| --- | --- |
| `current_version evaluate` | 301-record grouped internal evaluation |
| `current_suite primary` | Original centre-role primary comparisons |
| `current_suite mapping --method all` | Mapping and calibration controls |
| `current_suite gates` | Reporting-rule comparisons |
| `current_suite sequence` | First-report accounting |
| `linear_rbf_comparison` | Matched one-input linear/RBF comparison; see the package instructions for its stages |
| `manuscript_figure3`, `manuscript_figure4` | Current reporting figures |

These numerical runners require the separately authorized inputs and frozen
assignments listed in `current/RESTRICTED_INPUT_HASHES.json`. Follow
[`current/README.md`](current/README.md) before configuring them.
`requirements-observed.txt` records the analysis environment; workbook
rendering also uses `@oai/artifact-tool`. The input-source account and its
recorded provenance qualification are retained in the package documentation.

## Verify the download

From the repository root, use Python 3.10 or later:

```text
python scripts/verify_current_release.py
```

This checks file hashes, the current Python/JSON sources and the aggregate
workbooks without importing an analysis runner or loading private data.
With the original workspace available, also compare the submission archive,
workbooks and manuscript hashes:

```text
python scripts/verify_current_release.py --source-root <workspace-root>
```

## Historical implementation

`src/`, `snapshots/` and the four bundles under `supplementary/` preserve the
earlier implementation and supplementary analyses. Their original provenance
manifests remain available. Their READMEs describe those earlier versions;
use `current/` for the current submission. The small package installed with
`pip install -e .` remains the earlier Ridge-10 utility API and does not
replace the current runners.

The repository's Apache-2.0 license is in [`LICENSE`](LICENSE).
