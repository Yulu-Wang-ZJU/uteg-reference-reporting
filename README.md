# U-TEG reference-reporting code release

This is the local, manuscript-faithful code package assembled on 2026-10-30.
It preserves the current Ridge-10 translation/reporting implementation and
the source snapshots used by the recent baseline, supplementary and
sequence analyses.  It is deliberately a code-and-provenance release: no
raw records, patient identifiers, input archives, trained objects, prediction
arrays, result tables or manuscript source files are copied here.

The release is not a claim that the public folder alone reproduces the
reported numbers.  Full numerical replay still requires the private paired
U-TEG--TEG archives, frozen role/split manifests and the private trained
objects.  The historical code dependencies are included here; the package
does not copy the private inputs or infer them from the code.

## What is represented

The current manuscript contract is recorded in
[`provenance/SOURCE_PROVENANCE.json`](provenance/SOURCE_PROVENANCE.json),
including SHA-256 anchors for the current `main.tex` and
`supplement_current.tex` (the sources themselves are not bundled).

The curated source snapshots cover:

- Ridge-10 (`family=RIDGE`, `alpha=10`, four outputs `R`, `K`, `MA`,
  `alpha_R`) and the frozen MA-ERROR-LR gate;
- target-centre adaptation, including the auxiliary historical score `J`;
  `J` is not MA and does not define the 50/70-mm reporting states;
- the logistic/dual-head current-policy reporting path;
- the matched full-error-score histogram-gradient-boosting (HGB) baseline;
- 5-, 10- and 15-minute checkpoints (`300/600/900` seconds), with first
  report retained across checkpoints;
- baseline and supplementary lineage, including the nine-direction and
  four-way adaptation protocols.

`J` is retained as the source implementation’s auxiliary coagulation-index
score with coefficients `(-0.6516, -0.3772, 0.1224, 0.0759)` and intercept
`-7.7922`.  The reporting endpoint remains the fixed MA class rule:
`<50` low, `50--70` inclusive within range, and `>70` high.

## Safe entrypoints

From this directory, using Python 3.10+:

```text
python -m pip install -e .
python scripts/smoke_synthetic.py
python scripts/verify_provenance.py
```

The base install is sufficient for the synthetic API and provenance checks.
The historical runners use optional scientific dependencies; install them
only when a private replay is authorized:

```text
python -m pip install -e ".[historical]"
python -m pip install -e ".[figures]"       # optional SVG/PDF conversion
# or: python -m pip install -e ".[all]"
```

The extras provide libraries only. They do not contain the private paired
archives, frozen role/split files or trained objects required by a numerical
replay.

The synthetic smoke uses generated arrays only.  It checks the fixed MA
boundaries, 13-feature MA-ERROR-LR schema, gate fallback/scoring contract,
Ridge-10 forward mapping, affine/joint adaptation, auxiliary `J`, frozen
action logic and 5/10/15-minute first-report replay.  It does not read or
write any private workspace path.

To restore the exact workspace-relative paths expected by the historical
runners, first preview the plan and then explicitly materialize it into a
fresh private code root:

```text
python scripts/materialize_code.py --target-root <fresh-private-code-root>
python scripts/materialize_code.py --target-root <fresh-private-code-root> --write
```

The default is a dry-run.  `--write` copies only the code and protocol files
listed in the provenance manifest beneath the selected root; it copies no
`DATA15.npz`, `PREFIX_X.npz`, role/split manifest, `*.joblib`, prediction or
result file.  An existing root requires `--allow-existing`, and a differing
existing file additionally requires `--overwrite`.  After materialization,
follow the copied stage protocol with the separately authorized private
inputs and trained objects.

On Windows, use a short fresh target such as `E:/uteg_replay`; the materializer
checks every destination before creating directories and refuses a root that
would approach the legacy `MAX_PATH` limit, so a failed attempt cannot leave a
new partial copy.

To validate the dependency closure without private files or fitting, run the
materializer into a fresh temporary code root and then probe the imports in
isolated processes:

```text
python scripts/materialize_code.py --target-root <fresh-code-root> --write
python scripts/check_materialized_dependencies.py --target-root <fresh-code-root>
```

The probe covers the prefix construction/evaluator and adjacent runtimes,
nine-direction MA gate module, trust threshold core, scalar ordinal core and
D2 CI engine. It rejects private/model file extensions in the materialized
root and never calls a fit entrypoint. The data-eager Hubei QC runner is
explicitly reported as skipped because importing it would load its private
`STANDARDIZED_INPUT.npz` at module import time.

The current four-direction main sequence and the five-direction supplement
completion are checked together without fitting by:

```text
python scripts/check_nine_direction_integration.py
```

For an optional read-only local input check, supply private paths explicitly:

```text
python scripts/check_private_parity.py \
  --data15 <private-root>/outputs/uteg_legacy_aligned_threecenter_20261022_155614/DATA15.npz \
  --prefix-x <private-root>/outputs/uteg_legacy_aligned_threecenter_20261022_155614/PREFIX_X.npz
```

This command reads with `allow_pickle=False`, checks the expected `(480, 4)`
paired schema and `(2, 480, 4)` prefix schema, and prints only dimensions and
SHA-256 hashes.  It performs no fitting and writes nothing beside the inputs.
See [`config/private_layout.example.json`](config/private_layout.example.json)
for the intended private layout.

## Snapshots and historical replay

The 79 curated source snapshots are under [`snapshots/`](snapshots/), grouped
by stage; 78 are byte-for-byte identical to their source files.  The sole
documented exception is `snapshots/figures/figure2_times.py`: only its
workstation-specific Anaconda DLL path is replaced by
`UTEG_CAIROSVG_DLL_ROOT`/`CONDA_PREFIX` resolution, with no algorithm change.
In addition to the recent runners, the `snapshots/historical/` tree contains
the direct code dependencies for generalized-exponential prefix construction,
target-side Huber/CI-auxiliary adaptation, the 13-feature error-risk models,
and the low/high dual-head contract/selector.  Their original
workspace-relative paths, source and snapshot hashes, roles and privacy scan
are in the provenance manifest.  The historical scripts
intentionally retain their original relative path contracts and may
dynamically load private inputs or frozen objects.  Use
`scripts/materialize_code.py` to restore the code side of those contracts;
then perform a full private replay with the corresponding private inputs and
frozen objects, following each stage’s copied `PROTOCOL`/`REPRODUCE` file.
The release folder remains code-only and is not a replacement for those
private artifacts.

The small importable API in [`src/uteg_reference_reporting/core.py`](src/uteg_reference_reporting/core.py)
is an exact-contract extraction for smoke tests and controlled private reuse.
It does not silently substitute a new model or change the scientific protocol.

Figure/Table and nine-direction source lineage is recorded in
[`provenance/MANUSCRIPT_LINEAGE.json`](provenance/MANUSCRIPT_LINEAGE.json).
Run `python scripts/verify_lineage.py` from the workspace to hash-check the
current aggregate artifacts and every copied code snapshot against its source.
The map records private prerequisite paths but does not copy them.

## Provenance and exclusions

Run `python scripts/build_provenance.py` only after changing a deliberate
snapshot.  It rewrites the manifest with source/snapshot hashes and anchors.
`python scripts/verify_provenance.py` checks the snapshots and, when the
original workspace is present, checks the original hashes too.

The excluded input classes are explicit in the manifest: `DATA15.npz`,
`PREFIX_X.npz`, role/split manifests, `*.joblib` trained objects, prediction
archives and generated result tables.  The root Apache-2.0 license is copied
to [`LICENSE`](LICENSE).
