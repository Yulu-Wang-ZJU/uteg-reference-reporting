#!/usr/bin/env python3
"""Regenerate the release provenance manifest from exact snapshots."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


RELEASE = Path(__file__).resolve().parents[1]
PROJECT = RELEASE.parents[1]

SNAPSHOTS = [
    ("ridge10/runtime.py", "outputs/uteg_signal_model_goal_20261026_000929/ridge10_fullchain_20261026_122551/runtime.py", "runtime/cache isolation and historical dependency binding"),
    ("ridge10/run_ridge10_fullchain.py", "outputs/uteg_signal_model_goal_20261026_000929/ridge10_fullchain_20261026_122551/run_ridge10_fullchain.py", "Ridge-10 four-output mapping, adaptation and MA-ERROR-LR gate"),
    ("ridge10/evaluate_frozen_actions.py", "outputs/uteg_signal_model_goal_20261026_000929/ridge10_fullchain_20261026_122551/evaluate_frozen_actions.py", "frozen action evaluator and record-cluster summaries"),
    ("ridge10/freeze_ma_thresholds.py", "outputs/uteg_signal_model_goal_20261026_000929/ridge10_fullchain_20261026_122551/freeze_ma_thresholds.py", "finite outer-B score threshold freeze"),
    ("ridge10/preflight_ridge10_fullchain.py", "outputs/uteg_signal_model_goal_20261026_000929/ridge10_fullchain_20261026_122551/preflight_ridge10_fullchain.py", "read-only Ridge-10/source identity preflight"),
    ("supplement/batch.py", "outputs/uteg_supplement_completion_20261027_203140/batch.py", "supplement task plan and frozen training orchestration"),
    ("supplement/analyze.py", "outputs/uteg_supplement_completion_20261027_203140/analyze.py", "mapping, gate and temporal analysis"),
    ("supplement/finalize.py", "outputs/uteg_supplement_completion_20261027_203140/finalize.py", "independent appendix verification"),
    ("supplement/launch.py", "outputs/uteg_supplement_completion_20261027_203140/launch.py", "supplement launch entrypoint"),
    ("supplement/render.py", "outputs/uteg_supplement_completion_20261027_203140/render.py", "supplement figure rendering"),
    ("sequence/run.py", "outputs/uteg_current_ma_sequence_20261026/run.py", "5/10-minute current-policy prefix fitting and dual heads"),
    ("sequence/evaluate.py", "outputs/uteg_current_ma_sequence_20261026/evaluate.py", "frozen prefix scoring and first-report replay"),
    ("sequence/verify_saved.py", "outputs/uteg_current_ma_sequence_20261026/verify_saved.py", "saved sequence verification"),
    ("hgb/run.py", "outputs/uteg_table2_hgb_20261029/run.py", "matched HGB full-error-score baseline"),
    ("hgb/verify.py", "outputs/uteg_table2_hgb_20261029/verify.py", "HGB baseline verification"),
    ("adaptation/run.py", "outputs/uteg_adaptation_fourway_20261029/run.py", "raw, affine, joint and current adaptation comparison"),
    ("adaptation/verify.py", "outputs/uteg_adaptation_fourway_20261029/verify.py", "adaptation comparison verification"),
    ("historical/legacy/runtime.py", "outputs/uteg_legacy_aligned_threecenter_20261022_155614/runtime.py", "historical target/prefix runtime"),
    ("historical/legacy/train_target_legacy.py", "outputs/uteg_legacy_aligned_threecenter_20261022_155614/train_target_legacy.py", "target-side Huber and CI-auxiliary adaptation chain"),
    ("historical/legacy/downstream_15min.py", "outputs/uteg_legacy_aligned_threecenter_20261022_155614/downstream_15min.py", "legacy 15-minute downstream Huber/adapter/gate path"),
    ("historical/legacy/build_prefix_inputs.py", "outputs/uteg_legacy_aligned_threecenter_20261022_155614/build_prefix_inputs.py", "generalized-exponential U-TEG prefix preprocessing and fitting"),
    ("historical/legacy/prefix_source.py", "outputs/uteg_legacy_aligned_threecenter_20261022_155614/prefix_source.py", "frozen-config prefix source mapping"),
    ("historical/legacy/evaluate_prefix.py", "outputs/uteg_legacy_aligned_threecenter_20261022_155614/evaluate_prefix.py", "frozen prefix external replay"),
    ("historical/prefix/platform_process.py", "outputs/uteg_platform_filter_pilot_20261006_124636/process.py", "platform filtering and prefix preprocessing dependency"),
    ("historical/prefix/hubei_process.py", "outputs/uteg_hubei_qc_repair_20261021_171418/run.py", "Hubei fixed-length prefix QC and preprocessing dependency"),
    ("historical/prefix/curves.py", "outputs/uteg_platform_param_compare_20261006_125428/curves.py", "platform generalized-exponential curve fitting dependency"),
    ("historical/prefix/zero5_refit.py", "outputs/uteg_zero5_med4_20261006_113257/refit.py", "zero-tail curve refit dependency"),
    ("historical/hgb/old_runtime.py", "outputs/uteg_signal_model_goal_20261026_000929/ma_tail_weight_gate_run_20261026_141553/runtime.py", "weighted-gate runtime dependency for the HGB lineage"),
    ("historical/hgb/run_weighted_gate.py", "outputs/uteg_signal_model_goal_20261026_000929/ma_tail_weight_gate_run_20261026_141553/run_weighted_gate.py", "reviewed weighted MA gate implementation used by HGB selector"),
    ("historical/hgb/selector_weighted_gate.py", "outputs/uteg_signal_model_goal_20261026_000929/ma_tail_weight_gate_run_20261026_141553/selector_weighted_gate.py", "weighted-gate outer-B selector dependency"),
    ("historical/hgb/runtime.py", "outputs/uteg_signal_model_goal_20261026_000929/ma_tail_weight_gate_run_20261026_141553/ridge_ma13d_hgb_prepare_20261026_151041/runtime.py", "Ridge+MA13D-HGB run-local runtime"),
    ("historical/hgb/run_hgb_ma13d_driver.py", "outputs/uteg_signal_model_goal_20261026_000929/ma_tail_weight_gate_run_20261026_141553/ridge_ma13d_hgb_prepare_20261026_151041/run_hgb_ma13d_driver.py", "authorized Ridge+MA13D-HGB fitting driver"),
    ("historical/hgb/run_hgb_development_selector.py", "outputs/uteg_signal_model_goal_20261026_000929/ma_tail_weight_gate_run_20261026_141553/ridge_ma13d_hgb_prepare_20261026_151041/run_hgb_development_selector.py", "Ridge+MA13D-HGB development selector"),
    ("historical/hgb/run_ridge_ma13d_hgb_prepare.py", "outputs/uteg_signal_model_goal_20261026_000929/ma_tail_weight_gate_run_20261026_141553/ridge_ma13d_hgb_prepare_20261026_151041/run_ridge_ma13d_hgb_prepare.py", "HGB preflight/prepare entrypoint"),
    ("figures/formal_generate_figures.py", "outputs/uteg_manuscript_figures_20261024_234843/build/generate_figures.py", "editable Figure 1--4 formal figure generator"),
    ("figures/formal_validate_figures.py", "outputs/uteg_manuscript_figures_20261024_234843/build/validate_formal_figures.py", "formal Figure 1--4 artifact validator"),
    ("figures/results_build_figures.py", "outputs/uteg_results_figures_20261026/build_figures.py", "current Figure 2--4 plotting and aggregate source-data builder"),
    ("figures/results_unify_current.py", "outputs/uteg_results_figures_20261026/unify_current.py", "current-policy figure gallery and source identity builder"),
    ("figures/figure2_times.py", "outputs/uteg_manuscript_scaffold_20261008_185739/editorial/figure2_times.py", "Figure 2 font/format conversion; workstation DLL path portable redaction"),
    ("figures/redesign_figure3.py", "outputs/uteg_manuscript_scaffold_20261008_185739/editorial/redesign_figure3.py", "current Figure 3 frozen-statistics redraw"),
    ("figures/figure4_main_pair.py", "outputs/uteg_manuscript_scaffold_20261008_185739/editorial/figure4_main_pair.py", "current Figure 4 first-report paired trajectory generator"),
    ("figures/appendix_build.py", "outputs/uteg_appendix_review_20261028/build.py", "current supplementary figure/table assembly"),
    ("figures/appendix_check.py", "outputs/uteg_appendix_review_20261028/check.py", "supplementary artifact and link checker"),
    ("figures/revise_figure1_f.ps1", "outputs/uteg_manuscript_scaffold_20261008_185739/editorial/revise_figure1_f.ps1", "Figure 1 editable source revision"),
    ("figures/export_terminology_figure1_20261029.ps1", "outputs/uteg_manuscript_scaffold_20261008_185739/editorial/export_terminology_figure1_20261029.ps1", "Figure 1 terminology-aligned export"),
    ("figures/export_aligned_figure1.ps1", "outputs/uteg_manuscript_scaffold_20261008_185739/editorial/export_aligned_figure1.ps1", "Figure 1 aligned export"),
    ("historical/legacy/evaluate_15min.py", "outputs/uteg_legacy_aligned_threecenter_20261022_155614/evaluate_15min.py", "prefix evaluator dependency for frozen prediction loading"),
    ("historical/gates/nine_runtime.py", "outputs/uteg_nine_direction_ma_20261025_195548/code/nine_runtime.py", "nine-direction runtime imported by MA gate fitting"),
    ("historical/gates/independent_models.py", "outputs/uteg_trust_gate_selection_20261004_205412/independent_models.py", "independent gate model scorer loaded by threshold utilities"),
    ("historical/gates/trust_runtime.py", "outputs/uteg_trust_gate_selection_20261004_205412/runtime.py", "independent gate runtime dependency"),
    ("historical/gates/trust_threshold_runtime.py", "outputs/uteg_trust_threshold90_20261006_161248/runtime.py", "threshold utility runtime dependency"),
    ("historical/prefix/platform_runtime.py", "outputs/uteg_platform_param_compare_20261006_125428/runtime.py", "platform curve fitting runtime dependency"),
    ("historical/prefix/zero5_runtime.py", "outputs/uteg_zero5_med4_20261006_113257/runtime.py", "zero-tail refit runtime dependency"),
    ("historical/adaptation/d2_ci_runtime.py", "outputs/uteg_d2_ci_native85_20261007_012322/runtime.py", "D2 CI adaptation runtime wrapper"),
    ("historical/adaptation/d2_base_runtime.py", "outputs/uteg_d2_adapt60_trust_20261007_005909/runtime.py", "D2 base adaptation runtime"),
    ("historical/adaptation/d2_core.py", "outputs/uteg_d2_adapt60_trust_20261007_005909/core.py", "D2 base Huber/CI/gate core loaded by CI engine"),
    ("historical/adaptation/d2_audit.py", "outputs/uteg_d2_adapt60_trust_20261007_005909/audit.py", "D2 independent audit and adapter scorer"),
    ("historical/adaptation/crossfit_core.py", "outputs/uteg_trust_crossfit40_20261006_170625/core.py", "cross-fit split/calibration dependency"),
    ("historical/adaptation/scalar_runtime.py", "outputs/uteg_scalar_ordinal_trust_20261006_181750/runtime.py", "scalar ordinal runtime dependency"),
    ("historical/adaptation/fourparam_verify.py", "outputs/uteg_fourparam_mapping_trust_20261006_132600/verify.py", "four-parameter source function dependency"),
    ("historical/adaptation/zero5_verify.py", "outputs/uteg_zero5_med4_20261006_113257/verify_independent.py", "zero-tail source function dependency"),
    ("historical/adaptation/clinical_verify.py", "outputs/uteg_med4_source_clinical_20261006_104632/verify_independent.py", "clinical source function dependency"),
    ("historical/models/models.py", "outputs/uteg_hill_exp_matched_20261012_153337/models.py", "historical mapper families and Huber/auxiliary adapter assembly"),
    ("historical/models/early_decision_common.py", "outputs/uteg_med4_early_decision_20261004_191744/common.py", "shared projection, CI labels, calibration and prediction"),
    ("historical/models/capacity_core.py", "outputs/uteg_med4_capacity_20261006_001701/core.py", "capacity/model calibration helpers"),
    ("historical/models/source_clinical_core.py", "outputs/uteg_med4_source_clinical_20261006_104632/core.py", "source clinical loss/weight helpers"),
    ("historical/models/fourparam_mapping_core.py", "outputs/uteg_fourparam_mapping_trust_20261006_132600/core.py", "four-parameter mapper fit/predict"),
    ("historical/models/scalar_ordinal_core.py", "outputs/uteg_scalar_ordinal_trust_20261006_181750/core.py", "scalar ordinal auxiliary selection"),
    ("historical/models/platform_param_core.py", "outputs/uteg_platform_param_compare_20261006_125428/core.py", "platform parameter representation helpers"),
    ("historical/models/zero5_core.py", "outputs/uteg_zero5_med4_20261006_113257/core.py", "zero-tail model helper"),
    ("historical/gates/independent_statistics.py", "outputs/uteg_trust_gate_selection_20261004_205412/independent_statistics.py", "historical gate statistics"),
    ("historical/gates/risk_models.py", "outputs/uteg_trust_gate_selection_20261004_205412/risk_models.py", "13-feature error-risk models"),
    ("historical/gates/trust_threshold_core.py", "outputs/uteg_trust_threshold90_20261006_161248/core.py", "threshold and bootstrap utilities"),
    ("historical/gates/c60_models.py", "outputs/uteg_c60_new_trust_20261007_170322/models.py", "C60 gate models"),
    ("historical/gates/c60_stats.py", "outputs/uteg_c60_new_trust_20261007_170322/stats.py", "C60 gate statistics"),
    ("historical/adaptation/d2_ci_engine.py", "outputs/uteg_d2_ci_native85_20261007_012322/engine.py", "CI auxiliary adapter fit/adapt/select"),
    ("historical/gates/fit_ma_gates.py", "outputs/uteg_nine_direction_ma_20261025_195548/code/fit_ma_gates.py", "MA-ERROR-LR and finite threshold selector"),
    ("historical/dualhead/dual_head_contract.py", "outputs/uteg_ridge_dual_head_prepare_20261026_172952/dual_head_contract.py", "low/high dual-head feature and support contract"),
    ("historical/dualhead/run_unified_selector.py", "outputs/uteg_ridge_dual_head_unified_selector_20261026_190142/run_unified_selector.py", "development-only unified dual-head selector"),
]

DOCS = [
    ("snapshots/ridge10/PROTOCOL.md", "outputs/uteg_signal_model_goal_20261026_000929/ridge10_fullchain_20261026_122551/PROTOCOL.md", "Ridge-10 fullchain protocol"),
    ("snapshots/sequence/PROTOCOL.txt", "outputs/uteg_current_ma_sequence_20261026/PROTOCOL.txt", "current MA sequence protocol"),
    ("snapshots/hgb/PROTOCOL.md", "outputs/uteg_table2_hgb_20261029/PROTOCOL.md", "matched HGB baseline protocol"),
    ("snapshots/adaptation/PROTOCOL.md", "outputs/uteg_adaptation_fourway_20261029/PROTOCOL.md", "four-way adaptation protocol"),
    ("snapshots/supplement/PROTOCOL.md", "outputs/uteg_supplement_completion_20261027_203140/PROTOCOL.md", "supplement completion protocol"),
    ("snapshots/supplement/REPRODUCE.md", "outputs/uteg_supplement_completion_20261027_203140/REPRODUCE.md", "historical reproduction notes"),
]

REDACTIONS = {
    "figures/figure2_times.py": [
        "replaced the source's workstation-specific Anaconda DLL path with UTEG_CAIROSVG_DLL_ROOT or CONDA_PREFIX resolution"
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def entry(snapshot_rel: str, source_rel: str, role: str) -> dict[str, object]:
    snapshot = RELEASE / "snapshots" / snapshot_rel
    source = PROJECT / source_rel
    text = snapshot.read_text(encoding="utf-8-sig")
    suspicious = {
        # Avoid treating the ``https://`` in figure citations as a drive path.
        "absolute_windows_path": bool(re.search(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]", text)),
        "absolute_unix_path": bool(re.search(r"/(?:Users|home)/", text)),
        "credential_marker": bool(re.search(r"BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY", text)),
        "email_marker": bool(re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}", text)),
    }
    redactions = REDACTIONS.get(snapshot_rel, [])
    return {
        "snapshot_path": f"snapshots/{snapshot_rel}",
        "source_path": source_rel,
        "role": role,
        "copied_exact": not redactions,
        "redactions": redactions,
        "source_sha256": sha256(source) if source.exists() else None,
        "snapshot_sha256": sha256(snapshot),
        "bytes": snapshot.stat().st_size,
        "privacy_scan": suspicious,
    }


def main() -> None:
    snapshots = [entry(*item) for item in SNAPSHOTS]
    docs = []
    for snapshot_rel, source_rel, role in DOCS:
        snapshot = RELEASE / snapshot_rel
        source = PROJECT / source_rel
        docs.append({
            "snapshot_path": snapshot_rel,
            "source_path": source_rel,
            "role": role,
            "copied_exact": True,
            "source_sha256": sha256(source) if source.exists() else None,
            "snapshot_sha256": sha256(snapshot),
            "bytes": snapshot.stat().st_size,
        })
    manuscript = []
    for rel in (
        "outputs/uteg_manuscript_scaffold_20261008_185739/main.tex",
        "outputs/uteg_manuscript_scaffold_20261008_185739/supplement_current.tex",
    ):
        path = PROJECT / rel
        manuscript.append({"source_path": rel, "sha256": sha256(path), "bytes": path.stat().st_size})
    manifest = {
        "schema": "uteg_reference_reporting_source_provenance_v1",
        "release": "uteg_reference_reporting_20261030",
        "created": "2026-10-30",
        "snapshot_policy": "curated exact code and protocol snapshots unless an entry documents a workstation-path redaction; no data, trained objects, result tables or manuscript sources copied",
        "manuscript_anchors": manuscript,
        "snapshots": snapshots,
        "protocols": docs,
        "excluded_private_inputs": [
            {"path": "outputs/uteg_legacy_aligned_threecenter_20261022_155614/DATA15.npz", "reason": "paired U-TEG--TEG records and identifiers; private"},
            {"path": "outputs/uteg_legacy_aligned_threecenter_20261022_155614/PREFIX_X.npz", "reason": "prefix signal archive; private"},
            {"path": "outputs/uteg_legacy_aligned_threecenter_20261022_155614/ROLE_FREEZE.json", "reason": "record-role manifest; private"},
            {"path": "outputs/**/models/**/*.joblib", "reason": "trained objects are intentionally not distributed"},
            {"path": "outputs/**/eval/**/*.npz", "reason": "predictions/results may contain record-level material"},
        ],
        "model_contract": {
            "mapper": "RIDGE-10 (family RIDGE, alpha 10, four outputs R/K/MA/alpha_R)",
            "adaptation_auxiliary": "J is the historical auxiliary coagulation-index score; it is not MA and does not define reporting labels",
            "risk_heads": ["MA-ERROR-LR", "dual low/high heads", "matched HGB full-error-score baseline"],
            "checkpoints_seconds": [300, 600, 900],
            "ma_boundaries_mm": [50, 70],
            "seeds": [1000, 1019],
            "directions": 9,
        },
    }
    path = RELEASE / "provenance" / "SOURCE_PROVENANCE.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
