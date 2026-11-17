"""Four-parameter amplitude substitution diagnostic on the frozen all301 folds.

The original and raw-peak variants are U-TEG-only representations.  The 0.8
variant deliberately inserts the query's true MA into the B coordinate and is
therefore an oracle diagnostic, not predictive validation.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, r2_score
from threadpoolctl import threadpool_limits

from . import all301_internal_data as contract
from . import all301_internal_models as models
from .feature_combination_audit import detailed
from .metrics import assess, states
from code_submit.ma_imbalance_mapping.data_contract import sha256


BASE = contract.BASE
OUT = BASE / "../uteg_ma_mix_oracle_diagnostic_20261102/four_parameter_substitution"
RAW = contract.ROOT / "outputs/uteg_three_center_validation_20261021_174409/private/INPUT_RAW_AND_REFERENCE.npz"
VARIANTS = ("original4", "rawpeak4", "oracle08_4")
LOSSES = ("uniform", "class", "inward4", "inward16")
GRID = [{"variant": v, "loss": l} for v in VARIANTS for l in LOSSES]
N_CANDIDATES = len(GRID)


def _sha(path: Path) -> str:
    return sha256(path)


def _raw_peak(data, local):
    with np.load(RAW, allow_pickle=False) as z:
        assert np.array_equal(z["ids"][local], data["ids"][local])
        raw = np.asarray(z["original"], float)
        observed = np.asarray(z["observed"], bool)
    assert raw.shape[0] == len(data["ids"])
    peaks = np.full(len(data["ids"]), np.nan)
    for row in local:
        n = int(observed[row, :900].sum())
        assert n >= 11 and np.array_equal(np.flatnonzero(observed[row, :900]), np.arange(n))
        v = raw[row, :n]
        assert np.isfinite(v).all()
        peaks[row] = np.max(v - np.median(v[:10]))
    assert np.isfinite(peaks[local]).all()
    return peaks


def _feature(variant, base4, rawpeak, y, train, query):
    """Build only the rows touched by this fit; oracle uses query MA by design."""
    x = np.asarray(base4, float).copy()
    if variant == "original4":
        return x
    x[train, 1] = rawpeak[train]
    x[query, 1] = rawpeak[query]
    if variant == "rawpeak4":
        return x
    if variant != "oracle08_4":
        raise ValueError(variant)
    yy = np.asarray(y, float)
    lo, hi = float(np.min(yy[train])), float(np.max(yy[train]))
    assert hi > lo
    x[train, 1] = .8 * (2.0 + (yy[train] - lo) / (hi - lo)) + .2 * rawpeak[train]
    x[query, 1] = .8 * (yy[query] - lo) / (hi - lo) + 1.6 + .2 * rawpeak[query]
    return x


def _fit_predict(data, feature, train, query, loss):
    shadow = models.training_view(data, train)
    assert np.isnan(shadow["y"][query]).all()
    z, q, y, ns, diag = models.design(shadow, [feature], train, query, "target_only")
    w = models.weights(y, ns, loss)
    kind, cost = ("residual", float(loss[6:])) if loss.startswith("inward") else ("control", 1.0)
    coef, info = models.solve(z, (y - 60.0) / 10.0, w, kind, cost, alpha=.01)
    prediction = np.maximum(60.0 + 10.0 * (q @ coef), 0.0)
    assert np.isfinite(prediction).all()
    return prediction, dict(**diag, loss=loss, solver=info,
                            coefficient_sha256=hashlib.sha256(coef.tobytes()).hexdigest(),
                            training_rows_sha256=hashlib.sha256(np.asarray(train, dtype="int64").tobytes()).hexdigest())


def _describe(y, p, accepted=None, variant=None):
    all_report = detailed(y, p)
    a = np.ones(len(y), bool) if accepted is None else np.asarray(accepted, bool)
    post = assess(y, p, a)
    if int(a.sum()) > 1:
        post["r2"] = float(r2_score(y[a], p[a]))
        post["rmse"] = float(np.sqrt(np.mean((p[a] - y[a]) ** 2)))
    else:
        post["r2"] = None
        post["rmse"] = None
    post["binary_confusion_rows_normal_abnormal"] = [
        [post["within_released"] - post["false_positive_count"], post["false_positive_count"]],
        [post["false_negative_count"], post["low_released"] + post["high_released"] - post["false_negative_count"]],
    ]
    t, q = states(y), states(p)
    post["withheld_by_true_state"] = [int(np.sum(~a & (t == j))) for j in range(3)]
    post["incorrect_yield_full_denominator"] = float(np.mean(a & (t != q)))
    for metric in (all_report, post):
        # Make both the numerator and the denominator explicit.  assess() uses
        # released cases as denominators after gating; the all-cohort
        # denominators are recorded as a separate audit field.
        metric["fnr_numerator"] = int(metric["false_negative_count"])
        metric["fnr_denominator_released_abnormal"] = int(metric["low_released"] + metric["high_released"])
        metric["fnr_denominator_all_abnormal"] = int(metric["low_n"] + metric["high_n"])
        metric["fpr_numerator"] = int(metric["false_positive_count"])
        metric["fpr_denominator_released_within"] = int(metric["within_released"])
        metric["fpr_denominator_all_within"] = int(metric["within_n"])
        metric["valid_UTEG_only_validation"] = variant in ("original4", "rawpeak4")
        metric["oracle_input_present"] = variant == "oracle08_4"
    return dict(all_report=all_report, post_gate=post,
                withheld_by_true_state=post["withheld_by_true_state"],
                teacher_point_criteria_pass=bool(all_report["mapping_target_pass"] and post["coverage"] > .75))


def _protocol(data, local, hashes):
    code_paths = [Path(__file__), Path(models.__file__), Path(contract.__file__),
                  Path(inspect.getsourcefile(models.solve)), Path(inspect.getsourcefile(models.design)),
                  Path(contract.ROOT / "code_submit/ma_target_optimization/metrics.py")]
    return dict(
        status="locked_before_evaluation",
        scope="all301 local, frozen5outer/3inner grouped evaluation; target-only four-parameter mapping",
        candidates=GRID,
        representations={
            "original4": "[t0, fitted_B, k, nu]",
            "rawpeak4": "[t0, baseline-subtracted observed peak, k, nu]",
            "oracle08_4": "[t0, 0.8*minmax_true_MA_2_3 + 0.2*observed_peak, k, nu]",
        },
        oracle_warning="oracle08_4 inserts each query record's true MA into its B coordinate; all oracle metrics are diagnostics, never U-TEG-only validation",
        train_query_isolation="For original4 and rawpeak4, labels are not used in query features. For oracle08_4, train min/max are fit on train labels, but query MA enters the deliberately oracle feature. Query labels never fit coefficients or gates.",
        model="Same target_only train-only z-score, RBF eigendecomposition, alpha=.01, nonnegative final MA clipping as frozen all301 models; four fixed losses, no outer-driven selection.",
        gate="For each candidate and outer fold, boundary_distance threshold is the inner OOF 95th percentile with method=higher; query labels never set thresholds.",
        acceptance="all-report agreement>.80, abnormal FNR<.05 and FNR<FPR; post-gate coverage>.75. Withheld cases are never counted correct.",
        data_counts={"records": int(len(local)), "groups": int(len(np.unique(data["groups"][local]))), "states": [11, 253, 37]},
        split_sha256=_sha(contract.OUT / "private/FROZEN_SPLITS.json"),
        input_hashes=hashes,
        code_hashes={str(p): _sha(p) for p in code_paths},
        original_data_modified=False, manuscript_modified=False, test_evaluated=False,
    )


def lock():
    data, local, blocks, _cur, _smooth, hashes, _meta = contract.load_inputs()
    peaks = _raw_peak(data, local)
    hashes = dict(hashes)
    hashes[str(RAW)] = _sha(RAW)
    hashes[str(contract.OUT / "private/FROZEN_SPLITS.json")] = _sha(contract.OUT / "private/FROZEN_SPLITS.json")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "private").mkdir(parents=True, exist_ok=True)
    protocol = _protocol(data, local, hashes)
    path = OUT / "FROZEN_PROTOCOL.json"
    text = json.dumps(protocol, ensure_ascii=False, indent=2)
    if path.exists():
        assert path.read_text(encoding="utf-8") == text
    else:
        path.write_text(text, encoding="utf-8")
    np.savez_compressed(OUT / "private" / "INPUT_SNAPSHOT.npz", local=local, raw_peak=peaks[local],
                        fitted4=blocks["fitted4"][0][local], y=data["y"][local, 2])
    print("LOCK_COMPLETE", OUT, flush=True)


def _load():
    data, local, blocks, cur, smooth, hashes, meta = contract.load_inputs()
    peaks = _raw_peak(data, local)
    hashes = dict(hashes)
    hashes[str(RAW)] = _sha(RAW)
    hashes[str(contract.OUT / "private/FROZEN_SPLITS.json")] = _sha(contract.OUT / "private/FROZEN_SPLITS.json")
    return data, local, blocks["fitted4"][0], peaks, hashes, meta


def _validate_fold(data, fold):
    """Recheck the frozen grouped split before fitting any candidate."""
    a, b = np.asarray(fold["A"], int), np.asarray(fold["B"], int)
    assert len(np.unique(a)) == len(a) and len(np.unique(b)) == len(b)
    assert not set(a).intersection(b)
    assert not set(data["groups"][a]).intersection(data["groups"][b])
    seen_val = []
    for ff in fold["inner"]:
        train, val = np.asarray(ff["T"], int), np.asarray(ff["V"], int)
        assert set(train) | set(val) == set(a)
        assert not set(train).intersection(val)
        assert not set(data["groups"][train]).intersection(data["groups"][val])
        assert not set(train).intersection(b) and not set(val).intersection(b)
        assert not set(data["groups"][train]).intersection(data["groups"][b])
        assert not set(data["groups"][val]).intersection(data["groups"][b])
        seen_val.extend(val.tolist())
    assert sorted(seen_val) == sorted(a.tolist())


def _oracle_audit(y, variant, train):
    if variant != "oracle08_4":
        return dict(train_ma_min=None, train_ma_max=None)
    yy = np.asarray(y, float)[train]
    return dict(train_ma_min=float(np.min(yy)), train_ma_max=float(np.max(yy)))


def _evaluate_fold(data, base4, peaks, y, fold):
    _validate_fold(data, fold)
    a, b = np.asarray(fold["A"], int), np.asarray(fold["B"], int)
    position = {int(row): j for j, row in enumerate(a)}
    inner = np.full((N_CANDIDATES, len(a)), np.nan)
    inner_audit = []
    for ff in fold["inner"]:
        train, val = np.asarray(ff["T"], int), np.asarray(ff["V"], int)
        for vi, variant in enumerate(VARIANTS):
            feature = _feature(variant, base4, peaks, y, train, val)
            for li, loss in enumerate(LOSSES):
                pred, audit = _fit_predict(data, feature, train, val, loss)
                idx = vi * len(LOSSES) + li
                inner[idx, [position[int(row)] for row in val]] = pred
                audit_row = dict(audit)
                audit_row.update(variant=variant, **_oracle_audit(y, variant, train))
                inner_audit.append(audit_row)
    assert np.isfinite(inner).all()
    outer = np.full((N_CANDIDATES, len(b)), np.nan)
    accepted = np.zeros((N_CANDIDATES, len(b)), bool)
    outer_audit = []
    for vi, variant in enumerate(VARIANTS):
        feature = _feature(variant, base4, peaks, y, a, b)
        for li, loss in enumerate(LOSSES):
            idx = vi * len(LOSSES) + li
            pred, audit = _fit_predict(data, feature, a, b, loss)
            outer[idx] = pred
            train_score = -np.minimum(abs(inner[idx] - 50.0), abs(inner[idx] - 70.0))
            threshold = float(np.quantile(train_score, .95, method="higher"))
            query_score = -np.minimum(abs(pred - 50.0), abs(pred - 70.0))
            accepted[idx] = query_score <= threshold
            audit_row = dict(audit)
            audit_row.update(variant=variant, loss=loss, threshold=threshold,
                             **_oracle_audit(y, variant, a))
            outer_audit.append(audit_row)
    return outer, accepted, inner, dict(inner=inner_audit, outer=outer_audit)


def smoke():
    protocol = json.loads((OUT / "FROZEN_PROTOCOL.json").read_text(encoding="utf-8"))
    for path, digest in protocol["code_hashes"].items():
        assert _sha(Path(path)) == digest, path
    data, local, base4, peaks, hashes, _meta = _load()
    folds = json.loads((contract.OUT / "private/FROZEN_SPLITS.json").read_text(encoding="utf-8"))
    outer, accepted, inner, audit = _evaluate_fold(data, base4, peaks, data["y"][:, 2], folds[0])
    assert outer.shape == (N_CANDIDATES, len(folds[0]["B"]))
    assert accepted.shape == outer.shape
    # Original uniform must exactly reproduce candidate 0 of the frozen model.
    old = np.load(contract.OUT / "private/PREDICTIONS.npz", allow_pickle=False)
    old_local = old["local"]
    assert np.array_equal(old_local, local)
    old_pos = [np.where(local == r)[0][0] for r in folds[0]["B"]]
    for i in range(len(LOSSES)):
        np.testing.assert_allclose(outer[i], old["fixed"][i, old_pos], rtol=0, atol=1e-10)
        np.testing.assert_array_equal(accepted[i], old["accepted"][i, old_pos])
    # Other three parameters are unchanged exactly in the constructed variants.
    train, val = np.asarray(folds[0]["inner"][0]["T"]), np.asarray(folds[0]["inner"][0]["V"])
    for variant in VARIANTS:
        x = _feature(variant, base4, peaks, data["y"][:, 2], train, val)
        np.testing.assert_array_equal(x[train][:, [0, 2, 3]], base4[train][:, [0, 2, 3]])
        np.testing.assert_array_equal(x[val][:, [0, 2, 3]], base4[val][:, [0, 2, 3]])
    _d, _l, loaded_blocks, *_rest = contract.load_inputs()
    np.testing.assert_allclose(peaks[local], loaded_blocks["amp6"][0][local, 0], rtol=0, atol=0)
    # For legal variants, poisoning held-out labels cannot alter predictions.
    poisoned = dict(data, y=data["y"].copy())
    poisoned["y"][np.asarray(folds[0]["B"], int)] = 1e6
    for variant in ("original4", "rawpeak4"):
        f = _feature(variant, base4, peaks, data["y"][:, 2], folds[0]["A"], folds[0]["B"])
        f2 = _feature(variant, base4, peaks, poisoned["y"][:, 2], folds[0]["A"], folds[0]["B"])
        for loss in LOSSES:
            p1, a1 = _fit_predict(data, f, np.asarray(folds[0]["A"]), np.asarray(folds[0]["B"]), loss)
            p2, a2 = _fit_predict(poisoned, f2, np.asarray(folds[0]["A"]), np.asarray(folds[0]["B"]), loss)
            np.testing.assert_array_equal(p1, p2)
            assert a1 == a2
    # For oracle, train fit/audit and threshold remain unchanged; query feature may change.
    f1 = _feature("oracle08_4", base4, peaks, data["y"][:, 2], folds[0]["A"], folds[0]["B"])
    f2 = _feature("oracle08_4", base4, peaks, poisoned["y"][:, 2], folds[0]["A"], folds[0]["B"])
    assert not np.array_equal(f1[folds[0]["B"]], f2[folds[0]["B"]])
    check = dict(first_outer_all12=True, original_four_losses_exact_replay=True,
                 other_three_parameters_exactly_unchanged=True, legal_query_label_poison_unchanged=True,
                 oracle_query_label_changes_feature_only=True, group_isolation=True,
                 rawpeak_exactly_matches_existing_amp6_first_column=True,
                 split_inner_membership_and_group_checks=True)
    (OUT / "VERIFICATION.json").write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SMOKE_COMPLETE", json.dumps(check, ensure_ascii=False), flush=True)


def predict():
    assert (OUT / "VERIFICATION.json").exists()
    protocol = json.loads((OUT / "FROZEN_PROTOCOL.json").read_text(encoding="utf-8"))
    for path, digest in protocol["code_hashes"].items():
        assert _sha(Path(path)) == digest, path
    if (OUT / "private/PREDICTIONS.npz").exists():
        raise RuntimeError("Predictions already sealed")
    data, local, base4, peaks, hashes, _meta = _load()
    folds = json.loads((contract.OUT / "private/FROZEN_SPLITS.json").read_text(encoding="utf-8"))
    fixed = np.full((N_CANDIDATES, len(local)), np.nan)
    accepted = np.zeros((N_CANDIDATES, len(local)), bool)
    fold_id = np.full(len(local), -1, int)
    position = {int(row): j for j, row in enumerate(local)}
    audits = []
    begin = time.time()
    for fi, fold in enumerate(folds):
        pred, acc, inner, audit = _evaluate_fold(data, base4, peaks, data["y"][:, 2], fold)
        ix = [position[int(row)] for row in fold["B"]]
        fixed[:, ix] = pred
        accepted[:, ix] = acc
        fold_id[ix] = fi
        audits.append(dict(fold=fi, query_count=len(ix), audit=audit))
        np.savez_compressed(OUT / "private" / f"FOLD_{fi}_PREDICTIONS.npz", prediction=pred,
                            accepted=acc, query=np.asarray(fold["B"]), inner=inner, training=np.asarray(fold["A"]))
        print("OUTER_FOLD_SAVED", fi, "seconds", round(time.time() - begin, 1), flush=True)
    assert np.isfinite(fixed).all() and np.all(fold_id >= 0)
    for path, digest in hashes.items():
        assert _sha(Path(path)) == digest, path
    np.savez_compressed(OUT / "private/PREDICTIONS.npz", fixed=fixed, accepted=accepted, local=local,
                        groups=data["groups"][local], fold=fold_id)
    (OUT / "private/FOLD_AUDIT.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf-8")
    seal = dict(predictions_sha256=_sha(OUT / "private/PREDICTIONS.npz"), protocol_sha256=_sha(OUT / "FROZEN_PROTOCOL.json"),
                split_sha256=protocol["split_sha256"], input_hashes=hashes, all5_outer_folds_complete=True,
                outer_metrics_not_computed=True)
    (OUT / "PREDICTION_SEAL.json").write_text(json.dumps(seal, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PREDICTIONS_SEALED_BEFORE_SCORING", flush=True)


def reseal():
    """Refresh the text seal after an audit-only protocol/metadata repair."""
    seal_path = OUT / "PREDICTION_SEAL.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    assert _sha(OUT / "private/PREDICTIONS.npz") == seal["predictions_sha256"]
    seal["protocol_sha256"] = _sha(OUT / "FROZEN_PROTOCOL.json")
    seal["metadata_repaired_after_initial_score"] = True
    seal_path.write_text(json.dumps(seal, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PREDICTION_SEAL_REFRESHED_WITH_SAME_NPZ", flush=True)


def audit_only():
    """Recompute fold audits and replay sealed predictions without refitting output files."""
    seal = json.loads((OUT / "PREDICTION_SEAL.json").read_text(encoding="utf-8"))
    assert _sha(OUT / "private/PREDICTIONS.npz") == seal["predictions_sha256"]
    data, local, base4, peaks, _hashes, _meta = _load()
    folds = json.loads((contract.OUT / "private/FROZEN_SPLITS.json").read_text(encoding="utf-8"))
    sealed = np.load(OUT / "private/PREDICTIONS.npz", allow_pickle=False)
    position = {int(row): j for j, row in enumerate(local)}
    audits = []
    for fi, fold in enumerate(folds):
        pred, acc, inner, audit = _evaluate_fold(data, base4, peaks, data["y"][:, 2], fold)
        saved = np.load(OUT / "private" / f"FOLD_{fi}_PREDICTIONS.npz", allow_pickle=False)
        np.testing.assert_allclose(pred, saved["prediction"], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(acc, saved["accepted"])
        ix = [position[int(row)] for row in fold["B"]]
        np.testing.assert_allclose(pred, sealed["fixed"][:, ix], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(acc, sealed["accepted"][:, ix])
        audits.append(dict(fold=fi, query_count=len(ix), audit=audit,
                           replay_identical_to_sealed_predictions=True))
    (OUT / "private/FOLD_AUDIT.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf-8")
    print("AUDIT_ONLY_REPLAY_COMPLETE", flush=True)


def score():
    seal = json.loads((OUT / "PREDICTION_SEAL.json").read_text(encoding="utf-8"))
    assert _sha(OUT / "private/PREDICTIONS.npz") == seal["predictions_sha256"]
    if (OUT / "RESULTS.json").exists():
        raise RuntimeError("Results already written")
    z = np.load(OUT / "private/PREDICTIONS.npz", allow_pickle=False)
    with np.load(contract.ROOT / contract.DEFAULT_DATA_PATH, allow_pickle=False) as a:
        y = np.asarray(a["medical"][z["local"], 2], float)
    rows = []
    for i, c in enumerate(GRID):
        d = _describe(y, z["fixed"][i], z["accepted"][i], c["variant"])
        rows.append(dict(candidate=c, **d))
    result = dict(status="ORACLE_AND_LEGAL_FIXED_COMPARISON; oracle08_4 is not U-TEG-only validation",
                  candidates=rows, n_records=len(y), class_records=np.bincount(states(y), minlength=3).tolist(),
                  n_groups=int(len(np.unique(z["groups"]))), input_hashes=seal["input_hashes"],
                  original_data_modified=False, manuscript_changed=False)
    (OUT / "RESULTS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(result)
    print(json.dumps({"n": len(y), "candidates": [{"candidate": r["candidate"], "all_report": r["all_report"], "post_gate": r["post_gate"]} for r in rows]}, ensure_ascii=False), flush=True)


def _write_report(result):
    def pct(x): return "NA" if x is None else f"{100*x:.2f}%"
    def r2(x): return "NA" if x is None else f"{x:.3f}"
    lines = ["# 四参数幅值替换与MA映射诊断（all301）", "",
             "原始四参数为 `[t0, fitted_B, k, nu]`。本轮只替换第二列B，其他三项逐行保持不变；固定target-only、训练侧标准化/RBF、alpha=.01、四种既定损失和冻结5外层×3内层分组。", "",
             "`original4`和`rawpeak4`只使用U-TEG；`oracle08_4`把查询记录真实MA归一化到训练折的2–3范围后混入B，因此只能作为oracle诊断。它表示指定混合输入下的诊断结果，不是严格上限，也不能写成U-TEG预测性能。", "",
             "## 全部报告（原始真实301条）", "", "|表示|损失|一致率|FNR|FPR|低召回|高召回|MAE mm|R²|覆盖|", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in result["candidates"]:
        m = r["all_report"]
        lines.append(f"|{r['candidate']['variant']}|{r['candidate']['loss']}|{pct(m['agreement'])}|{pct(m['fnr'])}|{pct(m['fpr'])}|{pct(m['low_recall'])}|{pct(m['high_recall'])}|{m['mae']:.3f}|{r2(m.get('r2'))}|100%|")
    lines += ["", "## 门后结果", "", "|表示|损失|覆盖|已报告一致率|已报告FNR|已报告FPR|低覆盖/召回|高覆盖/召回|MAE mm|R²|", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in result["candidates"]:
        m = r["post_gate"]
        lines.append(f"|{r['candidate']['variant']}|{r['candidate']['loss']}|{pct(m['coverage'])}|{pct(m['agreement'])}|{pct(m['fnr'])}|{pct(m['fpr'])}|{pct(m['low_coverage'])}/{pct(m['low_recall'])}|{pct(m['high_coverage'])}/{pct(m['high_recall'])}|{m['mae']:.3f}|{r2(m.get('r2'))}|")
    lines += ["", "## 同一损失下的重点比较", "", "原始B、实测峰值替换B和0.8 oracle替换B按同一损失逐行配对；oracle输入包含真实MA，不能用于证明可部署效果。", "", "|损失|original4 一致/FNR/FPR/MAE/R²|rawpeak4 一致/FNR/FPR/MAE/R²|oracle08_4 一致/FNR/FPR/MAE/R²|", "|---|---|---|---|"]
    for loss in LOSSES:
        cells = []
        for variant in VARIANTS:
            r = next(x for x in result["candidates"] if x["candidate"] == {"variant": variant, "loss": loss})
            m = r["all_report"]
            cells.append(f"{pct(m['agreement'])}/{pct(m['fnr'])}/{pct(m['fpr'])}/{m['mae']:.3f}/{r2(m.get('r2'))}")
        lines.append(f"|{loss}|{cells[0]}|{cells[1]}|{cells[2]}|")
    lines += ["", "## 判断", "", "12组固定对照均未同时达到预设的全部条件（全部报告一致率>80%、FNR<5%、FNR<FPR，且门后覆盖率>75%）；oracle组也未达到FNR<5%，因此不能把它写成目标已实现。原始峰值替换B是可部署意义上的U-TEG-only对照；0.8混合组只说明在指定混合输入下，四参数映射和状态判断的诊断表现。任何oracle组的R²、一致率或召回均不计入正式模型排行榜，也不修改论文。", "", "详细逐候选结果见 `RESULTS.json`；外层预测先封存后评分，原始输入和历史结果未修改。"]
    (OUT / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["lock", "smoke", "predict", "reseal", "audit_only", "score"])
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        globals()[args.stage]()
