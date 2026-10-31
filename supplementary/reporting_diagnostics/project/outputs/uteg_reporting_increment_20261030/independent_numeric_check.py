from __future__ import annotations

"""Independent, read-only numerical audit for the 2026-10-30 increment.

This file intentionally does not import or execute ``increment_experiment.py``.
It reconstructs the frozen target-side score/action interface locally, treats
the source action archives as frozen inputs, and writes only the two audit
reports in this directory.  It is an audit, not a re-fit or a reproduction of
the development optimisers.
"""

import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.special import expit


ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[1]
OUT = BASE / "outputs"
DATA = OUT / "uteg_legacy_aligned_threecenter_20261022_155614" / "DATA15.npz"
FOCUSED = OUT / "uteg_nc_goal_20261030" / "focused_reporting_20261030"
OLD = OUT / "uteg_signal_model_goal_20261026_000929" / "ridge10_fullchain_20261026_122551"
SOURCE_ACTIONS = OUT / "uteg_ridge_dual_head_external_eval_20261026_191117" / "actions"
DIRS = ["HZ_to_SA", "HZ+HB_to_SA"]
SEEDS = list(range(1000, 1020))
CAPS = [0.0, 0.05, 0.10, 0.15, 0.20]
N_BOOT = 5000
PROTOCOL = "uteg-increment-20261030"
S14_EXPECTED = {
    "Target-only Ridge|S14": {"released": 449, "correct": 426, "incorrect": 23},
    "HZ_to_SA|S14": {"released": 536, "correct": 485, "incorrect": 51},
    "HZ+HB_to_SA|S14": {"released": 664, "correct": 620, "incorrect": 44},
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_report(report: dict[str, Any]) -> None:
    # The checker is intentionally restricted to these two writes.
    out_json = ROOT / "INDEPENDENT_NUMERIC_CHECK.json"
    out_md = ROOT / "INDEPENDENT_NUMERIC_CHECK.md"
    out_json.write_text(json.dumps(jsonable(report), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = [
        "# 独立数值验收",
        "",
        f"- 协议：`{PROTOCOL}`",
        f"- 状态：**{report.get('status', 'UNKNOWN')}**",
        "- 范围：只读重算；未训练、未拟合、未修改论文或原始实验目录。",
        "",
        "## 结论摘要",
    ]
    for item in report.get("issues", []):
        lines.append(f"- 问题：{item}")
    if not report.get("issues"):
        lines.append("- 未记录影响本轮数值验收的失败项。")
    if report.get("missing_artifacts"):
        lines.append("- 等待文件：" + ", ".join(report["missing_artifacts"]))
    lines += ["", "## 检查项"]
    for item in report.get("checks", []):
        mark = {"PASS": "通过", "FAIL": "失败", "WAITING": "等待", "INFO": "信息", "NA": "不适用"}.get(item.get("status"), item.get("status"))
        detail = item.get("detail", "")
        lines.append(f"- `{item.get('name')}`：{mark}；{detail}")
    if report.get("e0"):
        lines += ["", "## E0/S14 独立重算"]
        for name, row in report["e0"].items():
            lines.append(f"- {name}：released={row.get('released')}，correct={row.get('correct')}，incorrect={row.get('incorrect')}。")
    if report.get("qc_diagnostics"):
        lines += ["", "## E1 QC 诊断"]
        for name, row in report["qc_diagnostics"].items():
            lines.append(f"- {name}：test QC-false records={row.get('qc_false_records')}；unmasked releases among QC-false={row.get('unmasked_release_events')}；stored releases among QC-false={row.get('stored_release_events_qc_false')}；QC-masked/local action difference={row.get('qc_mask_difference_events')}。")
    if report.get("primary_bootstrap"):
        lines += ["", "## 预设主/次比较（source − target-only）"]
        for key, row in report["primary_bootstrap"].items():
            lines.append(f"- {key}：count delta={row.get('count_delta')}；correct-yield delta={row.get('correct_yield_delta')}（group CI={row.get('correct_group_ci')}；record CI={row.get('correct_record_ci')}）；incorrect-yield delta={row.get('incorrect_yield_delta')}（group CI={row.get('incorrect_group_ci')}；record CI={row.get('incorrect_record_ci')}）。")
    if report.get("boundary"):
        lines += ["", "## 核验边界"]
        for line in report["boundary"]:
            lines.append(f"- {line}")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_check(report: dict[str, Any], name: str, status: str, detail: str, **extra: Any) -> None:
    report.setdefault("checks", []).append({"name": name, "status": status, "detail": detail, **extra})
    if status == "FAIL":
        report.setdefault("issues", []).append(f"{name}: {detail}")


def load_data() -> dict[str, np.ndarray]:
    with np.load(DATA, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def ma_class(value: Any) -> np.ndarray:
    x = np.asarray(value, dtype=float)
    return np.where(~np.isfinite(x), 9, np.where(x < 50, -1, np.where(x <= 70, 0, 1))).astype(np.int8)


def project(y: np.ndarray) -> np.ndarray:
    out = np.asarray(y, dtype=float).copy()
    out[..., :3] = np.maximum(out[..., :3], 0.0)
    out[..., 3] = np.clip(out[..., 3], 0.0, 89.999999)
    return out


def predict(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    z = (np.asarray(x, dtype=float) - np.asarray(model["xm"])) / np.asarray(model["xs"])
    return project(np.asarray(model["ym"]) + np.asarray(model["ys"]) * (z @ np.asarray(model["coef"])))


def ma_features(x: np.ndarray, med: np.ndarray, ym: Any, ys: Any) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.asarray(med, dtype=float)
    pred_ma = med[:, 2]
    pc = ma_class(pred_ma)
    return np.column_stack((x, (med - np.asarray(ym)[None, :]) / np.asarray(ys)[None, :],
                            (pred_ma - 60.0) / 10.0,
                            np.minimum(np.abs(pred_ma - 50.0), np.abs(pred_ma - 70.0)) / 10.0,
                            (pc == -1).astype(float), (pc == 0).astype(float), (pc == 1).astype(float)))


def score_gate(model: dict[str, Any], features: np.ndarray, pred: np.ndarray) -> np.ndarray:
    pre = model["pre"]
    f = np.asarray(features, dtype=float)
    finite = np.isfinite(f)
    filled = np.where(finite, f, np.asarray(pre["median"])[None, :])
    z = np.clip((filled - np.asarray(pre["mean"])[None, :]) / np.asarray(pre["sd"])[None, :], -10.0, 10.0)
    if bool(model.get("fallback")):
        score = np.full(len(f), float(model["constant_score"]))
    else:
        score = expit(z @ np.asarray(model["coef"]) + float(model["intercept"]))
    valid = np.isin(np.asarray(pred), [-1, 0, 1]) & np.isfinite(f).all(axis=1)
    return np.where(valid, score, np.inf)


def head_scores(heads: dict[str, Any], x: np.ndarray, med: np.ndarray) -> np.ndarray:
    values = []
    for key in ("low", "high"):
        model = heads.get(key)
        if model is None:
            values.append(np.full(len(x), np.inf))
            continue
        pre = model["pre"]
        f = np.column_stack([np.asarray(x)[:, 1], np.asarray(med)[:, 2]])
        z = np.clip((np.where(np.isfinite(f), f, np.asarray(pre["median"])[None, :]) - np.asarray(pre["mean"])[None, :]) / np.asarray(pre["sd"])[None, :], -10.0, 10.0)
        theta = np.asarray(model["theta"])
        values.append(expit(z @ theta[:-1] + theta[-1]))
    return np.maximum(*values)


def s14_action(th: dict[str, Any], pred: np.ndarray, base: np.ndarray, dual: np.ndarray, qc: np.ndarray) -> np.ndarray:
    out = np.zeros(len(pred), dtype=bool)
    if bool(th.get("all_withhold")):
        return out
    if th.get("normal") is not None:
        out |= (pred == 0) & np.isfinite(dual) & (dual <= float(th["normal"]))
    if th.get("tail") is not None:
        out |= np.isin(pred, [-1, 1]) & np.isfinite(base) & (base <= float(th["tail"]))
    return out & np.asarray(qc, dtype=bool)


def policy_action(pred: np.ndarray, base: np.ndarray, dual: np.ndarray, th: dict[str, Any], qc: np.ndarray | None = None) -> np.ndarray:
    """The E1 policy; QC is explicit because the plan requires fail-closed release."""
    out = np.zeros(len(pred), dtype=bool)
    if not bool(th.get("all_withhold", False)):
        if th.get("normal") is not None:
            out |= (pred == 0) & np.isfinite(dual) & (dual <= float(th["normal"]))
        if th.get("tail") is not None:
            out |= np.isin(pred, [-1, 1]) & np.isfinite(base) & (base <= float(th["tail"]))
    return out if qc is None else (out & np.asarray(qc, dtype=bool))


def safe_name(name: str) -> str:
    return name.replace("|", "__").replace("+", "plus").replace("=", "_").replace(".", "p")


def stats(pred: np.ndarray, action: np.ndarray, truth: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    if mask is None:
        mask = np.ones(len(truth), dtype=bool)
    p = np.asarray(pred)[:, mask]
    a = np.asarray(action, dtype=bool)[:, mask]
    t = np.asarray(truth)[mask][None, :]
    correct = a & (p == t)
    incorrect = a & (p != t)
    released = int(a.sum())
    return {
        "records": int(mask.sum()),
        "record_seed_n": int(a.size),
        "released": released,
        "correct": int(correct.sum()),
        "incorrect": int(incorrect.sum()),
        "unreported": int((~a).sum()),
        "coverage": float(a.mean()),
        "agreement": float(correct.sum() / released) if released else None,
        "zero_report_seeds": int(np.sum(a.sum(axis=1) == 0)),
    }


def threshold(freeze: dict[str, Any], method: str, seed: int, cap: float) -> dict[str, Any]:
    for row in freeze.get("thresholds", []):
        if row.get("method") == method and int(row.get("seed")) == int(seed) and abs(float(row.get("cap")) - cap) < 1e-12:
            return row
    raise KeyError((method, seed, cap))


def compare(a: np.ndarray, b: np.ndarray) -> bool:
    if a.dtype.kind in "biuSU" or b.dtype.kind in "biuSU":
        return np.array_equal(a, b)
    return np.allclose(a, b, atol=1e-12, rtol=1e-12, equal_nan=True)


def load_recomputed(D: dict[str, np.ndarray], report: dict[str, Any]) -> dict[str, Any]:
    """Recompute target scores and old/new actions without calling experiment code."""
    m0 = joblib.load(FOCUSED / "model_1000.joblib")
    idx = np.asarray(m0["test"], dtype=int)
    target = {"pred": [], "base": [], "dual": [], "qc": np.tile(D["input_qc"][idx][None, :], (20, 1)), "s14": []}
    for seed in SEEDS:
        m = joblib.load(FOCUSED / f"model_{seed}.joblib")
        if not np.array_equal(np.asarray(m["test"], dtype=int), idx):
            raise AssertionError(f"focused model test index mismatch seed {seed}")
        med = predict(m["predictor"], D["x"][idx])
        pred = ma_class(med[:, 2])
        base = score_gate(m["error"], ma_features(D["x"][idx], med, m["ym"], m["ys"]), pred)
        dual = head_scores(m["heads"], D["x"][idx], med)
        target["pred"].append(pred)
        target["base"].append(base)
        target["dual"].append(dual)
        target["s14"].append(s14_action(m["threshold"], pred, base, dual, D["input_qc"][idx]))
    for key in ("pred", "base", "dual", "s14"):
        target[key] = np.asarray(target[key])

    sources = {}
    old_pred = {}
    for direction in DIRS:
        with np.load(SOURCE_ACTIONS / f"{direction}.npz", allow_pickle=False) as z:
            src = {k: z[k].copy() for k in z.files}
        with np.load(OLD / "eval" / "predictions" / f"{direction}.npz", allow_pickle=False) as z:
            old = {k: z[k].copy() for k in z.files}
        old_pred[direction] = old
        sources[direction] = src
        expected_ids = D["ids"][idx].astype(str)
        if not np.array_equal(src["test_ids"].astype(str), expected_ids):
            raise AssertionError(f"source test_ids mismatch {direction}")
        if not np.array_equal(src["input_qc"].astype(bool), D["input_qc"][idx].astype(bool)):
            raise AssertionError(f"source QC mismatch {direction}")
        if not np.array_equal(old["seed"].astype(int), np.asarray(SEEDS)):
            raise AssertionError(f"old source seed order mismatch {direction}")
        if not np.array_equal(old["test_global_index"].astype(int), idx):
            raise AssertionError(f"old source test index mismatch {direction}")
        for key in ("test_ids", "input_qc", "final_ma", "final_ma_class", "lr_score"):
            if not compare(src[key], old[key]):
                raise AssertionError(f"source-vs-old mismatch {direction}:{key}")

    truth = ma_class(D["medical"][idx, 2]).astype(int)
    result = {"idx": idx, "truth": truth, "target": target, "sources": sources, "old_source": old_pred}
    report["test_geometry"] = {
        "test_records": int(len(idx)),
        "test_records_unique": int(len(np.unique(idx))),
        "test_proxy_groups": int(len(np.unique(D["groups"][idx].astype(str)))),
        "test_index_sha256": hashlib.sha256(idx.tobytes()).hexdigest(),
        "test_id_alignment_sha256": hashlib.sha256("\n".join(D["ids"][idx].astype(str)).encode()).hexdigest(),
    }
    return result


def check_source_roles(D: dict[str, np.ndarray], idx: np.ndarray, report: dict[str, Any]) -> None:
    target0 = joblib.load(FOCUSED / "model_1000.joblib")
    pool = np.asarray(target0["pool"], dtype=int)
    role_rows = []
    all_pass = True
    for direction in DIRS:
        for seed in SEEDS:
            path = OLD / "models" / direction / "target15" / f"SAHZU_CFULL_seed{seed}.joblib"
            obj = joblib.load(path)
            p = np.asarray(obj["pool"], dtype=int); t = np.asarray(obj["test"], dtype=int)
            same_pool = np.array_equal(p, pool)
            same_test = np.array_equal(t, idx)
            disjoint = len(set(D["groups"][p].astype(str)) & set(D["groups"][t].astype(str))) == 0
            all_pass &= same_pool and same_test and disjoint
            role_rows.append({"direction": direction, "seed": seed, "pool_n": int(len(p)), "test_n": int(len(t)), "same_as_target_pool": same_pool, "same_as_target_test": same_test, "pool_test_proxygroup_overlap": 0 if disjoint else -1})
    report["source_role_alignment"] = {"all_source_target_pool_test_equal": all_pass, "rows": role_rows, "note": "仅核对各 source direction 与 target 的 pool/test 对齐及 pool-test proxy-group 不重叠；未声称跨 source/target 全局患者角色独立。"}
    add_check(report, "source_pool_test_alignment", "PASS" if all_pass else "FAIL", f"40 个 direction×seed 角色文件均与 target pool/test 对齐且 pool-test proxy-group 不重叠={all_pass}。")


def check_manifest(report: dict[str, Any]) -> None:
    path = ROOT / "input_manifest.json"
    if not path.exists():
        add_check(report, "input_manifest", "WAITING", "input_manifest.json 不存在。")
        return
    obj = json.loads(path.read_text(encoding="utf-8"))
    mismatches, missing = [], []
    for pstr, expected in obj.get("inputs", {}).items():
        p = Path(pstr)
        if not p.exists():
            missing.append(p.name)
        elif sha(p) != expected:
            mismatches.append(p.name)
    ok = not missing and not mismatches
    add_check(report, "input_manifest_hashes", "PASS" if ok else "FAIL", f"entries={len(obj.get('inputs', {}))}；missing={len(missing)}；hash_mismatch={len(mismatches)}。", missing=missing, mismatches=mismatches)
    report["manifest"] = {"protocol": obj.get("protocol"), "entries": len(obj.get("inputs", {})), "missing": missing, "mismatches": mismatches, "test_indices_sha256": obj.get("test_indices_sha256")}


def check_subset(D: dict[str, np.ndarray], idx: np.ndarray, report: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    path = ROOT / "subset_mask.json"
    obj = json.loads(path.read_text(encoding="utf-8"))
    groups = D["groups"].astype(str)
    test_groups = sorted(set(groups[idx]))
    ranked = sorted(test_groups, key=lambda g: hashlib.sha256(f"{PROTOCOL}|{g}".encode()).hexdigest())
    selected = ranked[:math.ceil(0.30 * len(ranked))]
    selected_set = set(selected)
    expected = np.asarray([groups[i] in selected_set for i in idx], dtype=bool)
    selected_hashes = [hashlib.sha256(f"{PROTOCOL}|group|{g}".encode()).hexdigest()[:24] for g in selected]
    all_hashes = [hashlib.sha256(f"{PROTOCOL}|group|{g}".encode()).hexdigest()[:24] for g in ranked]
    selected_idx = [int(i) for i in idx[expected]]
    complement_idx = [int(i) for i in idx[~expected]]
    checks = [
        bool(obj.get("uses_labels_or_scores") is False),
        int(obj.get("test_n", -1)) == 201,
        int(obj.get("proxy_group_n", -1)) == 190,
        int(obj.get("selected_proxy_group_n", -1)) == 57,
        int(obj.get("selected_record_n", -1)) == int(expected.sum()) == 60,
        int(obj.get("complement_record_n", -1)) == int((~expected).sum()) == 141,
        obj.get("selected_group_hashes") == selected_hashes,
        obj.get("all_group_hashes_sorted") == all_hashes,
        sorted(map(int, obj.get("selected_global_indices", []))) == sorted(selected_idx),
        sorted(map(int, obj.get("complement_global_indices", []))) == sorted(complement_idx),
    ]
    record_hashes = obj.get("record_hash_by_global_index", {})
    record_ok = all(record_hashes.get(str(int(i))) == hashlib.sha256(f"{PROTOCOL}|record|{int(i)}".encode()).hexdigest()[:20] for i in idx)
    checks.append(record_ok)
    ok = all(checks)
    report["subset"] = {"selection_hash_rule_recomputed": True, "test_groups": len(test_groups), "selected_groups": len(selected), "selected_records": int(expected.sum()), "complement_records": int((~expected).sum()), "selected_global_index_sha256": hashlib.sha256(np.asarray(selected_idx, dtype=np.int64).tobytes()).hexdigest(), "checks_passed": int(sum(checks)), "checks_total": len(checks)}
    add_check(report, "subset_hash_mask", "PASS" if ok else "FAIL", f"重算57/190 proxy groups、60/141 records；labels/scores未参与={obj.get('uses_labels_or_scores') is False}；{sum(checks)}/{len(checks)}项一致。")
    return expected, ~expected


def check_development_freeze(D: dict[str, np.ndarray], idx: np.ndarray, report: dict[str, Any]) -> None:
    freeze_path = ROOT / "frozen_thresholds.json"
    dev_path = ROOT / "development_oof_private.json"
    checks = []
    details: dict[str, Any] = {}
    if freeze_path.exists():
        f = json.loads(freeze_path.read_text(encoding="utf-8"))
        checks += [f.get("protocol") == PROTOCOL, f.get("test_read") is False, f.get("caps") == CAPS, len(f.get("thresholds", [])) == 300]
        counts = {}
        for row in f.get("thresholds", []):
            counts[(row.get("method"), int(row.get("seed", -1)), float(row.get("cap", -1)))] = counts.get((row.get("method"), int(row.get("seed", -1)), float(row.get("cap", -1))), 0) + 1
        checks.append(len(counts) == 300 and all(v == 1 for v in counts.values()))
        details["threshold_count"] = len(f.get("thresholds", [])); details["test_read"] = f.get("test_read"); details["caps"] = f.get("caps")
    else:
        checks.append(False); details["missing"] = "frozen_thresholds.json"
    if dev_path.exists():
        d = json.loads(dev_path.read_text(encoding="utf-8")); rows = d.get("rows", [])
        checks += [d.get("test_read") is False, len(rows) == 3 * 20 * 60]
        test_set = set(map(int, idx.tolist()))
        per = {}
        outside_test = True
        for r in rows:
            key = (r.get("method"), int(r.get("seed", -1)))
            per[key] = per.get(key, 0) + 1
            if int(r.get("global_index", -1)) in test_set:
                outside_test = False
        checks += [len(per) == 60 and all(v == 60 for v in per.values()), outside_test]
        details["development_rows"] = len(rows); details["development_method_seed_cells"] = len(per); details["development_rows_touch_test"] = not outside_test
    else:
        checks.append(False); details["missing_dev_oof"] = "development_oof_private.json"
    freezep = ROOT / "FREEZE.json"
    if freezep.exists():
        fr = json.loads(freezep.read_text(encoding="utf-8")); checks += [fr.get("protocol") == PROTOCOL, fr.get("test_evaluated") is False, int(fr.get("threshold_count", -1)) == 300]
        script_hash = fr.get("script_sha256")
        current_hash = sha(ROOT / "increment_experiment.py") if (ROOT / "increment_experiment.py").exists() else None
        details["freeze_script_hash_matches_current"] = script_hash == current_hash
        # This is provenance information; a stale hash is a failure of the audit chain,
        # even though it does not by itself prove that the numerical arrays are wrong.
        checks.append(script_hash == current_hash)
    else:
        checks.append(False); details["missing_freeze"] = "FREEZE.json"
    ok = all(checks)
    report["freeze"] = details
    add_check(report, "development_freeze", "PASS" if ok else "FAIL", f"冻结caps/300 thresholds、OOF不触及test、FREEZE test_evaluated=false及脚本哈希链检查={sum(checks)}/{len(checks)}。")


def check_source_action_shape(result: dict[str, Any], D: dict[str, np.ndarray], report: dict[str, Any]) -> None:
    idx = result["idx"]
    rows = {}
    ok_all = True
    for d, src in result["sources"].items():
        qc = src["input_qc"].astype(bool)
        a = src["unified_action"].astype(bool)
        p = src["final_ma_class"].astype(int)
        dual = src["dual_score"].astype(float)
        valid = src["dual_valid"].astype(bool)
        base = src["lr_score"].astype(float)
        ok = [a.shape == (20, 201), np.all(~(a & ~qc[None, :])), np.all(~(a & (p == 0) & ~(valid & np.isfinite(dual)))), np.all(~(a & np.isin(p, [-1, 1]) & ~np.isfinite(base)))]
        ok_all &= all(ok)
        rows[d] = {"shape": list(a.shape), "qc_false_records": int((~qc).sum()), "release_events": int(a.sum()), "release_events_qc_false": int((a & ~qc[None, :]).sum()), "normal_release_invalid_dual_events": int((a & (p == 0) & ~(valid & np.isfinite(dual))).sum()), "tail_release_nonfinite_base_events": int((a & np.isin(p, [-1, 1]) & ~np.isfinite(base)).sum())}
    report["source_actions"] = rows
    add_check(report, "source_action_fail_closed", "PASS" if ok_all else "FAIL", f"source actions shape/QC/dual-valid/finite-score checks={ok_all}；未以NPZ缺少seed字段为通过依据。")


def build_arms(result: dict[str, Any], freeze: dict[str, Any], qc_masked: bool = True) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    target = result["target"]; truth = result["truth"]; qc = target["qc"].astype(bool)
    arms: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "Target-only Ridge|S14": (target["pred"].astype(int), target["s14"].astype(bool)),
    }
    for d in DIRS:
        src = result["sources"][d]
        arms[f"{d}|S14"] = (src["final_ma_class"].astype(int), src["unified_action"].astype(bool))
    for cap in CAPS:
        for method in ["target_only", *DIRS]:
            if method == "target_only":
                pred = target["pred"].astype(int); base = target["base"].astype(float); dual = target["dual"].astype(float)
            else:
                src = result["sources"][method]
                pred = src["final_ma_class"].astype(int); base = src["lr_score"].astype(float)
                dual = np.where(src["dual_valid"].astype(bool) & np.isfinite(src["dual_score"]), src["dual_score"].astype(float), np.nan)
            actions = []
            for j, seed in enumerate(SEEDS):
                th = threshold(freeze, method, seed, cap)
                actions.append(policy_action(pred[j], base[j], dual[j], th, qc[j] if qc_masked else None))
            arms[f"{method}|cap={cap:.2f}"] = (pred, np.asarray(actions))
    return arms


def check_private_and_tables(result: dict[str, Any], subset_mask: np.ndarray, complement_mask: np.ndarray, report: dict[str, Any]) -> dict[str, Any]:
    required = ["evaluation_private.npz", "arm_name_map.json", "results.csv", "per_seed.csv", "class_destinations.csv", "paired_contrasts.json", "validation.json"]
    missing = [p for p in required if not (ROOT / p).exists()]
    report["missing_artifacts"] = missing
    if missing:
        add_check(report, "execution_artifacts", "WAITING", "主分析输出仍缺少：" + ", ".join(missing))
        return {}
    freeze = json.loads((ROOT / "frozen_thresholds.json").read_text(encoding="utf-8"))
    expected_arms = build_arms(result, freeze, qc_masked=True)
    expected_unmasked = build_arms(result, freeze, qc_masked=False)
    private = np.load(ROOT / "evaluation_private.npz", allow_pickle=False)
    arm_map = json.loads((ROOT / "arm_name_map.json").read_text(encoding="utf-8"))
    private_ok = True; qc_diff = {}; private_stats: dict[str, Any] = {}
    for safe, display in arm_map.items():
        if display not in expected_arms:
            private_ok = False; continue
        pk = "pred__" + safe; ak = "action__" + safe
        if pk not in private or ak not in private:
            private_ok = False; continue
        exp_pred, exp_action = expected_arms[display]
        got_pred = private[pk].astype(int); got_action = private[ak].astype(bool)
        pred_ok = compare(got_pred, exp_pred)
        action_ok = np.array_equal(got_action, exp_action)
        private_ok &= pred_ok and action_ok
        if not action_ok:
            unmasked = expected_unmasked.get(display, (None, np.zeros_like(got_action)))[1]
            qc_diff[display] = {"stored_matches_qc_masked": False, "stored_matches_unmasked": bool(np.array_equal(got_action, unmasked)), "stored_minus_qc_masked_events": int(np.sum(got_action != exp_action)), "stored_release_events_qc_false": int((got_action & ~result["target"]["qc"].astype(bool)).sum())}
        private_stats[display] = {"prediction_match": pred_ok, "action_match_qc_masked": action_ok}
    expected_safe = {safe_name(k) for k in expected_arms}
    private_ok &= set(arm_map) == expected_safe and len(arm_map) == len(expected_arms)
    add_check(report, "private_array_reconstruction", "PASS" if private_ok else "FAIL", f"18 arms 的prediction/action与独立重算（E1按计划显式QC fail-closed）一致={private_ok}。")
    report["private_arrays"] = private_stats

    # Results tables are re-aggregated from the private arrays, including all
    # class destinations and all three prespecified masks.
    masks = {"full201": np.ones(201, dtype=bool), "subset57groups": subset_mask, "complement133groups": complement_mask}
    result_rows = list(csv.DictReader((ROOT / "results.csv").open(encoding="utf-8-sig", newline="")))
    dest_rows = list(csv.DictReader((ROOT / "class_destinations.csv").open(encoding="utf-8-sig", newline="")))
    seed_rows = list(csv.DictReader((ROOT / "per_seed.csv").open(encoding="utf-8-sig", newline="")))
    table_ok = True; row_checks = 0; row_total = 0
    for row in result_rows:
        display = row["arm"]; mask = masks.get(row["subset"])
        if display not in expected_arms or mask is None:
            table_ok = False; continue
        p, a = expected_arms[display]
        st = stats(p, a, result["truth"], mask)
        row_total += 1
        ok = all(int(row[k]) == int(st["correct" if k == "correct_yield" else "incorrect" if k == "incorrect_yield" else k]) for k in ("records", "record_seed_n", "correct_yield", "incorrect_yield", "unreported"))
        ok &= abs(float(row["coverage"]) - st["coverage"]) < 1e-12
        if st["agreement"] is None:
            ok &= row["agreement"] in ("", "NA", "nan", "None")
        else:
            ok &= abs(float(row["agreement"]) - st["agreement"]) < 1e-12
        table_ok &= ok; row_checks += int(ok)
    table_ok &= len(result_rows) == 18 * 3
    # Destination tables use the implementation's unconditional N = records*20
    # denominator; check both cell counts and all class/report partitions.
    dest_ok = True; dest_checks = 0; dest_total = 0
    for row in dest_rows:
        display = row["arm"]; mask = masks.get(row["subset"])
        if display not in expected_arms or mask is None:
            dest_ok = False; continue
        p, a = expected_arms[display]; t = result["truth"][mask][None, :]
        true_state = int(row["true_state"]); report_state = int(row["report_state"])
        reported = np.where(a[:, mask], p[:, mask], 9)
        expected_n = int(((reported == report_state) & (t == true_state)).sum())
        N = int(mask.sum() * 20)
        ok = int(row["n"]) == expected_n and int(row["N"]) == N and abs(float(row["proportion"]) - expected_n / N) < 1e-12
        dest_ok &= ok; dest_checks += int(ok); dest_total += 1
    dest_ok &= len(dest_rows) == 18 * 3 * 3 * 4
    # Per-seed rows retain every seed, including a zero-report seed.
    seed_ok = True; seed_checks = 0; seed_total = 0
    for row in seed_rows:
        display = row["arm"]; seed = int(row["seed"])
        if display not in expected_arms or seed not in SEEDS:
            seed_ok = False; continue
        j = SEEDS.index(seed); p, a = expected_arms[display]; st = stats(p[j:j+1], a[j:j+1], result["truth"])
        ok = all(int(row[k]) == int(st[k]) for k in ("released", "correct", "incorrect", "unreported"))
        ok &= row["agreement"] in ("", "NA", "nan", "None") if st["agreement"] is None else abs(float(row["agreement"]) - st["agreement"]) < 1e-12
        seed_ok &= ok; seed_checks += int(ok); seed_total += 1
    seed_ok &= len(seed_rows) == 18 * 20
    add_check(report, "results_count_identities", "PASS" if table_ok else "FAIL", f"results.csv行/计数/coverage/agreement检查={row_checks}/{row_total}且总行={len(result_rows)}。")
    add_check(report, "class_destination_partitions", "PASS" if dest_ok else "FAIL", f"class_destinations.csv单元格重算={dest_checks}/{dest_total}且总行={len(dest_rows)}。")
    add_check(report, "per_seed_zero_report_retention", "PASS" if seed_ok else "FAIL", f"per_seed.csv逐seed重算={seed_checks}/{seed_total}且总行={len(seed_rows)}。")
    report["table_counts"] = {"results_rows": len(result_rows), "destinations_rows": len(dest_rows), "per_seed_rows": len(seed_rows), "results_match_rows": row_checks, "destination_match_rows": dest_checks, "per_seed_match_rows": seed_checks}

    # Check paired point deltas for all 30 prespecified comparisons; intervals
    # are audited below for the primary and secondary full-cohort contrasts.
    paired = json.loads((ROOT / "paired_contrasts.json").read_text(encoding="utf-8"))
    pair_ok = paired.get("n_boot") == N_BOOT and len(paired.get("comparisons", {})) == 30
    point_deltas = {}
    for cap in CAPS:
        for subset_name, mask in masks.items():
            for direction in DIRS:
                target_name = f"target_only|cap={cap:.2f}"; source_name = f"{direction}|cap={cap:.2f}"
                key = f"{direction}|cap={cap:.2f}|{subset_name}"
                if key not in paired.get("comparisons", {}):
                    pair_ok = False; continue
                tp, ta = expected_arms[target_name]; sp, sa = expected_arms[source_name]
                ts = stats(tp, ta, result["truth"], mask); ss = stats(sp, sa, result["truth"], mask)
                point = {"correct_yield": int(ss["correct"] - ts["correct"]), "incorrect_yield": int(ss["incorrect"] - ts["incorrect"]), "unreported": int(ss["unreported"] - ts["unreported"]), "coverage": float(ss["coverage"] - ts["coverage"])}
                point_deltas[key] = point
                entry = paired["comparisons"][key]
                for metric in ("correct_yield", "incorrect_yield", "unreported", "coverage"):
                    if metric not in entry.get("group_bootstrap", {}) or metric not in entry.get("record_bootstrap", {}):
                        pair_ok = False; continue
                    for unit_name in ("group_bootstrap", "record_bootstrap"):
                        obj = entry[unit_name][metric]
                        if metric == "coverage":
                            value = obj.get("point")
                        else:
                            count_obj = obj.get("count_delta", obj)
                            value = count_obj.get("point")
                        # The first three point fields are record-seed counts;
                        # coverage is already a fraction.
                        if value is None or abs(float(value) - point[metric]) > 1e-12:
                            pair_ok = False
    add_check(report, "paired_point_deltas", "PASS" if pair_ok else "FAIL", f"30个预设比较的point delta、n_boot=5000及所有指标结构一致={pair_ok}。")
    report["paired_point_deltas"] = point_deltas
    return {"expected_arms": expected_arms, "paired": paired, "masks": masks}


def bootstrap_weights(groups: np.ndarray, mask: np.ndarray, unit: str, rng_seed: int, n_boot: int) -> np.ndarray:
    rec = np.where(mask)[0]
    unique_groups = np.unique(groups[rec])
    rng = np.random.default_rng(rng_seed)
    samples = []
    if unit == "proxy_group":
        for _ in range(n_boot):
            picks = rng.integers(0, len(unique_groups), size=len(unique_groups))
            weights = np.bincount(picks, minlength=len(unique_groups))
            rw = np.zeros(len(groups), dtype=float)
            for g, w in zip(unique_groups, weights):
                rw[groups == g] = w
            samples.append(rw)
    elif unit == "record":
        for _ in range(n_boot):
            picks = rng.integers(0, len(rec), size=len(rec))
            weights = np.bincount(picks, minlength=len(rec))
            rw = np.zeros(len(groups), dtype=float); rw[rec] = weights
            samples.append(rw)
    else:
        raise ValueError(unit)
    return np.asarray(samples)


def bootstrap_local(pred_a: np.ndarray, act_a: np.ndarray, pred_b: np.ndarray, act_b: np.ndarray, truth: np.ndarray, groups: np.ndarray, mask: np.ndarray, unit: str, seed: int, n_boot: int = N_BOOT) -> dict[str, Any]:
    weights = bootstrap_weights(groups, mask, unit, seed, n_boot)
    t = truth[None, :]
    ca = act_a & (pred_a == t); cb = act_b & (pred_b == t)
    ea = act_a & (pred_a != t); eb = act_b & (pred_b != t)
    ua = ~act_a; ub = ~act_b
    denom = weights[:, mask].sum(axis=1) * len(SEEDS)
    out: dict[str, Any] = {}
    for key, aa, bb in (("correct_yield", ca, cb), ("incorrect_yield", ea, eb), ("unreported", ua, ub)):
        count_delta = np.einsum("sr,br->b", (bb.astype(float) - aa.astype(float)), weights)
        yield_delta = count_delta / denom
        out[key] = {
            "count_point": float(bb[:, mask].sum() - aa[:, mask].sum()),
            "count_mean": float(count_delta.mean()),
            "count_ci": [float(np.percentile(count_delta, 2.5)), float(np.percentile(count_delta, 97.5))],
            "yield_point": float((bb[:, mask].sum() - aa[:, mask].sum()) / (mask.sum() * len(SEEDS))),
            "yield_mean": float(yield_delta.mean()),
            "yield_ci": [float(np.percentile(yield_delta, 2.5)), float(np.percentile(yield_delta, 97.5))],
            "unit": unit,
            "n_boot": n_boot,
        }
    cov_count = np.einsum("sr,br->b", (act_b.astype(float) - act_a.astype(float)), weights)
    cov_delta = cov_count / denom
    out["coverage"] = {
        "count_point": float(act_b[:, mask].sum() - act_a[:, mask].sum()),
        "yield_point": float(act_b[:, mask].mean() - act_a[:, mask].mean()),
        "yield_mean": float(cov_delta.mean()),
        "yield_ci": [float(np.percentile(cov_delta, 2.5)), float(np.percentile(cov_delta, 97.5))],
        "unit": unit,
        "n_boot": n_boot,
    }
    return out


def check_bootstrap(result: dict[str, Any], pair_bundle: dict[str, Any], report: dict[str, Any]) -> None:
    if not pair_bundle:
        add_check(report, "bootstrap_intervals", "WAITING", "paired_contrasts.json 尚未可用。")
        return
    groups = result["group_codes"]
    masks = pair_bundle["masks"]
    pairs = pair_bundle["paired"]["comparisons"]
    out: dict[str, Any] = {}
    interval_ok = True
    for direction in DIRS:
        key = f"{direction}|cap=0.15|full201"
        entry = pairs[key]
        target = pair_bundle["expected_arms"]["target_only|cap=0.15"]
        source = pair_bundle["expected_arms"]["%s|cap=0.15" % direction]
        seed_group = 20261030 + 150 + len("full201")
        seed_record = 20301030 + 150 + len("full201")
        local_g = bootstrap_local(target[0], target[1], source[0], source[1], result["truth"], groups, masks["full201"], "proxy_group", seed_group)
        local_r = bootstrap_local(target[0], target[1], source[0], source[1], result["truth"], groups, masks["full201"], "record", seed_record)
        row = {"group": local_g, "record": local_r, "reported": entry}
        out[direction] = {"count_delta": {k: local_g[k]["count_point"] for k in ("correct_yield", "incorrect_yield", "unreported")},
                          "correct_yield_delta": local_g["correct_yield"]["yield_point"], "incorrect_yield_delta": local_g["incorrect_yield"]["yield_point"],
                          "correct_group_ci": local_g["correct_yield"]["yield_ci"], "correct_record_ci": local_r["correct_yield"]["yield_ci"],
                          "incorrect_group_ci": local_g["incorrect_yield"]["yield_ci"], "incorrect_record_ci": local_r["incorrect_yield"]["yield_ci"],
                          "coverage_group_ci": local_g["coverage"]["yield_ci"], "coverage_record_ci": local_r["coverage"]["yield_ci"]}
        # Always require the legacy count fields to match.  If execution added
        # explicit yield fields, require those too; otherwise mark the audit
        # as a semantic-gap warning rather than silently treating counts as rates.
        for unit, local, reported_key in (("group", local_g, "group_bootstrap"), ("record", local_r, "record_bootstrap")):
            rep = entry.get(reported_key, {})
            for metric in ("correct_yield", "incorrect_yield", "unreported"):
                r = rep.get(metric, {})
                count_obj = r.get("count_delta", r)
                if "point" not in count_obj or abs(float(count_obj["point"]) - local[metric]["count_point"]) > 1e-12:
                    interval_ok = False
                for nm, val in (("bootstrap_mean", local[metric]["count_mean"]), ("ci95_low", local[metric]["count_ci"][0]), ("ci95_high", local[metric]["count_ci"][1])):
                    if nm in count_obj and abs(float(count_obj[nm]) - val) > 1e-10:
                        interval_ok = False
                # New schema explicitly separates count delta and true yield
                # delta.  If an older artifact lacks the latter, retain the
                # independent recomputation as a semantic-gap warning.
                rate_obj = r.get("yield_delta") if isinstance(r.get("yield_delta"), dict) else None
                if rate_obj is not None:
                    for nm, val in (("point", local[metric]["yield_point"]), ("bootstrap_mean", local[metric]["yield_mean"]), ("ci95_low", local[metric]["yield_ci"][0]), ("ci95_high", local[metric]["yield_ci"][1])):
                        if nm in rate_obj and abs(float(rate_obj[nm]) - val) > 1e-10:
                            interval_ok = False
                else:
                    report.setdefault("semantic_warnings", []).append(f"{key}/{unit}/{metric}: reported schema exposes count delta only; independent yield-rate interval recomputed separately")
            cov = rep.get("coverage", {})
            if "point" not in cov or abs(float(cov["point"]) - local["coverage"]["yield_point"]) > 1e-12:
                interval_ok = False
            for nm, val in (("bootstrap_mean", local["coverage"]["yield_mean"]), ("ci95_low", local["coverage"]["yield_ci"][0]), ("ci95_high", local["coverage"]["yield_ci"][1])):
                if nm in cov and abs(float(cov[nm]) - val) > 1e-10:
                    interval_ok = False
        # Keep both count and rate intervals in the audit record; do not expose
        # a rate as if it came from the original implementation.
        out[direction]["group_count_ci"] = local_g["correct_yield"]["count_ci"]
        out[direction]["record_count_ci"] = local_r["correct_yield"]["count_ci"]
        out[direction]["group_yield_ci"] = local_g["correct_yield"]["yield_ci"]
        out[direction]["record_yield_ci"] = local_r["correct_yield"]["yield_ci"]
    report["primary_bootstrap"] = out
    add_check(report, "bootstrap_intervals", "PASS" if interval_ok else "FAIL", f"主/次 full201 cap=0.15 的5000次 proxy-group/record bootstrap与报告字段核对={interval_ok}；独立yield分母为每次重采样记录数×20。")


def manual_bootstrap_check(report: dict[str, Any]) -> None:
    groups = np.asarray([0, 0, 1, 2])
    mask = np.ones(4, dtype=bool)
    pred_a = np.asarray([[0, 0, 1, -1], [0, 1, 1, -1]])
    pred_b = np.asarray([[0, 0, 1, 1], [0, 1, 1, -1]])
    act_a = np.asarray([[1, 1, 1, 1], [1, 0, 1, 1]], dtype=bool)
    act_b = np.asarray([[1, 1, 1, 1], [1, 1, 0, 1]], dtype=bool)
    truth = np.asarray([0, 0, 1, -1])
    seed = 381
    n = 17
    got = bootstrap_weights(groups, mask, "proxy_group", seed, n)
    rng = np.random.default_rng(seed); explicit = []
    ug = np.unique(groups[mask])
    for _ in range(n):
        picks = rng.integers(0, len(ug), size=len(ug)); counts = np.bincount(picks, minlength=len(ug)); row = np.zeros(len(groups), float)
        for g, w in zip(ug, counts): row[groups == g] = w
        explicit.append(row)
    ok = np.array_equal(got, np.asarray(explicit))
    local = bootstrap_local(pred_a, act_a, pred_b, act_b, truth, groups, mask, "proxy_group", seed, n_boot=n)
    report["manual_small_sample"] = {"n_boot": n, "group_weight_generator_matches_explicit_loop": bool(ok), "example_correct_count_point": local["correct_yield"]["count_point"], "example_correct_yield_point": local["correct_yield"]["yield_point"]}
    add_check(report, "manual_small_bootstrap", "PASS" if ok else "FAIL", f"4-record/3-group、2-seed、17次抽样的显式循环与向量权重一致={ok}。")


def run() -> dict[str, Any]:
    report: dict[str, Any] = {"protocol": PROTOCOL, "status": "PASS", "issues": [], "boundary": [
        "仅重算冻结 predictor/error/head 与 source action；未重新拟合、训练或运行 increment_experiment.py。",
        "20 个 seed 是 record–seed 计数重复，不是 20 个新患者；proxy group 也不是已核实 patient ID。",
        "子集是既有201测试队列的 hash 固定敏感性复核，不是 independent validation。",
        "报告中若只保存 count delta，独立报告另算 yield delta；两者不可混写。",
    ]}
    required_core = [DATA, FOCUSED / "model_1000.joblib", ROOT / "input_manifest.json", ROOT / "subset_mask.json"]
    missing_core = [p.name for p in required_core if not p.exists()]
    if missing_core:
        report["status"] = "WAITING_FOR_EXECUTION_ARTIFACTS"; report["missing_artifacts"] = missing_core
        add_check(report, "core_inputs", "WAITING", "缺少核心输入：" + ", ".join(missing_core))
        write_report(report); return report
    try:
        D = load_data()
        m0 = joblib.load(FOCUSED / "model_1000.joblib")
        idx = np.asarray(m0["test"], dtype=int)
        check_manifest(report)
        pool = np.asarray(m0["pool"], dtype=int)
        pool_test_overlap = len(set(D["groups"][pool].astype(str)) & set(D["groups"][idx].astype(str)))
        geometry_ok = len(idx) == 201 and len(np.unique(idx)) == 201 and len(pool) == 60 and pool_test_overlap == 0
        add_check(report, "test_pool_geometry", "PASS" if geometry_ok else "FAIL", f"target pool/test={len(pool)}/{len(idx)}；pool-test proxy-group overlap={pool_test_overlap}。")
        result = load_recomputed(D, report)
        check_source_roles(D, idx, report)
        check_source_action_shape(result, D, report)
        subset_mask, complement_mask = check_subset(D, idx, report)
        check_development_freeze(D, idx, report)
        truth = result["truth"]
        e0 = {}
        target = result["target"]
        e0["Target-only Ridge|S14"] = stats(target["pred"], target["s14"], truth)
        for d in DIRS:
            src = result["sources"][d]
            e0[f"{d}|S14"] = stats(src["final_ma_class"], src["unified_action"], truth)
        report["e0"] = {k: {"released": v["released"], "correct": v["correct"], "incorrect": v["incorrect"]} for k, v in e0.items()}
        e0_ok = all(e0[k][x] == v[x] for k, v in S14_EXPECTED.items() for x in ("released", "correct", "incorrect"))
        add_check(report, "e0_s14_reproduction", "PASS" if e0_ok else "FAIL", f"独立重算三套S14 released/correct/incorrect与预期一致={e0_ok}。")
        # Source action order is anchored to the old eval/prediction seed vector;
        # source action archives themselves have no seed field.
        source_seed_ok = all(np.array_equal(result["old_source"][d]["seed"], np.asarray(SEEDS)) for d in DIRS)
        add_check(report, "source_seed_order_anchor", "PASS" if source_seed_ok else "FAIL", "source NPZ 无 seed 字段；以对应旧 eval/predictions 的 seed=[1000..1019]、test_global_index、MA/class/score 对齐锚定顺序。")

        # QC diagnostics on all new E1 policy arms.  This catches an action
        # archive that silently omitted the plan's explicit QC fail-closed rule.
        freeze = json.loads((ROOT / "frozen_thresholds.json").read_text(encoding="utf-8"))
        result["group_codes"] = np.unique(D["groups"][idx].astype(str), return_inverse=True)[1]
        target_arms_qc = build_arms(result, freeze, qc_masked=True)
        target_arms_unmasked = build_arms(result, freeze, qc_masked=False)
        qc_false = ~D["input_qc"][idx].astype(bool)
        for name, (_, a) in target_arms_unmasked.items():
            stored = None
            if name in target_arms_qc:
                stored = target_arms_qc[name][1]
            qc_diagnostics = report.setdefault("qc_diagnostics", {})
            qc_diagnostics[name] = {"qc_false_records": int(qc_false.sum()), "unmasked_release_events": int((a & qc_false[None, :]).sum()), "stored_release_events_qc_false": int((stored & qc_false[None, :]).sum()) if stored is not None else None, "qc_mask_difference_events": int(np.sum(a != stored)) if stored is not None else None}

        bundle = check_private_and_tables(result, subset_mask, complement_mask, report)
        check_bootstrap(result, bundle, report)
        manual_bootstrap_check(report)
    except Exception as exc:
        report["status"] = "FAIL"
        add_check(report, "audit_runtime", "FAIL", f"独立验收脚本异常：{type(exc).__name__}: {exc}")
    # A waiting state is only used when post-freeze execution artifacts are not
    # all present.  Any actual failed check takes precedence over waiting.
    if report.get("issues"):
        report["status"] = "FAIL"
    elif report.get("missing_artifacts"):
        report["status"] = "WAITING_FOR_EXECUTION_ARTIFACTS"
    else:
        report["status"] = "PASS"
    write_report(report)
    return report


if __name__ == "__main__":
    run()
