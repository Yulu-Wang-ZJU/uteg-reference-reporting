from __future__ import annotations

"""Read-only presentation and class-flow summaries for the equal-error study.

This module consumes the frozen policy-action archive and the already computed D2/D3
selection records.  It does not fit models, choose thresholds, or alter any frozen
artifact.  It is intentionally separate from equal_error_analysis.py and its audit.
Run only after the D3 producer has passed its own validation.
"""

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


# The Windows scientific environment has Chinese fonts installed; selecting one
# here keeps the diagnostic figure readable instead of silently falling back to
# DejaVu Sans (which lacks the CJK glyphs).
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parent
ACTION_NPZ = ROOT / "policy_actions_private.npz"
ACTION_MAP = ROOT / "policy_actions_map.json"
D2_FULL = ROOT / "d2_post_full.json"
D2_BOOT = ROOT / "d2_post_bootstrap.json"
D2_POINTS = ROOT / "d2_all_strategy_points.csv"
D3_ROWS = ROOT / "d3_fold_results.csv"

METHODS = ["target_only", "HZ_to_SA", "HZ+HB_to_SA"]
METHOD_LABEL = {
    "target_only": "target-only",
    "HZ_to_SA": "HZ→SA",
    "HZ+HB_to_SA": "HZ+HB→SA",
}
SEEDS = list(range(1000, 1020))
TRUE_STATES = [-1, 0, 1]
REPORT_STATES = [-1, 0, 1, 9]
REPORT_LABEL = {-1: "low", 0: "within", 1: "high", 9: "unreported"}
STATE_LABEL = {-1: "low", 0: "within", 1: "high"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def fnum(x: Any) -> float:
    if x is None or x == "" or (isinstance(x, str) and x.lower() in {"na", "nan", "none"}):
        return float("nan")
    return float(x)


def int_or_none(x: Any) -> int | None:
    if x is None or x == "" or str(x).lower() in {"na", "nan", "none"}:
        return None
    return int(float(x))


def safe_config(config_id: str, mapping: dict[str, dict[str, Any]]) -> str:
    for safe, entry in mapping.items():
        if entry.get("config_id") == config_id:
            return safe
    # This fallback is useful for the simple target-only ids, but should never be
    # needed for the frozen map.  Do not silently normalize plus signs to avoid
    # confusing two different identifiers.
    candidate = config_id.replace("+", "plus")
    if candidate in mapping:
        return candidate
    raise KeyError(f"config id absent from policy_actions_map.json: {config_id}")


def config_index(config_id: str, points_by_config: dict[str, dict[str, Any]]) -> int:
    if config_id in points_by_config and points_by_config[config_id].get("strategy_index") is not None:
        return int(points_by_config[config_id]["strategy_index"])
    m = re.search(r"cfg(\d+)$", config_id)
    return int(m.group(1)) if m else -1


def load_context() -> dict[str, Any]:
    mapping = json.loads(ACTION_MAP.read_text(encoding="utf-8"))
    z = np.load(ACTION_NPZ, allow_pickle=False)
    truth = z["truth"].astype(int)
    groups = z["group"].astype(int)
    folds = z["fold"].astype(int)
    assert truth.shape == groups.shape == folds.shape == (201,)
    points = read_csv(D2_POINTS)
    points_by_config: dict[str, dict[str, Any]] = {}
    for r in points:
        cfg = str(r["config_id"])
        points_by_config[cfg] = {
            "method": r["method"],
            "strategy_index": int(r["strategy_index"]),
            "error_yield": fnum(r["error_yield"]),
            "correct_yield": fnum(r["correct_yield"]),
            "coverage_yield": fnum(r["coverage_yield"]),
        }
    # All-withhold is absent from d2_all_strategy_points.csv in some producer
    # versions, so add it explicitly as the common origin.
    for safe, entry in mapping.items():
        cfg = str(entry["config_id"])
        if cfg not in points_by_config:
            points_by_config[cfg] = {
                "method": entry["method"],
                "strategy_index": int(entry.get("strategy_index", -1)),
                "error_yield": 0.0,
                "correct_yield": 0.0,
                "coverage_yield": 0.0,
            }
    return {
        "mapping": mapping,
        "z": z,
        "truth": truth,
        "groups": groups,
        "folds": folds,
        "points": points_by_config,
        "denom": int(truth.size * len(SEEDS)),
    }


def action_arrays(ctx: dict[str, Any], config_id: str) -> tuple[np.ndarray, np.ndarray]:
    safe = safe_config(config_id, ctx["mapping"])
    z = ctx["z"]
    a = z[f"action__{safe}"].astype(bool)
    p = z[f"pred__{safe}"].astype(int)
    assert a.shape == p.shape == (20, 201)
    return a, p


def destination_counts(ctx: dict[str, Any], config_id: str, mask: np.ndarray | None = None) -> np.ndarray:
    a, p = action_arrays(ctx, config_id)
    truth = ctx["truth"]
    if mask is None:
        mask = np.ones(truth.size, dtype=bool)
    report = np.where(a, p, 9)
    out = np.zeros((3, 4), dtype=float)
    for ti, t in enumerate(TRUE_STATES):
        for ri, r in enumerate(REPORT_STATES):
            out[ti, ri] = float(((report[:, mask] == r) & (truth[mask][None, :] == t)).sum())
    return out


def event_stats(ctx: dict[str, Any], config_id: str, mask: np.ndarray | None = None) -> dict[str, float]:
    a, p = action_arrays(ctx, config_id)
    truth = ctx["truth"]
    if mask is None:
        mask = np.ones(truth.size, dtype=bool)
    aa = a[:, mask]
    pp = p[:, mask]
    tt = truth[mask][None, :]
    correct = aa & (pp == tt)
    error = aa & (pp != tt)
    released = aa
    denom = float(aa.size)
    c = float(correct.sum())
    e = float(error.sum())
    r = float(released.sum())
    return {
        "correct": c,
        "error": e,
        "released": r,
        "unreported": denom - r,
        "denom": denom,
        "correct_yield": c / denom,
        "error_yield": e / denom,
        "coverage": r / denom,
        "agreement": c / r if r else float("nan"),
        "zero_report_seeds": float(np.sum(released.sum(axis=1) == 0)),
    }


def mix_counts(ctx: dict[str, Any], cfg_i: str, cfg_j: str, weight_j: float,
               mask: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, float]]:
    wi = 1.0 - float(weight_j)
    wj = float(weight_j)
    assert -1e-10 <= wj <= 1.0 + 1e-10
    wj = min(1.0, max(0.0, wj))
    di = destination_counts(ctx, cfg_i, mask)
    dj = destination_counts(ctx, cfg_j, mask)
    flow = wi * di + wj * dj
    si = event_stats(ctx, cfg_i, mask)
    sj = event_stats(ctx, cfg_j, mask)
    s = {k: wi * si[k] + wj * sj[k] for k in ["correct", "error", "released", "unreported"]}
    s["denom"] = si["denom"]
    s["correct_yield"] = s["correct"] / s["denom"]
    s["error_yield"] = s["error"] / s["denom"]
    s["coverage"] = s["released"] / s["denom"]
    s["agreement"] = s["correct"] / s["released"] if s["released"] else float("nan")
    # A mixture has zero expected release only when both complete policies do.
    s["zero_report_seeds"] = float("nan")
    return flow, s


def all_policy_destinations(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    denom = ctx["denom"]
    for safe, entry in ctx["mapping"].items():
        cfg = str(entry["config_id"])
        method = str(entry["method"])
        idx = config_index(cfg, ctx["points"])
        flow = destination_counts(ctx, cfg)
        for ti, t in enumerate(TRUE_STATES):
            for ri, r in enumerate(REPORT_STATES):
                count = float(flow[ti, ri])
                rows.append({
                    "method": method,
                    "config_id": cfg,
                    "strategy_index": idx,
                    "true_state": t,
                    "true_label": STATE_LABEL[t],
                    "report_state": r,
                    "report_label": REPORT_LABEL[r],
                    "expected_count": count,
                    "denom": denom,
                    "expected_yield": count / denom,
                    "all_withhold": bool(entry.get("all_withhold", False)),
                })
    return rows


def load_d2_mix(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    d2 = json.loads(D2_FULL.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        for budget_s, record in d2["matching"][method].items():
            if budget_s == "hull":
                continue
            exact = record.get("exact", {})
            cfg_i = str(exact["strategy_i_config_id"])
            cfg_j = str(exact["strategy_j_config_id"])
            wj = float(exact["weight_j"])
            flow, stats = mix_counts(ctx, cfg_i, cfg_j, wj)
            budget = float(budget_s)
            for ti, t in enumerate(TRUE_STATES):
                for ri, r in enumerate(REPORT_STATES):
                    count = float(flow[ti, ri])
                    rows.append({
                        "method": method,
                        "budget": budget,
                        "kind": "exact_mix",
                        "strategy_i_config_id": cfg_i,
                        "strategy_j_config_id": cfg_j,
                        "weight_j": wj,
                        "true_state": t,
                        "true_label": STATE_LABEL[t],
                        "report_state": r,
                        "report_label": REPORT_LABEL[r],
                        "expected_count": count,
                        "denom": ctx["denom"],
                        "expected_yield": count / ctx["denom"],
                        "exact_reachable": bool(exact.get("exact", False)),
                        "reported_error_yield": stats["error_yield"],
                        "reported_correct_yield": stats["correct_yield"],
                    })
    return rows


def d2_per_seed(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    d2 = json.loads(D2_FULL.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        for budget_s, record in d2["matching"][method].items():
            if budget_s == "hull":
                continue
            exact = record.get("exact", {})
            cfg_i = str(exact["strategy_i_config_id"])
            cfg_j = str(exact["strategy_j_config_id"])
            wj = min(1.0, max(0.0, float(exact["weight_j"])))
            ai, pi = action_arrays(ctx, cfg_i)
            aj, pj = action_arrays(ctx, cfg_j)
            truth = ctx["truth"]
            for j, seed in enumerate(SEEDS):
                ci = float((ai[j] & (pi[j] == truth)).sum())
                ei = float((ai[j] & (pi[j] != truth)).sum())
                ri = float(ai[j].sum())
                cj = float((aj[j] & (pj[j] == truth)).sum())
                ej = float((aj[j] & (pj[j] != truth)).sum())
                rj = float(aj[j].sum())
                c = (1 - wj) * ci + wj * cj
                e = (1 - wj) * ei + wj * ej
                rel = (1 - wj) * ri + wj * rj
                rows.append({
                    "method": method,
                    "budget": float(budget_s),
                    "seed": seed,
                    "strategy_i_config_id": cfg_i,
                    "strategy_j_config_id": cfg_j,
                    "weight_j": wj,
                    "correct_expected": c,
                    "error_expected": e,
                    "unreported_expected": 201.0 - rel,
                    "release_expected": rel,
                    "correct_yield": c / 201.0,
                    "error_yield": e / 201.0,
                    "coverage": rel / 201.0,
                    "agreement": c / rel if rel else float("nan"),
                    "zero_report": bool(ri == 0 and rj == 0),
                    "exact_reachable": bool(exact.get("exact", False)),
                })
    return rows


def d3_records(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    if not D3_ROWS.exists():
        raise FileNotFoundError(D3_ROWS)
    rows = read_csv(D3_ROWS)
    required = {"method", "fold", "budget", "strategy_i", "strategy_j", "weight_j"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise ValueError(f"d3_fold_results.csv missing columns: {sorted(missing)}")
    return rows


def d3_outputs(ctx: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = d3_records(ctx)
    truth = ctx["truth"]
    fold_values = sorted(set(int(r["fold"]) for r in rows))
    expected_dest: dict[tuple[str, float, int, int], float] = defaultdict(float)
    per_seed: dict[tuple[str, float, int], dict[str, float]] = defaultdict(lambda: {"correct": 0.0, "error": 0.0, "release": 0.0})
    selected_folds: dict[tuple[str, float], set[int]] = defaultdict(set)
    selected_rows = 0
    for r in rows:
        method = str(r["method"])
        fold = int(r["fold"])
        budget = float(r["budget"])
        cfg_i = str(r["strategy_i"])
        cfg_j = str(r["strategy_j"])
        if not cfg_i or cfg_i.lower() in {"nan", "none", "na"}:
            continue
        if not cfg_j or cfg_j.lower() in {"nan", "none", "na"}:
            cfg_j = cfg_i
        if "status" in r and r["status"] and r["status"].upper() not in {"OK", "SELECTED", "VALID", "COMPLETE"}:
            continue
        wj = min(1.0, max(0.0, float(r["weight_j"])))
        mask = ctx["folds"] == fold
        flow, _ = mix_counts(ctx, cfg_i, cfg_j, wj, mask)
        selected_rows += 1
        selected_folds[(method, budget)].add(fold)
        for ti, t in enumerate(TRUE_STATES):
            for ri, rep in enumerate(REPORT_STATES):
                expected_dest[(method, budget, t, rep)] += float(flow[ti, ri])
        ai, pi = action_arrays(ctx, cfg_i)
        aj, pj = action_arrays(ctx, cfg_j)
        for j, seed in enumerate(SEEDS):
            ci = float((ai[j, mask] & (pi[j, mask] == truth[mask])).sum())
            ei = float((ai[j, mask] & (pi[j, mask] != truth[mask])).sum())
            ri = float(ai[j, mask].sum())
            cj = float((aj[j, mask] & (pj[j, mask] == truth[mask])).sum())
            ej = float((aj[j, mask] & (pj[j, mask] != truth[mask])).sum())
            rj = float(aj[j, mask].sum())
            key = (method, budget, seed)
            per_seed[key]["correct"] += (1 - wj) * ci + wj * cj
            per_seed[key]["error"] += (1 - wj) * ei + wj * ej
            per_seed[key]["release"] += (1 - wj) * ri + wj * rj
    if selected_rows == 0:
        raise ValueError("d3_fold_results.csv contained no valid selection rows")
    expected_dest_rows: list[dict[str, Any]] = []
    full_denom = ctx["denom"]
    for (method, budget, t, rep), count in sorted(expected_dest.items()):
        expected_dest_rows.append({
            "method": method,
            "budget": budget,
            "kind": "d3_holdout_mix",
            "true_state": t,
            "true_label": STATE_LABEL[t],
            "report_state": rep,
            "report_label": REPORT_LABEL[rep],
            "expected_count": count,
            "denom": full_denom,
            "expected_yield": count / full_denom,
            "valid_fold_rows": len(selected_folds[(method, budget)]),
            "actual_fold_count": len(selected_folds[(method, budget)]),
            "total_selection_rows": selected_rows,
        })
    per_seed_rows: list[dict[str, Any]] = []
    for (method, budget, seed), s in sorted(per_seed.items()):
        denom = 201.0
        per_seed_rows.append({
            "method": method,
            "budget": budget,
            "seed": seed,
            "correct_expected": s["correct"],
            "error_expected": s["error"],
            "unreported_expected": 201.0 - s["release"],
            "release_expected": s["release"],
            "correct_yield": s["correct"] / denom,
            "error_yield": s["error"] / denom,
            "coverage": s["release"] / denom,
            "agreement": s["correct"] / s["release"] if s["release"] else float("nan"),
            "zero_report": bool(s["release"] == 0),
            "valid_fold_rows": len(selected_folds[(method, budget)]),
            "actual_fold_count": len(selected_folds[(method, budget)]),
            "total_selection_rows": selected_rows,
        })
    return expected_dest_rows, per_seed_rows


def hull_from_json(ctx: dict[str, Any], method: str) -> list[dict[str, float]]:
    d2 = json.loads(D2_FULL.read_text(encoding="utf-8"))
    return [
        {"error_yield": float(p["error_yield"]), "correct_yield": float(p["correct_yield"]),
         "config_id": str(p.get("config_id", "")), "strategy_index": int(p.get("strategy_index", -1))}
        for p in d2["matching"][method].get("hull", [])
    ]


def plot_outputs(ctx: dict[str, Any]) -> None:
    points = read_csv(D2_POINTS)
    boot = json.loads(D2_BOOT.read_text(encoding="utf-8"))
    budgets = [0.005, 0.01, 0.02, 0.05, 0.075, 0.1]
    colors = {"target_only": "#4b5563", "HZ_to_SA": "#2563eb", "HZ+HB_to_SA": "#c62828"}
    markers = {"target_only": "o", "HZ_to_SA": "s", "HZ+HB_to_SA": "^"}
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.8, 5.2), constrained_layout=True)
    for method in METHODS:
        pts = [r for r in points if r["method"] == method]
        x = np.array([fnum(r["error_yield"]) * 100 for r in pts])
        y = np.array([fnum(r["correct_yield"]) * 100 for r in pts])
        ax0.scatter(x, y, s=14, color=colors[method], alpha=0.22, marker=markers[method], linewidths=0)
        hull = hull_from_json(ctx, method)
        if hull:
            hx = [p["error_yield"] * 100 for p in hull]
            hy = [p["correct_yield"] * 100 for p in hull]
            ax0.plot(hx, hy, color=colors[method], lw=2.0, label=METHOD_LABEL[method])
    ax0.set_xlabel("错误报告比例（%）")
    ax0.set_ylabel("正确报告比例（%）")
    ax0.set_title("D2：策略族的错误–正确前沿")
    ax0.grid(alpha=0.25)
    ax0.legend(frameon=False, fontsize=9)
    ax0.text(0.02, 0.03, "线段是标签参与定位的随机混合期望\n不是确定性部署阈值", transform=ax0.transAxes,
             fontsize=8, color="#374151", va="bottom")

    comp = boot["comparisons"]["HZ+HB_to_SA"]
    x = np.arange(len(budgets))
    point = []
    low = []
    high = []
    valid = []
    for b in budgets:
        rec = comp[str(b)]
        item = rec["exact_c"]
        point.append(100 * float(rec["point"]["exact_correct_delta"]))
        low.append(100 * float(item["ci95_low"]))
        high.append(100 * float(item["ci95_high"]))
        valid.append(int(item["valid"]))
    point = np.asarray(point); low = np.asarray(low); high = np.asarray(high)
    ax1.errorbar(x, point, yerr=[point - low, high - point], fmt="o", color="#c62828",
                 capsize=4, lw=1.4)
    ax1.axhline(0, color="#111827", lw=0.8)
    ax1.set_xticks(x, [f"{100*b:g}%" for b in budgets])
    ax1.set_xlabel("目标错误负担")
    ax1.set_ylabel("双源−target-only 正确报告差（百分点）")
    ax1.set_title("D2：精确等错误的配对区间")
    ax1.set_ylim(float(np.nanmin(low)) - 1.0, float(np.nanmax(high)) + 2.0)
    ax1.grid(alpha=0.25)
    for i, n in enumerate(valid):
        ax1.text(i, high[i] + 0.8, f"n={n}/5000", ha="center", va="bottom", fontsize=7)
    ax1.text(0.02, 0.03, "10%点有569次无法精确匹配；区间条件于可达样本\n点态描述区间；不代表非劣或安全保证", transform=ax1.transAxes,
             fontsize=8, color="#374151", va="bottom")
    fig.savefig(ROOT / "equal_error_diagnostic.png", dpi=220)
    fig.savefig(ROOT / "equal_error_diagnostic.pdf")
    plt.close(fig)


def notes(ctx: dict[str, Any], d2_rows: list[dict[str, Any]], d3_rows: list[dict[str, Any]]) -> None:
    d2 = json.loads(D2_FULL.read_text(encoding="utf-8"))
    boot = json.loads(D2_BOOT.read_text(encoding="utf-8"))
    lines = [
        "# 等错误诊断呈现说明",
        "",
        "本目录只读取冻结的策略动作、D2匹配结果和D3留出折选择记录；没有重新训练、调参或修改冻结文件。所有D2线段代表两个冻结完整策略按记录–seed事件随机混合后的期望流，线段定位使用测试标签，因此是回顾性、label-aware诊断，不是已经部署的确定性阈值。",
        "",
        "## 输出文件",
        "",
        "- `all_policy_destinations.csv`：D1策略族中每个冻结配置的真实类别×报告目的地（low/within/high/unreported）。",
        "- `d2_expected_destinations.csv`：六个错误预算处的精确混合3×4期望流；`d2_expected_per_seed.csv`保留20个seed的global mix，不在seed间重新选策略。",
        "- `d3_expected_destinations.csv`和`d3_expected_per_seed.csv`：固定5折留出混合的完整类别流和逐seed汇总；实际错误负担仍需与目标校准预算分开解读。",
        "- `equal_error_diagnostic.png/pdf`：所有策略点、上方凹包络和双源精确等错误正确报告差的配对区间。",
        "",
        "## 解释边界",
        "",
        "D2精确混合可在预算网格上达到声明的错误yield，但这是期望值，不是确定性阈值实测值。D2确定性预算结果（见`d2_post_full.json`）仍需看实际错误yield，不能把预算以内误写成精确等错误。D3使用既有测试队列的校准标签，只是内部交叉校准诊断，不是独立验证，也不能替代原60例开发端部署流程。",
        "",
        f"D2预算：{', '.join(f'{100*b:g}%' for b in d2['budgets'])}。D2完整样本分母为{ctx['denom']}个record–seed事件；20个seed不是20个独立病例。",
        "",
        "在双源10%精确等错误点，bootstrap有效次数不足5000时必须连同有效次数阅读；当前结果中有效次数为4431/5000。条件区间不能按普通显著性直接解释。预算以内的5000次结果另行报告，不能用它们补救精确匹配的不可达重复。",
        "",
        "全部真实类别均保留；正常类别增量不能自动被视为无价值，但也不能用正常类别占多数来声称低值/高值场景已验证。",
    ]
    (ROOT / "PRESENTATION_NOTES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ctx = load_context()
    all_rows = all_policy_destinations(ctx)
    write_csv(ROOT / "all_policy_destinations.csv", all_rows)
    d2_dest = load_d2_mix(ctx)
    write_csv(ROOT / "d2_expected_destinations.csv", d2_dest)
    d2_seed = d2_per_seed(ctx)
    write_csv(ROOT / "d2_expected_per_seed.csv", d2_seed)
    d3_dest, d3_seed = d3_outputs(ctx)
    write_csv(ROOT / "d3_expected_destinations.csv", d3_dest)
    write_csv(ROOT / "d3_expected_per_seed.csv", d3_seed)
    plot_outputs(ctx)
    notes(ctx, d2_dest, d3_dest)
    print(json.dumps({
        "all_policy_destinations": len(all_rows),
        "d2_expected_destinations": len(d2_dest),
        "d2_expected_per_seed": len(d2_seed),
        "d3_expected_destinations": len(d3_dest),
        "d3_expected_per_seed": len(d3_seed),
        "plot": ["equal_error_diagnostic.png", "equal_error_diagnostic.pdf"],
        "status": "PRESENTATION_COMPLETE",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
