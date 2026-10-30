"""External-test replay for the frozen 5/10-min prefix branches.

The target-prefix workers are fit and frozen before this script reads any
target-test labels.  It uses exactly the 15-min evaluator's prediction path;
only the observed-prefix four-parameter input and frozen prefix model change.
"""
import json
import sys
import time
from pathlib import Path

import runtime as r
import joblib
import numpy as np

from evaluate_15min import _load_predictions

TARGETS = ("Huzhou", "SAHZU", "Hubei")
SEEDS = tuple(range(1000, 1020))


def _summarize(payload, target, budget, horizon):
    yp = payload["pred_class"]
    y = payload["reference_class"][0]
    act = payload["action"]
    known = y != 9
    abnormal = known & (y != 0)
    error = (yp != y[None, :]) & known[None, :]
    early = act & known[None, :]
    n_early = early.sum(axis=1)
    wrong = (early & error).sum(axis=1)
    ab_false = (early & abnormal[None, :] & (yp == 0)).sum(axis=1)
    ab_correct = (early & abnormal[None, :] & (yp == y[None, :])).sum(axis=1)
    base_correct = (yp[:, known] == y[None, known]).sum(axis=1)
    return dict(
        horizon_seconds=int(horizon), target=target, budget=budget,
        n_records=int(y.size), known_n=int(known.sum()), abnormal_n=int(abnormal.sum()),
        base_accuracy=float(base_correct.mean() / known.sum()) if known.any() else None,
        coverage=float(early.mean()), early_n=float(n_early.mean()),
        early_error=float(wrong.sum() / early.sum()) if early.sum() else None,
        wait_fraction=float(1.0 - early.mean()),
        abnormal_early_n=float((early & abnormal[None, :]).sum(axis=1).mean()),
        abnormal_false_normal_n=float(ab_false.mean()),
        abnormal_correct_n=float(ab_correct.mean()),
        abnormal_false_normal_rate=float(ab_false.mean() / abnormal.sum()) if abnormal.any() else None,
        abnormal_early_recall=float(ab_correct.mean() / abnormal.sum()) if abnormal.any() else None,
        J10=float(np.mean(np.where(early, error, 0.10))),
        selected_gate_counts={str(k): int(v) for k, v in zip(*np.unique(payload["selected_gate"], return_counts=True))},
        selected_adapter_counts={str(k): int(v) for k, v in zip(*np.unique(payload["selected_adapter"], return_counts=True))},
    )


def _run_horizon(horizon):
    freeze_candidates = [r.ROOT / f"PREFIX_TARGET_{horizon}_ALL_FROZEN.json",
                         r.ROOT / "PREFIX_TARGET_FULL_ONLY_ALL_FROZEN.json"]
    freeze_path = next((p for p in freeze_candidates if p.exists()), None)
    if freeze_path is None:
        raise RuntimeError(f"a frozen prefix target manifest is required before external replay: {freeze_candidates}")
    P = np.load(r.ROOT / "PREFIX_X.npz", allow_pickle=True)
    horizons = [int(x) for x in P["horizon_seconds"]]
    hi = horizons.index(int(horizon))
    D = {k: v for k, v in np.load(r.ROOT / "DATA15.npz", allow_pickle=True).items()}
    D["x"] = np.asarray(P["x"][hi], dtype=float)
    role = r.read(r.ROOT / "ROLE_FREEZE.json")
    outdir = r.ROOT / f"evalprefix_{horizon}"
    outdir.mkdir(parents=True, exist_ok=True)
    summary, seed_rows = [], []
    for target in TARGETS:
        test = list(role["roles"][target]["test"])
        requested = []
        for budget, label in (("C20", "20"), ("CFULL", "FULL")):
            paths = [r.ROOT / f"targetprefix_{horizon}" / f"{target}_{budget}_seed{seed}.joblib" for seed in SEEDS]
            if all(p.exists() for p in paths):
                requested.append((budget, label))
        # A prefix stage may be intentionally frozen with only FULL targets
        # (for example when the C20 branch was already recorded as an extra
        # pilot).  Do not invent a C20 result or force a refit just to make a
        # rectangular table.
        for budget, label in requested:
            rows = []
            for seed in SEEDS:
                p = r.ROOT / f"targetprefix_{horizon}" / f"{target}_{budget}_seed{seed}.joblib"
                if not p.exists():
                    raise FileNotFoundError(p)
                rows.append(_load_predictions(D, role, target, budget, seed, joblib.load(p)))
            keys = ["raw", "med", "huber_med", "pred_ci", "pred_class", "score", "allowed", "action", "qc_forced_wait", "reference_ci", "reference_class"]
            payload = {k: np.stack([q[k] for q in rows], axis=0) for k in keys}
            payload.update(seed=np.asarray(SEEDS), test=np.asarray(test, dtype=int),
                           test_ids=np.asarray(D["ids"][test]),
                           selected_adapter=np.asarray([q["selected_adapter"] for q in rows]),
                           selected_gate=np.asarray([q["selected_gate"] for q in rows]),
                           threshold=np.asarray([np.nan if q["threshold"] is None else q["threshold"] for q in rows]))
            np.savez_compressed(outdir / f"{target}_{label}.npz", **payload)
            summary.append(_summarize(payload, target, budget, horizon))
            yp, y, act = payload["pred_class"], payload["reference_class"][0], payload["action"]
            known = y != 9
            abnormal = known & (y != 0)
            for i, seed in enumerate(SEEDS):
                early = act[i] & known
                err = (yp[i] != y) & known
                seed_rows.append(dict(horizon_seconds=int(horizon), target=target, budget=budget, seed=int(seed),
                                      n_records=int(y.size), known_n=int(known.sum()),
                                      base_correct=int((yp[i, known] == y[known]).sum()),
                                      base_accuracy=float((yp[i, known] == y[known]).mean()),
                                      early_n=int(early.sum()), coverage=float(early.mean()),
                                      early_correct=int((early & ~err).sum()), early_wrong=int((early & err).sum()),
                                      abnormal_n=int(abnormal.sum()),
                                      abnormal_false_normal=int((early & abnormal & (yp[i] == 0)).sum()),
                                      abnormal_correct=int((early & abnormal & (yp[i] == y)).sum())))
    return summary, seed_rows


def main():
    all_summary, all_seed = [], []
    horizons = tuple(int(x) for x in sys.argv[1:]) if len(sys.argv) > 1 else (300, 600)
    for horizon in horizons:
        rows, seed_rows = _run_horizon(horizon)
        all_summary.extend(rows)
        all_seed.extend(seed_rows)
    tag = "_".join(str(x) for x in horizons)
    summary_name = "EVALUATION_PREFIX_SUMMARY_V3.json" if len(horizons) > 1 else f"EVALUATION_PREFIX_SUMMARY_{tag}_V3.json"
    seed_name = "EVALUATION_PREFIX_SEED_SUMMARY_V3.json" if len(horizons) > 1 else f"EVALUATION_PREFIX_SEED_SUMMARY_{tag}_V3.json"
    r.write(summary_name, dict(
        version="legacy_aligned_threecenter_prefix_eval_v3", time=time.time(), rows=all_summary,
        source=r.sha(r.ROOT / "PREFIX_SOURCE_FREEZE_V2.json"),
        target300=r.sha(r.ROOT / "PREFIX_TARGET_300_ALL_FROZEN.json") if (r.ROOT / "PREFIX_TARGET_300_ALL_FROZEN.json").exists() else None,
        target600=r.sha(r.ROOT / "PREFIX_TARGET_600_ALL_FROZEN.json") if (r.ROOT / "PREFIX_TARGET_600_ALL_FROZEN.json").exists() else None,
        target_full_only=r.sha(r.ROOT / "PREFIX_TARGET_FULL_ONLY_ALL_FROZEN.json") if (r.ROOT / "PREFIX_TARGET_FULL_ONLY_ALL_FROZEN.json").exists() else None,
        test_labels_read_after_freeze=True, curve_metrics_not_evaluated=True,
        abnormal_correct_definition="reported class equals reference class and reference is abnormal",
    ))
    r.write(seed_name, dict(version="legacy_aligned_threecenter_prefix_seed_v2",
          time=time.time(), rows=all_seed))
    print(json.dumps(all_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
