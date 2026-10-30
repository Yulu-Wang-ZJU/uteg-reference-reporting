#!/usr/bin/env python3
"""Run a deterministic synthetic smoke without private inputs or model files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RELEASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE / "src"))

from uteg_reference_reporting import (  # noqa: E402
    apply_calibration,
    auxiliary_j,
    class_actions,
    cumulative_first_report,
    fit_calibration,
    fit_ma_error_lr,
    ma_class,
    ma_features,
    metrics,
    raw_ridge10,
    score_ma_error_lr,
)


def main() -> None:
    rng = np.random.default_rng(20261030)
    x = rng.normal(size=(32, 4))
    readouts = np.column_stack(
        (
            1.0 + 0.2 * rng.normal(size=32),
            2.0 + 0.3 * rng.normal(size=32),
            60.0 + 8.0 * rng.normal(size=32),
            45.0 + 5.0 * rng.normal(size=32),
        )
    )
    readouts[:, 2] = np.linspace(42.0, 78.0, len(readouts))
    assert ma_class([49.999, 50.0, 70.0, 70.001, np.nan]).tolist() == [-1, 0, 0, 1, 9]

    features = ma_features(x, readouts, np.zeros(4), np.ones(4))
    assert features.shape == (32, 13)
    predicted = ma_class(readouts[:, 2])
    truth = np.roll(predicted, 3)
    gate = fit_ma_error_lr(features, predicted, truth, fit_id="synthetic", seed=20261030)
    scores = score_ma_error_lr(gate, features, predicted)
    assert scores.shape == (32,)
    assert np.isfinite(scores).all()

    source_model = {
        "xm": np.zeros(4),
        "xs": np.ones(4),
        "ym": np.zeros(4),
        "ys": np.ones(4),
        "coef": np.eye(4),
        "intercept": np.zeros(4),
    }
    raw = raw_ridge10(source_model, x)
    adaptation = fit_calibration(raw, raw + 0.05, source_model, "joint")
    adjusted = apply_calibration(adaptation, raw)
    assert adjusted.shape == raw.shape and np.isfinite(adjusted).all()
    assert np.isfinite(auxiliary_j(adjusted)).all()

    action_5 = np.zeros((2, 4), dtype=bool)
    action_5[0, 0] = True
    action_10 = action_5.copy()
    action_10[1, 1] = True
    action_15 = action_10.copy()
    pred_5 = np.zeros((2, 4), dtype=np.int8)
    pred_10 = pred_5.copy()
    pred_10[1, 1] = 1
    pred_15 = pred_10.copy()
    replay = cumulative_first_report(
        [(300, action_5, pred_5), (600, action_10, pred_10), (900, action_15, pred_15)]
    )
    assert replay["first_time"].tolist() == [[300, -1, -1, -1], [-1, 600, -1, -1]]
    final_metrics = metrics(replay["reported"], replay["first_class"], np.zeros(4, dtype=np.int8))
    assert final_metrics["released"] == 2

    release_actions = class_actions(
        np.array([0, -1, 1]),
        np.array([0.1, 0.8, 0.9]),
        np.array([0.2, 0.8, 0.8]),
        np.array([True, True, True]),
        {"normal": 0.5, "tail": 0.5, "all_withhold": False},
    )
    assert release_actions.tolist() == [True, False, False]
    print(json.dumps({"status": "PASS", "features": list(features.shape), "gate": gate["id"], "released": final_metrics["released"]}))


if __name__ == "__main__":
    main()
