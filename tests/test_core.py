from __future__ import annotations

import ast
import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uteg_reference_reporting.core import (  # noqa: E402
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


RELEASE = Path(__file__).resolve().parents[1]


def _ast_functions(path: Path, names: set[str], namespace: dict[str, Any]) -> dict[str, Any]:
    """Load only selected definitions from an exact source snapshot."""

    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    body = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def _import_snapshot(path: Path, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import snapshot: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CoreTests(unittest.TestCase):
    def test_ma_boundary_and_feature_contract(self) -> None:
        self.assertEqual(ma_class([49.999, 50, 70, 70.001, np.nan]).tolist(), [-1, 0, 0, 1, 9])
        x = np.zeros((4, 4))
        med = np.asarray([[1, 2, 49, 10], [1, 2, 50, 10], [1, 2, 70, 10], [1, 2, 71, 10]], dtype=float)
        self.assertEqual(ma_features(x, med, np.zeros(4), np.ones(4)).shape, (4, 13))

    def test_gate_fit_score_and_fail_closed(self) -> None:
        rng = np.random.default_rng(7)
        features = rng.normal(size=(24, 13))
        pred = np.asarray([-1, 0, 1] * 8)
        # Mixed correct/error rows exercise the fitted branch (the historical
        # guard falls back when all rows are errors or one-sided).
        truth = pred.copy()
        truth[[1, 4, 7, 10]] = np.asarray([1, -1, 0, 1])
        model = fit_ma_error_lr(features, pred, truth)
        self.assertFalse(model["fallback"])
        score = score_ma_error_lr(model, features, pred)
        self.assertEqual(score.shape, (24,))
        self.assertTrue(np.isfinite(score).all())
        bad = pred.copy()
        bad[0] = 9
        self.assertTrue(np.isinf(score_ma_error_lr(model, features, bad)[0]))

    def test_gate_fit_score_matches_ast_exact_snapshot(self) -> None:
        """The importable extraction remains numerically equal to the source."""

        source = RELEASE / "snapshots" / "ridge10" / "run_ridge10_fullchain.py"
        namespace: dict[str, Any] = {
            "np": np,
            "math": math,
            "minimize": minimize,
            "expit": expit,
            "Any": Any,
            "Mapping": Mapping,
            "MA_CLASSES": (-1, 0, 1),
        }
        _ast_functions(
            source,
            {"_preprocess_fit", "_empty_gate_pre", "_fit_ma_gate", "score_ma_gate"},
            namespace,
        )
        rng = np.random.default_rng(19)
        features = rng.normal(size=(18, 13))
        features[2, 4] = np.nan
        features[13, 1] = np.inf
        pred = np.asarray([-1, 0, 1] * 6)
        truth = pred.copy()
        truth[[1, 4, 7, 10]] = np.asarray([1, -1, 0, 1])
        original = namespace["_fit_ma_gate"](features, pred, truth, "mixed", 19)
        extracted = fit_ma_error_lr(features, pred, truth, fit_id="mixed", seed=19)
        self.assertFalse(original["fallback"])
        self.assertFalse(extracted["fallback"])
        for key in ("n", "errors", "correct", "legal_count", "finite_input_rows"):
            self.assertEqual(extracted[key], original[key])
        for key in ("constant_score", "intercept"):
            self.assertAlmostEqual(extracted[key], original[key], places=14)
        for key in ("mean", "sd", "median"):
            np.testing.assert_allclose(extracted["pre"][key], original["pre"][key], rtol=0, atol=1e-14)
        np.testing.assert_allclose(extracted["coef"], original["coef"], rtol=0, atol=1e-12)
        score_features = features.copy()
        score_features[0, 0] = np.nan
        score_pred = pred.copy()
        score_pred[-1] = 9
        expected_score = namespace["score_ma_gate"](original, score_features, score_pred)
        actual_score = score_ma_error_lr(extracted, score_features, score_pred)
        np.testing.assert_allclose(actual_score, expected_score, rtol=0, atol=1e-14, equal_nan=True)

    def test_calibration_matches_ast_exact_snapshot(self) -> None:
        source = RELEASE / "snapshots" / "adaptation" / "run.py"
        namespace: dict[str, Any] = {"np": np}
        _ast_functions(source, {"project", "raw", "fit", "pred"}, namespace)
        model = {
            "xm": np.asarray([0.1, 0.2, 0.3, 0.4]),
            "xs": np.asarray([1.1, 1.2, 1.3, 1.4]),
            "ym": np.asarray([10.0, 20.0, 30.0, 40.0]),
            "ys": np.asarray([2.0, 3.0, 4.0, 5.0]),
            "coef": np.asarray([[1.0, 0.1, 0.0, 0.0], [0.0, 1.0, 0.1, 0.0], [0.0, 0.0, 1.0, 0.1], [0.1, 0.0, 0.0, 1.0]]),
            "intercept": np.asarray([0.0, 0.1, -0.1, 0.2]),
        }
        x = np.asarray([[0.0, 1.0, 2.0, 3.0], [1.0, 1.5, 2.5, 3.5], [2.0, 2.5, 3.5, 4.5], [3.0, 3.5, 4.5, 5.5]])
        original_raw = namespace["raw"](model, x)
        extracted_raw = raw_ridge10(model, x)
        np.testing.assert_allclose(extracted_raw, original_raw, rtol=0, atol=1e-14)
        reference = original_raw + np.asarray([[0.2, -0.1, 0.3, -0.2], [0.1, 0.2, -0.2, 0.1], [-0.1, 0.0, 0.2, 0.0], [0.0, -0.2, 0.1, 0.2]])
        for arm in ("affine", "joint"):
            expected = namespace["fit"](original_raw, reference, model, arm)
            actual = fit_calibration(extracted_raw, reference, model, arm)
            for key in ("matrix", "vbar", "ubar", "mu", "scale"):
                np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-14)
            np.testing.assert_allclose(
                apply_calibration(actual, extracted_raw),
                namespace["pred"](expected, original_raw),
                rtol=0,
                atol=1e-14,
            )

    def test_ridge_and_adaptation_contract(self) -> None:
        model = {"xm": np.zeros(4), "xs": np.ones(4), "ym": np.zeros(4), "ys": np.ones(4), "coef": np.eye(4), "intercept": np.zeros(4)}
        x = np.asarray([[1, 2, 3, 4], [-1, 0, 5, 99]], dtype=float)
        raw = raw_ridge10(model, x)
        self.assertEqual(raw[1, 0], 0)
        self.assertLess(raw[1, 3], 90)
        calibration = fit_calibration(raw, raw + 0.2, model, "affine")
        self.assertEqual(apply_calibration(calibration, raw).shape, raw.shape)
        self.assertTrue(np.isfinite(auxiliary_j(raw)).all())

    def test_first_report_and_actions(self) -> None:
        actions_5 = np.asarray([[True, False], [False, False]])
        actions_10 = np.asarray([[True, False], [False, True]])
        pred_5 = np.zeros((2, 2), dtype=np.int8)
        pred_10 = np.asarray([[1, 0], [0, 1]], dtype=np.int8)
        replay = cumulative_first_report([(300, actions_5, pred_5), (600, actions_10, pred_10)])
        self.assertEqual(replay["first_time"].tolist(), [[300, -1], [-1, 600]])
        self.assertEqual(replay["first_class"].tolist(), [[0, 9], [9, 1]])
        out = metrics(replay["reported"], replay["first_class"], np.zeros(2, dtype=np.int8))
        self.assertEqual(out["released"], 2)
        self.assertEqual(out["correct"], 1)
        selected = class_actions(np.asarray([0, -1, 1]), np.asarray([0.1, 0.8, 0.9]), np.asarray([0.2, 0.8, 0.8]), np.ones(3, bool), {"normal": 0.5, "tail": 0.5, "all_withhold": False})
        self.assertEqual(selected.tolist(), [True, False, False])

    def test_dualhead_support_and_frozen_selector_contracts(self) -> None:
        contract = _import_snapshot(
            RELEASE / "snapshots" / "historical" / "dualhead" / "dual_head_contract.py",
            "uteg_dual_head_contract_snapshot",
        )
        pred_ma = np.asarray([60.0, 60.0, 60.0, np.nan, 60.0, 60.0])
        pred_class = ma_class(pred_ma)
        truth = np.asarray([-1, 0, 1, 0, -1, 1])
        legal = np.ones(6, dtype=bool)
        z, stats = contract.build_features(np.asarray([1.0, np.nan, 3.0, 4.0, 5.0, 6.0]), pred_ma, legal)
        self.assertEqual(z.shape, (6, 2))
        self.assertTrue(np.isfinite(z).all())
        self.assertEqual(set(stats), {"median", "mean", "sd"})
        low_mask, low_y = contract.build_head_mask(truth, pred_class, pred_ma, legal, "low")
        high_mask, high_y = contract.build_head_mask(truth, pred_class, pred_ma, legal, "high")
        self.assertEqual(low_mask.tolist(), [True, True, False, False, True, False])
        self.assertEqual(high_mask.tolist(), [False, True, True, False, False, True])
        self.assertEqual(low_y.tolist(), [1.0, 0.0, 1.0])
        self.assertEqual(high_y.tolist(), [0.0, 1.0, 1.0])
        support = contract.support_summary(np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=float))
        self.assertEqual(support["status"], "SUPPORTED")
        weights = contract.class_balanced_weights(np.asarray([0, 0, 0, 1, 1, 1], dtype=float))
        self.assertAlmostEqual(float(weights.mean()), 1.0)
        self.assertAlmostEqual(float(weights[:3].sum()), float(weights[3:].sum()))
        with self.assertRaises(contract.AuthorizationRequired):
            contract.fit_head_guard()

        selector = _import_snapshot(
            RELEASE / "snapshots" / "historical" / "dualhead" / "run_unified_selector.py",
            "uteg_unified_selector_snapshot",
        )
        result = selector._synthetic_tests()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["common_tail_strict_gt75"], "PASS")
        self.assertEqual(result["fit_units"], 0)


if __name__ == "__main__":
    unittest.main()
