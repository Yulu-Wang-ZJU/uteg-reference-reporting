"""Source-derived utilities for the U-TEG reference-reporting release.

The package contains only small, audited helpers extracted from the exact
manuscript-era snapshots under ``snapshots/``.  It deliberately does not
ship private records, trained objects, or generated result tables.
"""

from .core import (
    AUXILIARY_J_COEFFICIENTS,
    MA_CLASSES,
    apply_calibration,
    auxiliary_j,
    class_actions,
    cumulative_first_report,
    fit_calibration,
    fit_ma_error_lr,
    ma_class,
    ma_features,
    metrics,
    project_readouts,
    raw_ridge10,
    score_ma_error_lr,
)

__all__ = [
    "AUXILIARY_J_COEFFICIENTS",
    "MA_CLASSES",
    "apply_calibration",
    "auxiliary_j",
    "class_actions",
    "cumulative_first_report",
    "fit_calibration",
    "fit_ma_error_lr",
    "ma_class",
    "ma_features",
    "metrics",
    "project_readouts",
    "raw_ridge10",
    "score_ma_error_lr",
]
