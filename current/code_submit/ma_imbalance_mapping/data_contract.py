"""Read-only data contract for the MA-imbalance mapping experiments.

This module deliberately contains no module-level file reads or writes.  The
training entry point should call :func:`load_development_inputs` once, use the
returned ``data['y']`` (which hides every test label), and keep
:func:`load_test_references` confined to a separate evaluation stage.

The paths below are relative to the project workspace by default.  They point
to the frozen 15-minute inputs and the frozen target partitions used by the
current manuscript.  A caller may supply equivalent paths in ``config`` for a
private copy, but the validation contract remains unchanged.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np


DEFAULT_DATA_PATH = Path(
    "outputs/uteg_legacy_aligned_threecenter_20261022_155614/DATA15.npz"
)
DEFAULT_PARTITION_ROOT = Path(
    "outputs/uteg_signal_model_goal_20261026_000929/"
    "ridge10_fullchain_20261026_122551/models"
)
DEFAULT_SEEDS = tuple(range(1000, 1020))
DEFAULT_DIRECTIONS = ("HZ_to_SA", "HZ+HB_to_SA")
DEFAULT_SOURCE_COUNTS = {"HZ_to_SA": 95, "HZ+HB_to_SA": 174}
DEFAULT_SOURCE_CENTERS = {
    "HZ_to_SA": frozenset(("Huzhou",)),
    "HZ+HB_to_SA": frozenset(("Huzhou", "Hubei")),
}


class DataContractError(ValueError):
    """Raised when a frozen input or partition violates the experiment contract."""


def sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of one file without changing filesystem state."""

    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_path(workspace: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(workspace) / path


def _as_1d_int(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise DataContractError(f"{name} must be one-dimensional")
    if not np.issubdtype(array.dtype, np.integer):
        try:
            cast = array.astype(np.int64)
        except (TypeError, ValueError) as exc:
            raise DataContractError(f"{name} must contain integer indices") from exc
        if not np.array_equal(array, cast):
            raise DataContractError(f"{name} contains non-integer indices")
        array = cast
    return np.asarray(array, dtype=np.int64).copy()


def _check_indices(values: np.ndarray, n: int, name: str) -> None:
    if len(np.unique(values)) != len(values):
        raise DataContractError(f"{name} contains duplicate indices")
    if values.size and (int(values.min()) < 0 or int(values.max()) >= n):
        raise DataContractError(f"{name} contains an index outside [0, {n})")


def _check_group_disjoint(
    groups: np.ndarray, left: np.ndarray, right: np.ndarray, name: str
) -> None:
    if set(groups[left].astype(str).tolist()) & set(groups[right].astype(str).tolist()):
        raise DataContractError(f"{name} contains identity-proxy group overlap")


def _partition_path(root: Path, direction: str, seed: int) -> Path:
    return root / direction / "target15" / f"SAHZU_CFULL_seed{int(seed)}.joblib"


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        return {str(key): np.asarray(archive[key]).copy() for key in archive.files}


def _load_partition(path: Path) -> Mapping[str, Any]:
    """Load one explicitly requested frozen joblib partition.

    The repository's frozen model objects are trusted local inputs.  This
    function does not import or execute any historical runner.
    """

    if not path.exists():
        raise FileNotFoundError(path)
    object_value = joblib.load(path)
    if not isinstance(object_value, Mapping):
        raise DataContractError(f"partition artifact is not a mapping: {path}")
    required = {"pool", "test", "positions"}
    missing = required - set(object_value)
    if missing:
        raise DataContractError(f"partition artifact missing {sorted(missing)}: {path}")
    return object_value


def _canonical_indices(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def _validate_outer_position(
    position: Mapping[str, Any],
    *,
    position_number: int,
    pool: np.ndarray,
    groups: np.ndarray,
    n_records: int,
) -> dict[str, Any]:
    if "A" not in position or "B" not in position:
        raise DataContractError(f"position {position_number} lacks A/B")
    A = _as_1d_int(position["A"], f"position[{position_number}].A")
    B = _as_1d_int(position["B"], f"position[{position_number}].B")
    _check_indices(A, n_records, f"position[{position_number}].A")
    _check_indices(B, n_records, f"position[{position_number}].B")
    pool_set = set(pool.tolist())
    if not set(A.tolist()) <= pool_set or not set(B.tolist()) <= pool_set:
        raise DataContractError(f"position {position_number} leaves the target pool")
    if set(A.tolist()) & set(B.tolist()):
        raise DataContractError(f"position {position_number} has A/B overlap")
    if set(A.tolist()) | set(B.tolist()) != pool_set:
        raise DataContractError(f"position {position_number} does not partition the pool")
    _check_group_disjoint(groups, A, B, f"position {position_number} A/B")

    inner_rows: list[dict[str, Any]] = []
    inner = position.get("inner")
    if not isinstance(inner, Sequence) or isinstance(inner, (str, bytes)):
        raise DataContractError(f"position {position_number} lacks inner folds")
    if len(inner) != 3:
        raise DataContractError(f"position {position_number} must have three inner folds")
    A_set = set(A.tolist())
    for fold_number, fold in enumerate(inner):
        if not isinstance(fold, Mapping) or "T" not in fold or "V" not in fold:
            raise DataContractError(
                f"position {position_number} inner fold {fold_number} lacks T/V"
            )
        T = _as_1d_int(fold["T"], f"position[{position_number}].inner[{fold_number}].T")
        V = _as_1d_int(fold["V"], f"position[{position_number}].inner[{fold_number}].V")
        _check_indices(T, n_records, f"position[{position_number}].inner[{fold_number}].T")
        _check_indices(V, n_records, f"position[{position_number}].inner[{fold_number}].V")
        if not set(T.tolist()) <= A_set or not set(V.tolist()) <= A_set:
            raise DataContractError(f"position {position_number} inner fold leaves A")
        if set(T.tolist()) & set(V.tolist()):
            raise DataContractError(f"position {position_number} inner T/V overlap")
        if set(T.tolist()) | set(V.tolist()) != A_set:
            raise DataContractError(f"position {position_number} inner fold does not cover A")
        _check_group_disjoint(
            groups, T, V, f"position {position_number} inner fold {fold_number}"
        )
        inner_rows.append({"T": T, "V": V})
    return {"A": A, "B": B, "inner": inner_rows}


def _validate_direction_partitions(
    object_value: Mapping[str, Any],
    *,
    direction: str,
    seed: int,
    n_records: int,
    groups: np.ndarray,
) -> dict[str, Any]:
    pool = _as_1d_int(object_value["pool"], f"{direction} seed {seed} pool")
    test = _as_1d_int(object_value["test"], f"{direction} seed {seed} test")
    if len(pool) != 60 or len(test) != 201:
        raise DataContractError(
            f"{direction} seed {seed} must contain pool=60 and test=201"
        )
    _check_indices(pool, n_records, f"{direction} seed {seed} pool")
    _check_indices(test, n_records, f"{direction} seed {seed} test")
    if set(pool.tolist()) & set(test.tolist()):
        raise DataContractError(f"{direction} seed {seed} pool/test overlap")
    _check_group_disjoint(groups, pool, test, f"{direction} seed {seed} pool/test")

    raw_positions = object_value["positions"]
    if isinstance(raw_positions, Mapping):
        positions = [raw_positions[key] for key in sorted(raw_positions)[:3]]
    elif isinstance(raw_positions, Sequence) and not isinstance(raw_positions, (str, bytes)):
        positions = list(raw_positions[:3])
    else:
        raise DataContractError(f"{direction} seed {seed} positions are invalid")
    if len(positions) != 3:
        raise DataContractError(f"{direction} seed {seed} needs three outer positions")
    folds = [
        _validate_outer_position(
            position,
            position_number=index,
            pool=pool,
            groups=groups,
            n_records=n_records,
        )
        for index, position in enumerate(positions)
    ]
    B_all = np.concatenate([fold["B"] for fold in folds])
    if len(B_all) != len(pool) or set(B_all.tolist()) != set(pool.tolist()):
        raise DataContractError(f"{direction} seed {seed} outer B folds do not cover pool")
    if len(np.unique(B_all)) != len(B_all):
        raise DataContractError(f"{direction} seed {seed} outer B folds overlap")
    return {"pool": pool, "test": test, "folds": folds}


def class_support(
    y: np.ndarray,
    indices: Iterable[int] | None = None,
    *,
    low_boundary: float = 50.0,
    high_boundary: float = 70.0,
) -> dict[str, int]:
    """Count finite low/within/high MA records in a training subset.

    This helper only returns aggregate support counts; it never returns
    patient-level rows.  The caller should pass training indices, never test
    indices.
    """

    values = np.asarray(y, dtype=float)
    if values.ndim != 2 or values.shape[1] < 3:
        raise DataContractError("y must have at least three output columns")
    selected = np.arange(len(values), dtype=np.int64) if indices is None else _as_1d_int(indices, "indices")
    _check_indices(selected, len(values), "class-support indices")
    ma = values[selected, 2]
    finite = np.isfinite(ma)
    return {
        "low": int(np.sum(finite & (ma < low_boundary))),
        "within": int(np.sum(finite & (ma >= low_boundary) & (ma <= high_boundary))),
        "high": int(np.sum(finite & (ma > high_boundary))),
        "finite": int(np.sum(finite)),
    }


def _mask_nontraining_labels(medical: np.ndarray, allowed: np.ndarray,
                             test: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Expose labels only for declared source/development rows, never tests."""
    _check_indices(allowed, len(medical), "allowed training rows")
    _check_indices(test, len(medical), "test union")
    if np.intersect1d(allowed, test).size:
        raise DataContractError("training/test row overlap")
    _check_group_disjoint(groups, allowed, test, "all training/test groups")
    if not np.isfinite(medical[allowed]).all():
        raise DataContractError("non-finite source/development reference")
    masked = np.full_like(medical, np.nan, dtype=float)
    masked[allowed] = medical[allowed]
    return masked


def load_development_inputs(
    workspace: str | Path,
    config: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]], dict[str, str]]:
    """Load frozen inputs and hide all test reference labels.

    The returned ``data['y']`` exposes only declared source/development rows;
    every other row, including the test union, is replaced by NaN.  This
    is an explicit training-interface guard: the private input file is read
    locally, but downstream fitting code cannot accidentally consume test
    labels from the returned object.
    """

    workspace_path = Path(workspace).resolve()
    cfg = dict(config or {})
    data_path = _resolve_path(workspace_path, cfg.get("data_path", DEFAULT_DATA_PATH))
    partition_root = _resolve_path(
        workspace_path, cfg.get("partition_root", DEFAULT_PARTITION_ROOT)
    )
    seeds = tuple(int(seed) for seed in cfg.get("seeds", DEFAULT_SEEDS))
    directions = tuple(str(direction) for direction in cfg.get("directions", DEFAULT_DIRECTIONS))
    if not seeds or len(set(seeds)) != len(seeds):
        raise DataContractError("seeds must be a non-empty unique sequence")
    if not directions or len(set(directions)) != len(directions):
        raise DataContractError("directions must be a non-empty unique sequence")

    archive = _load_npz(data_path)
    required = {"x", "medical", "ids", "groups", "centers", "input_qc"}
    missing = required - set(archive)
    if missing:
        raise DataContractError(f"DATA15 is missing keys: {sorted(missing)}")
    x = np.asarray(archive["x"], dtype=float)
    medical = np.asarray(archive["medical"], dtype=float)
    if x.ndim != 2 or x.shape[1] != 4 or medical.ndim != 2 or medical.shape[1] != 4:
        raise DataContractError("x and medical must both have shape (n, 4)")
    n_records = len(x)
    if len(medical) != n_records:
        raise DataContractError("x and medical record counts differ")
    ids = np.asarray(archive["ids"]).copy()
    groups = np.asarray(archive["groups"]).copy()
    centers = np.asarray(archive["centers"]).astype(str, copy=True)
    input_qc = np.asarray(archive["input_qc"], dtype=bool).copy()
    if any(len(array) != n_records for array in (ids, groups, centers, input_qc)):
        raise DataContractError("metadata lengths do not match x")

    partitions: dict[int, dict[str, Any]] = {}
    data_digest = sha256(data_path)
    expected_data_digest = cfg.get("expected_data_sha256")
    if expected_data_digest is not None:
        if isinstance(expected_data_digest, Mapping):
            expected_data_digest = expected_data_digest.get("data")
        if expected_data_digest is None:
            raise DataContractError(
                "expected_data_sha256 mapping must provide a 'data' digest"
            )
        if str(expected_data_digest).lower() != data_digest:
            raise DataContractError(
                "DATA15 SHA-256 does not match expected_data_sha256"
            )
    hashes: dict[str, str] = {str(data_path.resolve()): data_digest}
    all_test_indices: list[np.ndarray] = []
    for seed in seeds:
        direction_parts: dict[str, dict[str, Any]] = {}
        for direction in directions:
            path = _partition_path(partition_root, direction, seed)
            hashes[str(path.resolve())] = sha256(path)
            artifact = _load_partition(path)
            direction_parts[direction] = _validate_direction_partitions(
                artifact,
                direction=direction,
                seed=seed,
                n_records=n_records,
                groups=groups,
            )
        anchor_direction = directions[0]
        anchor = direction_parts[anchor_direction]
        for direction in directions[1:]:
            candidate = direction_parts[direction]
            if not np.array_equal(anchor["pool"], candidate["pool"]):
                raise DataContractError(f"seed {seed}: source directions disagree on pool")
            if not np.array_equal(anchor["test"], candidate["test"]):
                raise DataContractError(f"seed {seed}: source directions disagree on test")
            for position, (left, right) in enumerate(zip(anchor["folds"], candidate["folds"])):
                if not np.array_equal(left["A"], right["A"]) or not np.array_equal(left["B"], right["B"]):
                    raise DataContractError(f"seed {seed}: source directions disagree at fold {position}")
                for inner_index, (lf, rf) in enumerate(zip(left["inner"], right["inner"])):
                    if not np.array_equal(lf["T"], rf["T"]) or not np.array_equal(lf["V"], rf["V"]):
                        raise DataContractError(
                            f"seed {seed}: source directions disagree at fold {position}/inner {inner_index}"
                        )
        partitions[seed] = anchor
        all_test_indices.append(anchor["test"])

        target_rows = np.concatenate((anchor["pool"], anchor["test"]))
        if not np.all(centers[target_rows] == "SAHZU"):
            raise DataContractError(f"seed {seed}: target rows must belong to SAHZU")
        if seed != seeds[0]:
            first = partitions[seeds[0]]
            if not np.array_equal(anchor["test"], first["test"]) or not np.array_equal(anchor["pool"], first["pool"]):
                raise DataContractError("all seeds must share the same ordered pool/test cohorts")

    test_union = np.unique(np.concatenate(all_test_indices))
    for seed, partition in partitions.items():
        pool = partition["pool"]
        if not np.isfinite(x[pool]).all() or not np.isfinite(medical[pool]).all():
            raise DataContractError(f"seed {seed}: non-finite training input or label")
        if np.intersect1d(pool, test_union).size:
            raise DataContractError(
                f"seed {seed}: test-union masking would hide a pool label"
            )

    source_rows: dict[str, np.ndarray] = {}
    for direction in directions:
        allowed_centers = DEFAULT_SOURCE_CENTERS.get(direction)
        if allowed_centers is None:
            raise DataContractError(f"no frozen source-center rule for {direction}")
        # The frozen source rule is asymmetric: all Huzhou rows are eligible,
        # whereas Hubei rows additionally require input_qc.  Do not silently
        # drop non-finite eligible rows; report them as a contract violation.
        if direction == "HZ_to_SA":
            center_mask = centers == "Huzhou"
        else:
            center_mask = (centers == "Huzhou") | (
                (centers == "Hubei") & input_qc
            )
        rows = np.flatnonzero(center_mask).astype(np.int64)
        if rows.size and (
            not np.isfinite(x[rows]).all() or not np.isfinite(medical[rows]).all()
        ):
            raise DataContractError(
                f"{direction}: eligible source rows contain non-finite values"
            )
        expected = DEFAULT_SOURCE_COUNTS.get(direction)
        if expected is not None and len(rows) != expected:
            raise DataContractError(f"{direction}: expected {expected} source rows, found {len(rows)}")
        for seed, partition in partitions.items():
            if set(rows.tolist()) & set(partition["pool"].tolist()):
                raise DataContractError(f"{direction} source rows overlap seed {seed} pool")
            if set(rows.tolist()) & set(partition["test"].tolist()):
                raise DataContractError(f"{direction} source rows overlap seed {seed} test")
            _check_group_disjoint(groups, rows, partition["pool"], f"{direction} source/pool")
            _check_group_disjoint(groups, rows, partition["test"], f"{direction} source/test")
            source_centers = set(centers[rows].tolist())
            target_centers = set(
                centers[np.concatenate((partition["pool"], partition["test"]))].tolist()
            )
            if source_centers & target_centers:
                raise DataContractError(
                    f"{direction} source and target centers overlap at seed {seed}"
                )
        source_rows[direction] = rows

    allowed_training = np.unique(np.concatenate(
        list(source_rows.values()) + [part["pool"] for part in partitions.values()]
    ))
    y = _mask_nontraining_labels(medical, allowed_training, test_union, groups)

    data = {
        "x": x,
        "y": y,
        "ids": ids,
        "groups": groups,
        "centers": centers,
        "input_qc": input_qc,
        "source_rows": source_rows,
        "allowed_training_indices": allowed_training,
    }
    return data, partitions, hashes


def load_test_references(
    workspace: str | Path,
    config: Mapping[str, Any] | None,
    indices: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Load raw reference labels for an explicit evaluation-only index set.

    Training code should never call this function.  It is intentionally kept
    separate from :func:`load_development_inputs`, whose returned labels are
    masked at every configured test index.
    """

    cfg = dict(config or {})
    path = _resolve_path(Path(workspace).resolve(), cfg.get("data_path", DEFAULT_DATA_PATH))
    archive = _load_npz(path)
    if "medical" not in archive:
        raise DataContractError("DATA15 is missing medical references")
    medical = np.asarray(archive["medical"], dtype=float)
    selected = _as_1d_int(indices, "evaluation indices")
    _check_indices(selected, len(medical), "evaluation indices")
    return medical[selected].copy()

