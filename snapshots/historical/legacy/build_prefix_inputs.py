"""Build and fit the real 5/10-min U-TEG prefixes for all 480 records.

This is a narrow signal-fitting stage only.  It does not train a mapper,
calibrator, or report gate.  The 15-min Hubei fits are reused from the saved
HUBEI_SIGNAL_FIT_RESULTS.json and are not run again.
"""
from pathlib import Path
import ast
import hashlib
import json
import os
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent
CACHE = ROOT / "cache" / "prefix_signal"
CACHE.mkdir(parents=True, exist_ok=True)
for name in (
    "TEMP", "TMP", "TMPDIR", "PYTHONPYCACHEPREFIX", "JOBLIB_TEMP_FOLDER",
    "NUMBA_CACHE_DIR", "TORCH_HOME", "MPLCONFIGDIR", "XDG_CACHE_HOME",
):
    os.environ[name] = str(CACHE / name)
    Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402
from scipy.special import expit  # noqa: E402


RAW = OUT / "uteg_three_center_validation_20261021_174409" / "private" / "INPUT_RAW_AND_REFERENCE.npz"
DATA15 = ROOT / "DATA15.npz"
CANDIDATE = OUT / "uteg_hubei_qc_repair_20261021_171418" / "CANDIDATE_SIGNALS.npz"
PLATFORM_PROCESS = OUT / "uteg_platform_filter_pilot_20261006_124636" / "process.py"
HUBEI_PROCESS = OUT / "uteg_hubei_qc_repair_20261021_171418" / "run.py"
FITTER_SOURCE = OUT / "uteg_platform_param_compare_20261006_125428" / "curves.py"
FITS_DIR = ROOT / "prefix_signal_fits"
LEDGER_DIR = ROOT / "ledger"
HORIZONS = (300, 600)


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_signal(value):
    return hashlib.sha256(np.asarray(value, dtype=float).tobytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2), encoding="utf-8")


def extract_function(path, name, namespace):
    source = Path(path).read_text(encoding="utf-8-sig")
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1, (path, name, len(nodes))
    module = ast.Module(body=[nodes[0]], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


def exp_shape(q, t, family):
    """Frozen generalized-exponential/Hill callback used by the old F4 fitter."""
    t0, amplitude, k, nu = np.asarray(q, dtype=float)
    z = np.maximum(np.asarray(t, dtype=float) - t0, 0.0)
    v = np.zeros(len(z), dtype=float)
    good = z > 0
    if np.any(good):
        power = nu * (np.log(z[good]) - np.log(k))
        if family == "HILL":
            v[good] = amplitude * expit(power)
        else:
            v[good] = amplitude * (-np.expm1(-np.exp(np.minimum(power, 700.0))))
    return v - np.median(v[:10])


def load_fit_simple():
    ns = dict(np=np, time=time, least_squares=least_squares, expit=expit, shape=exp_shape)
    extract_function(FITTER_SOURCE, "fit_simple", ns)
    return ns["fit_simple"]


def load_preprocessors():
    # The two functions are extracted from their historical files, without
    # importing either runner or allowing them to write to their old outputs.
    sa_ns = dict(np=np, savgol_filter=savgol_filter)
    extract_function(PLATFORM_PROCESS, "apply", sa_ns)
    hb_ns = dict(np=np, savgol_filter=savgol_filter)
    extract_function(HUBEI_PROCESS, "preprocess", hb_ns)
    return sa_ns["apply"], hb_ns["preprocess"]


def metric(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    d = pred - y
    sse = float(d @ d)
    sst = float(np.square(y - y.mean()).sum())
    return {
        "n": int(len(y)),
        "MAE": float(np.mean(np.abs(d))),
        "RMSE": float(np.sqrt(sse / max(len(y), 1))),
        "R2": float(1.0 - sse / max(sst, 1e-30)),
    }


def guard():
    size = sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file())
    assert size < 2 * 2**30
    assert shutil.disk_usage(ROOT).free > 6 * 2**30
    return {"bytes": size, "free": shutil.disk_usage(ROOT).free}


def ledger_entry(horizon, index, center, source, input_hash, status):
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    path = LEDGER_DIR / f"signal_{horizon}_{index:03d}.json"
    value = dict(
        kind="signal",
        units=1,
        horizon=int(horizon),
        tube=str(center),
        oldsource=str(source),
        input_hash=str(input_hash),
        status=str(status),
    )
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        assert old == clean(value), (path, old, value)
    else:
        write_json(path, value)


def source_segment(raw_signal, mask, center, horizon, sa_apply, hb_preprocess):
    n_raw = int(np.asarray(mask, dtype=bool).sum())
    n = min(int(horizon), n_raw)
    assert n >= 11, (center, horizon, n_raw)
    y = np.asarray(raw_signal[:n], dtype=float)
    assert np.isfinite(y).all()
    if center == "SAHZU":
        processed, info = sa_apply(y, 0.05)
        mode = "SAHZU_platform_apply_q0.05_on_prefix"
        source = PLATFORM_PROCESS
    elif center == "Huzhou":
        # The Huzhou signal in the frozen raw bridge is the device-exported
        # value already used by the old Huzhou pathway.  Do not re-filter it.
        processed = y.copy()
        info = {
            "mode": "device_export_original_no_extra_filter",
            "n": n,
            "horizon": int(horizon),
        }
        mode = "Huzhou_device_export_original_no_extra_filter"
        source = RAW
    elif center == "Hubei":
        processed, _, info = hb_preprocess(y)
        mode = "Hubei_doctor_approved_preprocess_p5_on_prefix"
        source = HUBEI_PROCESS
    else:
        raise ValueError(center)
    processed = np.asarray(processed, dtype=float)
    assert processed.shape == (n,) and np.isfinite(processed).all()
    return processed, n_raw, n, mode, source, info


def make_inputs():
    raw = dict(np.load(RAW, allow_pickle=False))
    data = dict(np.load(DATA15, allow_pickle=False))
    candidate = dict(np.load(CANDIDATE, allow_pickle=False))
    assert np.array_equal(raw["ids"], data["ids"])
    assert np.array_equal(raw["center"], data["centers"])
    assert raw["original"].shape == (480, 900)
    assert raw["observed"].shape == (480, 900)
    # The candidate archive uses unprefixed HB identifiers; its signal order
    # must still match the prefixed IDs in the three-center bridge.
    assert np.array_equal(
        np.char.add("Hubei_", candidate["ids"].astype(str)),
        raw["ids"][399:].astype(str),
    )
    assert np.array_equal(candidate["observed_mask"], raw["observed"][399:])
    sa_apply, hb_preprocess = load_preprocessors()

    processed = np.full((2, 480, 600), np.nan, dtype=float)
    observed = np.zeros((2, 480, 600), dtype=bool)
    observed_n = np.zeros((2, 480), dtype=np.int32)
    raw_n = np.asarray(raw["observed"], dtype=bool).sum(axis=1).astype(np.int32)
    modes = np.empty((2, 480), dtype=object)
    source_sha = np.empty((2, 480), dtype=object)
    info_rows = []
    for hi, horizon in enumerate(HORIZONS):
        for i in range(480):
            center = str(data["centers"][i])
            y, n_raw, n, mode, source, info = source_segment(
                raw["original"][i], raw["observed"][i], center, horizon, sa_apply, hb_preprocess
            )
            processed[hi, i, :n] = y
            observed[hi, i, :n] = True
            # Storage padding is explicitly not marked observed and is never
            # passed to fit_simple.  It only gives consumers a fixed window.
            if n < horizon:
                processed[hi, i, n:horizon] = y[-1]
            observed_n[hi, i] = n
            modes[hi, i] = mode
            source_sha[hi, i] = sha_file(source)
            info_rows.append(
                dict(
                    horizon=int(horizon),
                    index=int(i),
                    record_id=str(data["ids"][i]),
                    center=center,
                    n_raw_observed=int(n_raw),
                    n_observed=int(n),
                    mode=mode,
                    preprocessing_source=str(source),
                    preprocessing_source_sha256=sha_file(source),
                    preprocessing_info=info,
                    input_hash=sha_signal(y),
                )
            )

    np.savez_compressed(
        ROOT / "PREFIX_INPUTS.npz",
        horizon_seconds=np.asarray(HORIZONS, dtype=np.int32),
        ids=np.asarray(data["ids"]),
        centers=np.asarray(data["centers"]),
        groups=np.asarray(data["groups"]),
        raw_observed_n=np.broadcast_to(raw_n, (2, 480)).copy(),
        observed_n=observed_n,
        observed=observed,
        processed=processed,
        preprocessing_mode=modes.astype(str),
        preprocessing_source_sha256=source_sha.astype(str),
    )
    return info_rows, raw, data


def write_freeze(info_rows):
    planned = len(HORIZONS) * 480
    assert len(info_rows) == planned
    freeze = dict(
        version="three_center_prefix_EXP_signal_v1",
        fit_family="generalized_exponential_F4",
        horizons_seconds=list(HORIZONS),
        records=480,
        planned_positions=planned,
        fit_uses_true_observed_segment_only=True,
        padding_storage_only=True,
        five_start_fitter=True,
        fitter_source=str(FITTER_SOURCE),
        fitter_source_sha256=sha_file(FITTER_SOURCE),
        preprocessing_sources={
            "SAHZU": str(PLATFORM_PROCESS),
            "Huzhou": "frozen INPUT_RAW_AND_REFERENCE.npz device-exported original; no extra filter",
            "Hubei": str(HUBEI_PROCESS),
        },
        preprocessing_source_sha256={
            "SAHZU": sha_file(PLATFORM_PROCESS),
            "Huzhou": sha_file(RAW),
            "Hubei": sha_file(HUBEI_PROCESS),
        },
        input_sha256=sha_file(ROOT / "PREFIX_INPUTS.npz"),
        raw_sha256=sha_file(RAW),
        data15_sha256=sha_file(DATA15),
        candidate_sha256=sha_file(CANDIDATE),
        new_mapper_calibrator_gate_fits=0,
        new_15min_signal_fits=0,
        new_test_evaluated=False,
        created_before_first_prefix_fit=True,
    )
    write_json(ROOT / "PREFIX_SIGNAL_FREEZE.json", freeze)
    return freeze


def fit_prefixes(info_rows, freeze):
    fit_simple = load_fit_simple()
    fits = []
    start_time = time.time()
    fresh = 0
    reused = 0
    signal_array = np.load(ROOT / "PREFIX_INPUTS.npz", allow_pickle=False)
    for j, info in enumerate(info_rows):
        horizon = int(info["horizon"])
        i = int(info["index"])
        n = int(info["n_observed"])
        y = np.asarray(signal_array["processed"][HORIZONS.index(horizon), i, :n], dtype=float)
        assert len(y) == n and np.isfinite(y).all()
        path = FITS_DIR / f"EXP_{horizon}_{i:03d}.joblib"
        fit = None
        status = "FIT_RETURNED"
        cache_reused = False
        if path.exists():
            candidate = joblib.load(path)
            valid = (
                candidate.get("horizon") == horizon
                and candidate.get("index") == i
                and candidate.get("record_id") == info["record_id"]
                and candidate.get("n_observed") == n
                and candidate.get("signal_hash") == info["input_hash"]
                and candidate.get("family") == "EXP"
            )
            if valid:
                fit = candidate
                cache_reused = True
                reused += 1
        if fit is None:
            try:
                t = np.arange(1, n + 1, dtype=float)
                z = fit_simple(y, "F4")
                q = np.asarray(z["q"], dtype=float)
                pred = exp_shape(q, t, "EXP")
                z = dict(z)
                z.update(
                    horizon=horizon,
                    index=i,
                    record_id=info["record_id"],
                    center=info["center"],
                    family="EXP",
                    n_raw_observed=int(info["n_raw_observed"]),
                    n_observed=n,
                    signal_hash=info["input_hash"],
                    preprocessing_mode=info["mode"],
                    preprocessing_source=info["preprocessing_source"],
                    processed_observed=metric(y, pred),
                    padding_points=int(horizon - n),
                    cache_reused=False,
                )
                fit = z
                path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(fit, path, compress=3)
                fresh += 1
            except Exception as exc:
                fit = dict(
                    horizon=horizon,
                    index=i,
                    record_id=info["record_id"],
                    center=info["center"],
                    family="EXP",
                    n_raw_observed=int(info["n_raw_observed"]),
                    n_observed=n,
                    signal_hash=info["input_hash"],
                    preprocessing_mode=info["mode"],
                    preprocessing_source=info["preprocessing_source"],
                    status="FIT_EXCEPTION",
                    error=repr(exc),
                    cache_reused=False,
                )
                status = "FIT_EXCEPTION"
                path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(fit, path, compress=3)
                fresh += 1
        if cache_reused:
            fit["cache_reused"] = True
        fit.setdefault("status", status)
        # Ledger status describes the fit result, not whether this invocation
        # reused its cache; this keeps reruns hash-stable.
        ledger_entry(horizon, i, info["center"], info["preprocessing_source"], info["input_hash"], status)
        fits.append(fit)
        if j == 0:
            # This is the first formal position timing record; it is not a
            # pilot replacement and the same object is reused if rerun.
            write_json(
                ROOT / "PREFIX_SIGNAL_PILOT.json",
                dict(
                    horizon=horizon,
                    index=i,
                    seconds=time.time() - start_time,
                    cache_reused=cache_reused,
                    formal_position_reused=True,
                    fit_calls_executed=0 if cache_reused else 1,
                ),
            )
        if (j + 1) % 50 == 0 or j == len(info_rows) - 1:
            print(f"PREFIX_SIGNAL {j + 1}/{len(info_rows)} fresh={fresh} reused={reused}", flush=True)
        guard()

    summary = dict(
        version="three_center_prefix_EXP_signal_v1",
        records=480,
        horizons_seconds=list(HORIZONS),
        positions= len(fits),
        fresh_fit_positions=fresh,
        cache_reused_positions=reused,
        status_counts={s: sum(z.get("status") == s for z in fits) for s in sorted(set(z.get("status") for z in fits))},
        fit_family="generalized_exponential_F4",
        all_fit_inputs_true_observed_only=True,
        no_15min_fit_rerun=True,
        input_sha256=freeze["input_sha256"],
        freeze_sha256=sha_file(ROOT / "PREFIX_SIGNAL_FREEZE.json"),
        ledger_positions=len(list(LEDGER_DIR.glob("signal_*.json"))),
        elapsed_seconds=time.time() - start_time,
        fits=fits,
    )
    write_json(ROOT / "PREFIX_SIGNAL_FITS.json", summary)
    return summary


def main():
    info_rows, raw, data = make_inputs()
    freeze = write_freeze(info_rows)
    # The freeze must exist before the first call to fit_simple.
    summary = fit_prefixes(info_rows, freeze)
    assert summary["positions"] == 960
    assert summary["no_15min_fit_rerun"]
    print(
        json.dumps(
            {
                "positions": summary["positions"],
                "fresh_fit_positions": summary["fresh_fit_positions"],
                "cache_reused_positions": summary["cache_reused_positions"],
                "status_counts": summary["status_counts"],
                "elapsed_seconds": summary["elapsed_seconds"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
