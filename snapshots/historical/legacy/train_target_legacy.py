"""Three-centre target-side fits for the legacy-aligned 15-min protocol.

This file is intentionally a mechanical wrapper around the 2026-10-12
downstream implementation.  It does not add a candidate model, read test
labels, or evaluate the external-test rows.  ``run_one`` is the stable
interface used by the later evaluation worker.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import runtime as r

# Import numerical libraries only after the task-local runtime has set all
# E-drive cache/TMP variables and thread limits.
import joblib
import numpy as np


TARGETS = ("Huzhou", "SAHZU", "Hubei")
SEEDS = tuple(range(1000, 1020))
OLD_MODELS = r.ROOT.parent / "uteg_hill_exp_matched_20261012_153337" / "models.py"

# Loading through the task-local runtime is important: the old models module
# imports ``runtime`` and must see this run's cache/ledger paths.
m = r.load("legacy_aligned_models_target15", OLD_MODELS)


def _budget_label(budget: str | int) -> str:
    b = str(budget).upper().replace("C", "")
    if b in {"FULL", "40", "60"}:
        return "FULL"
    if b in {"20", "C20"}:
        return "20"
    raise ValueError(f"unsupported target budget: {budget!r}")


def _source_file(target: str, seed: int, source_path: str | Path | None) -> Path:
    if source_path is None:
        p = r.ROOT / "source" / f"{target}_seed{seed}.joblib"
    else:
        q = Path(source_path)
        p = q / f"{target}_seed{seed}.joblib" if q.is_dir() else q
    if not p.exists():
        raise FileNotFoundError(f"missing frozen source model: {p}")
    return p


def _load_data(data_path: str | Path | None, mode: str = "T900") -> dict:
    # Prefix fits reuse the frozen medical/CI labels and roles from DATA15;
    # only the already-fitted generalized-exponential U-TEG representation is
    # replaced by the requested true-observed prefix.
    if mode != "T900" and (data_path is None or Path(data_path).name == "PREFIX_X.npz"):
        p = r.ROOT / "PREFIX_X.npz" if data_path is None else Path(data_path)
        z = np.load(p, allow_pickle=True)
        horizon = int(mode[1:])
        hs = [int(v) for v in np.asarray(z["horizon_seconds"]).tolist()]
        if horizon not in hs:
            raise ValueError(f"prefix horizon missing from {p}: {horizon}")
        base = np.load(r.ROOT / "DATA15.npz", allow_pickle=True)
        out = {k: base[k] for k in base.files}
        out["x"] = np.asarray(z["x"])[hs.index(horizon)]
        out["prefix_horizon_seconds"] = np.asarray(horizon, dtype=np.int32)
        return out
    p = r.ROOT / "DATA15.npz" if data_path is None else Path(data_path)
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _load_role() -> dict:
    return r.read(r.ROOT / "ROLE_FREEZE.json")


def _spec(role: dict, target: str, seed: int, budget: str) -> dict:
    z = role["roles"][target]
    arm = "FULL" if _budget_label(budget) == "FULL" else "C20"
    pool = list(z["pool_full"] if arm == "FULL" else z["c20_by_seed"][str(seed)])
    raw_positions = z["positions_full"][str(seed)] if arm == "FULL" else z["positions_c20"][str(seed)]
    # The frozen C20 manifest retains a code-3 placeholder whose B list is
    # the complement of C20 (it is an external/full-pool bookkeeping list,
    # not an inner validation fold).  Restrict every position to this arm's
    # calibration pool so no out-of-pool row can enter a fit or OOF score.
    poolset = set(pool)
    positions = []
    for p in raw_positions:
        q = dict(p)
        q["A"] = [int(v) for v in p["A"] if int(v) in poolset]
        q["B"] = [int(v) for v in p["B"] if int(v) in poolset]
        q["inner"] = []
        for f in p["inner"]:
            q["inner"].append({
                "T": [int(v) for v in f["T"] if int(v) in poolset],
                "V": [int(v) for v in f["V"] if int(v) in poolset],
            })
        positions.append(q)
    return {
        "target": target,
        "arm": arm,
        "seed": int(seed),
        "d": 1 if target == "Huzhou" else 2,
        "pool": pool,
        "test": list(z["test"]),
        "positions": positions,
        "target_n": int(z["pool_n"]),
        "c20_n": int(z["c20_n"]),
    }


def _fit_error_lr(F: np.ndarray, yh: np.ndarray, yt: np.ndarray):
    ok = (yh != 9) & (yt != 9)
    return m.rm.fit_model(
        dict(id="ERROR-LR", family="ERROR-LR", regularizer=.1),
        F[ok], yh[ok], yt[ok], 0,
    )


def _lr_score(mod, F: np.ndarray, yh: np.ndarray) -> np.ndarray:
    score = np.full(len(yh), np.inf)
    ok = yh != 9
    if ok.any():
        score[ok] = m.rm.predict_model(mod, F[ok], yh[ok])
    return score


def _freeze_before_fit(role: dict, data_path: str | Path | None,
                       mode: str = "T900", source_path: str | Path | None = None) -> None:
    """Freeze fit inputs/rules before the first target-side fit."""
    if mode == "T900":
        p = r.ROOT / "TARGET_FIT_FREEZE.json"
    else:
        p = r.ROOT / f"PREFIX_TARGET_FIT_FREEZE_{mode[1:]}.json"
    sources = {}
    for target in TARGETS:
        for seed in SEEDS:
            if mode == "T900":
                sf = _source_file(target, seed, None)
            else:
                horizon = int(mode[1:])
                sp = r.ROOT / "prefix_source_v2" if source_path is None else Path(source_path)
                sf = sp / f"{target}_t{horizon}_seed{seed}.joblib"
                if not sf.exists():
                    raise FileNotFoundError(sf)
            sources[f"{target}_seed{seed}"] = r.sha(sf)
    if mode == "T900":
        data_file = r.ROOT / "DATA15.npz" if data_path is None else Path(data_path)
    else:
        data_file = r.ROOT / "PREFIX_X.npz" if data_path is None else Path(data_path)
    z = {
        "version": "legacy_aligned_threecenter_target15_v1" if mode == "T900" else f"legacy_aligned_threecenter_prefix_{mode[1:]}_v1",
        "created": time.time(),
        "data": str(data_file),
        "data_sha256": r.sha(data_file),
        "role_sha256": r.sha(r.ROOT / "ROLE_FREEZE.json"),
        "source_sha256": sources,
        "targets": TARGETS,
        "seeds": list(SEEDS),
        "budgets": ["FULL", "C20"],
        "mode": mode,
        "source_models_frozen": True,
        "target_labels_used": "calibration_pool_and_inner_A_only",
        "external_test_labels_read": False,
        "external_test_evaluated": False,
        "math_identity": "mechanical reuse of 20261012 downstream_15min.py and legacy models.py; frozen selected source config for prefix" if mode != "T900" else "mechanical reuse of 20261012 downstream_15min.py and legacy models.py",
        "d1": {"target": "Huzhou", "calibration": "ordinal90", "adapters": ["H"]},
        "d2": {"targets": ["SAHZU", "Hubei"], "adapters": ["H", "L100", "L10", "K100", "K10"], "gate_names": list(m.gm.NAMES), "gate_threshold": 0.15},
        "rules": {
            "no_new_candidates": True,
            "no_source_refit": True,
            "no_test_label_access": True,
            "test_label_copy_masked": True,
            "no_test_selection": True,
        },
    }
    if p.exists():
        old = r.read(p)
        if old.get("data_sha256") != z["data_sha256"] or old.get("role_sha256") != z["role_sha256"] or old.get("source_sha256") != sources or old.get("mode") != mode:
            raise RuntimeError(f"{p.name} does not match current inputs")
    else:
        r.write(p.relative_to(r.ROOT), z)


def _hide_test_labels(D: dict, test: list[int]) -> dict:
    """Return a fit-only copy with external-test reference labels masked."""
    out = dict(D)
    for key in ("ci", "medical"):
        arr = np.array(D[key], copy=True)
        arr[np.asarray(test, dtype=int)] = np.nan
        out[key] = arr
    out["external_test_labels_masked"] = True
    return out


def run_one(
    target: str,
    seed: int,
    budget: str | int,
    mode: str = "T900",
    source_path: str | Path | None = None,
    data_path: str | Path | None = None,
):
    """Fit one frozen target role and budget.

    ``T900`` uses the frozen 15-min source models.  ``T300`` and ``T600`` use
    the separately frozen selected-config prefix source models and the same
    nested target-side calibration/gate procedure; they do not re-select a
    mapper, gate family, or optimizer budget.
    """
    if target not in TARGETS:
        raise ValueError(target)
    if int(seed) not in SEEDS:
        raise ValueError(seed)
    if mode not in {"T900", "T300", "T600"}:
        raise ValueError(f"unsupported mode: {mode}")
    role = _load_role()
    _freeze_before_fit(role, data_path, mode=mode, source_path=source_path)
    D0 = _load_data(data_path, mode=mode)
    spec = _spec(role, target, int(seed), str(budget))
    arm = spec["arm"]
    label = "FULL" if arm == "FULL" else "20"
    outdir = r.ROOT / "target15" if mode == "T900" else r.ROOT / f"targetprefix_{mode[1:]}"
    outpath = outdir / f"{target}_C{label}_seed{int(seed)}.joblib"
    if outpath.exists():
        return joblib.load(outpath)

    # Preserve the protocol's separation explicitly.  All arrays used below
    # are from the calibration pool/inner folds; no test labels can be read.
    D = _hide_test_labels(D0, spec["test"])
    if mode == "T900":
        source_file = _source_file(target, int(seed), source_path)
    else:
        horizon = int(mode[1:])
        sp = r.ROOT / "prefix_source_v2" if source_path is None else Path(source_path)
        source_file = sp / f"{target}_t{horizon}_seed{int(seed)}.joblib"
        if not source_file.exists():
            raise FileNotFoundError(source_file)
    old = joblib.load(source_file)
    if mode == "T900":
        src = old["families"][old["source_selection"]]["model"]
        source_family = old["source_selection"]
    else:
        src = old["model"]
        source_family = old["source_family"]
    pool = spec["pool"]
    loc = {v: k for k, v in enumerate(pool)}
    raw = m.predict(src, D["x"][pool])
    cache: dict[str, dict] = {}
    positions: dict[int, dict] = {}
    scores: dict[str, dict] = {}
    d = spec["d"]
    As = ["H"] if d == 1 else list(m.ans["A"])
    OOF = {a: np.full((len(pool), 4), np.nan) for a in As}
    scalar_models: dict = {}
    gate_models: dict = {}
    extra: dict = {}
    key = f"target15_{target}_C{label}_seed{seed}" if mode == "T900" else f"targetprefix{mode[1:]}_{target}_C{label}_seed{seed}"
    # A failed pilot may have committed a few ledger entries before its
    # output object was written.  Preserve that audit trail and use an
    # explicit retry namespace rather than overwriting ledger records.
    if any(p.name.startswith(key + "_") for p in (r.ROOT / "ledger").glob("*.json")):
        retry = 1
        while any(p.name.startswith(f"{key}_retry{retry}_") for p in (r.ROOT / "ledger").glob("*.json")):
            retry += 1
        key += f"_retry{retry}"
    started = time.time()

    def get(T):
        T = list(map(int, T))
        token = ",".join(map(str, sorted(T)))
        if token not in cache:
            ix = [loc[v] for v in T]
            rr = raw[ix]
            hkey = hashlib.sha256(token.encode()).hexdigest()[:12]
            r.ledger(f"{key}_H_{hkey}", "Huber", train=T)
            cal = m.fit_cal(rr, D["medical"][T], src["ym"], src["ys"])
            H = m.calibrated(cal, rr)
            adapters = {}
            if d == 2:
                for a in As[1:]:
                    r.ledger(f"{key}_{a}_{hkey}", "adapter", train=T)
                    adapters[a] = m.ans["fit_adapter"](a, D["x"][T], rr, H, D["ci"][T], src)
            cache[token] = {"train": T, "cal": cal, "adapters": adapters}
        return token

    def preds(token, I):
        I = list(map(int, I))
        ca = cache[token]
        rr = raw[[loc[v] for v in I]]
        H = m.calibrated(ca["cal"], rr)
        return dict(H=H, **{a: m.ans["adapt"](mod, D["x"][I], rr, H, src) for a, mod in ca["adapters"].items()})

    # The four registered positions are exactly those in ROLE_FREEZE.  The
    # first three provide outer OOF examples; code 3 is the complete pool fit.
    for pos in spec["positions"]:
        code, A, B = int(pos["code"]), list(pos["A"]), list(pos["B"])
        al = {v: k for k, v in enumerate(A)}
        cross = {a: np.full((len(A), 4), np.nan) for a in As}
        deps = []
        for f in pos["inner"]:
            token = get(f["T"])
            pp = preds(token, f["V"])
            for a in As:
                cross[a][[al[v] for v in f["V"]]] = pp[a]
            deps.append(dict(T=list(f["T"]), V=list(f["V"]), token=token))
        base = get(A)
        vv = preds(base, B) if B else None
        positions[code] = dict(A=A, B=B, inner=deps, base=base, cross=cross)
        scalar_models[code], gate_models[code] = {}, {}
        # DATA15 stores the frozen reference CI as a scalar.  ``m.ci`` is
        # only for a four-parameter MED4 array and must not be applied here.
        yt = m.labels(D["ci"][A])

        if d == 1:
            yp = m.labels(m.ci(cross["H"]))
            ok = (yt != 9) & (yp != 9)
            for j, name in enumerate(["SOURCE-CI-ORDINAL", "HUBER-CI-ORDINAL"]):
                ciin = m.ci(raw[[loc[v] for v in A]]) if j == 0 else m.ci(cross["H"])
                r.ledger(f"{key}_scalar{code}_{j}", "ordinal", train=A)
                sm = m.scalar["fit"](name, ciin[ok], yt[ok])
                scalar_models[code][name] = sm
                scores.setdefault(name, dict(score=np.full(len(pool), np.nan), family=name, params=3, order=j))
                if B:
                    bi = [loc[v] for v in B]
                    ciB = m.ci(raw[bi]) if j == 0 else m.ci(vv["H"])
                    scores[name]["score"][bi] = m.scalar["score"](sm, ciB, m.labels(m.ci(vv["H"])))
        else:
            Fs, lrs = {}, {}
            for a in As:
                F = m.gm.features(D["x"][A], cross[a], src)
                Fs[a] = F
                yp = m.labels(m.ci(cross[a]))
                r.ledger(f"{key}_LR_{code}_{a}", "LR", train=A)
                lr = _fit_error_lr(F, yp, yt)
                lrs[a] = lr
                for gi, gate in enumerate(["ERROR-LR", "MARGIN"]):
                    nm = a + "|" + gate
                    params = (0 if a == "H" else 9 if a.startswith("L") else len(pool) + 1)
                    params += 14 if gate == "ERROR-LR" else 0
                    scores.setdefault(nm, dict(score=np.full(len(pool), np.nan), adapter=a, gate=gate, params=params, order=As.index(a) * 2 + gi))
                    if B:
                        vp = m.labels(m.ci(vv[a]))
                        VF = m.gm.features(D["x"][B], vv[a], src)
                        sc = _lr_score(lr, VF, vp) if gate == "ERROR-LR" else -np.minimum(abs(m.ci(vv[a]) + 3), abs(m.ci(vv[a]) - 3))
                        scores[nm]["score"][[loc[v] for v in B]] = sc
            positions[code].update(features=Fs, lrs=lrs)
        if B:
            for a in As:
                OOF[a][[loc[v] for v in B]] = vv[a]

    yt_pool = m.labels(D["ci"][pool])
    if d == 1:
        selection = m.st["select"](scores, m.labels(m.ci(OOF["H"])), yt_pool, .1)
        final_selection = selection["selected"]
    else:
        selection = m.ans["select"](scores, {a: m.labels(m.ci(p)) for a, p in OOF.items()}, yt_pool)
        gate_scores = {n: np.full(len(pool), np.nan) for n in m.gm.NAMES}
        allowed = {n: np.zeros(len(pool), bool) for n in m.gm.NAMES}
        oof = np.full((len(pool), 4), np.nan)
        for code, p in positions.items():
            A, B, cross = p["A"], p["B"], p["cross"]
            yi = m.labels(D["ci"][A])
            if code == 3:
                a = selection["selected"]["adapter"]
            else:
                rows = []
                for order, a0 in enumerate(As):
                    j = m.ci(cross[a0])
                    rows.append(dict(m.gs["threshold"](-np.minimum(abs(j + 3), abs(j - 3)), m.labels(j), yi), name=a0, params=0 if a0 == "H" else 9 if a0.startswith("L") else len(A) + 1, order=order))
                a = m.gs["choose"](rows)["name"]
            # fit_all internally keeps the old six-head implementations and
            # writes its own per-head ledger records through current runtime.
            mods = m.gm.fit_all(p["features"][a], cross[a], D["ci"][A], p["lrs"][a], key + f"_sixgate{code}")
            gate_models[code] = dict(adapter=a, models=mods)
            if B:
                bi = [loc[v] for v in B]
                med = OOF[a][bi]
                VF = m.gm.features(D["x"][B], med, src)
                oof[bi] = med
                for n in m.gm.NAMES:
                    gate_scores[n][bi], allowed[n][bi], _ = m.gm.score(n, mods[n], VF)
        yy = m.labels(m.ci(oof))
        thresholds = {n: dict(m.gs["threshold"](gate_scores[n], yy, yt_pool, allowed[n]), name=n, params=m.gm.PARAMS[j], order=j) for j, n in enumerate(m.gm.NAMES)}
        final_selection = m.gs["choose"](list(thresholds.values()))
        extra = dict(gate_oof=oof, gate_scores=gate_scores, allowed=allowed, thresholds=thresholds)

    out = dict(
        target=target,
        arm=arm,
        budget=label,
        mode=mode,
        seed=int(seed),
        source_file=str(source_file),
        source_sha256=r.sha(source_file),
        source_family=source_family,
        pool=pool,
        test=spec["test"],
        test_labels_hidden=True,
        cache=cache,
        positions=positions,
        scalar_models=scalar_models,
        gate_models=gate_models,
        source_model=src,
        selection=selection,
        final_selection=final_selection,
        OOF=OOF,
        scores=scores,
        extra=extra,
        fit_seconds=float(time.time() - started),
        external_test_evaluated=False,
    )
    m.dump(outpath.relative_to(r.ROOT), out)
    r.guard()
    print("TARGET15", target, arm, seed, round(out["fit_seconds"], 2), flush=True)
    return out


def _write_final_freeze(files: list[Path]) -> None:
    ledger_files = sorted((r.ROOT / "ledger").glob("*.json"))
    kinds = {}
    units = 0
    for p in ledger_files:
        z = r.read(p)
        kind = str(z.get("kind", ""))
        units += int(z.get("units", 0))
        kinds[kind] = kinds.get(kind, 0) + int(z.get("units", 0))
    r.write("TARGET_ALL_FROZEN.json", {
        "version": "legacy_aligned_threecenter_target15_v1",
        "created": time.time(),
        "files": {str(p.relative_to(r.ROOT)): r.sha(p) for p in files},
        "source_freeze": r.sha(r.ROOT / "SOURCE_FREEZE.json"),
        "target_fit_freeze": r.sha(r.ROOT / "TARGET_FIT_FREEZE.json"),
        "roles": r.sha(r.ROOT / "ROLE_FREEZE.json"),
        "code": r.sha(r.ROOT / "train_target_legacy.py"),
        "budgets": ["FULL", "C20"],
        "targets": list(TARGETS),
        "seeds": list(SEEDS),
        "fit_units": units,
        "ledger_kinds": kinds,
        "external_test_evaluated": False,
        "test_labels_read": False,
        "selection_frozen_from_calibration_only": True,
    })
    r.write("RESOURCE_LEDGER_TARGET15.json", {"fit_units": units, "kinds": kinds, "resources": r.guard()})


def _write_prefix_final_freeze(mode: str, files: list[Path]) -> None:
    horizon = int(mode[1:])
    ledger_files = sorted((r.ROOT / "ledger").glob("*.json"))
    kinds, units = {}, 0
    for p in ledger_files:
        z = r.read(p)
        kind = str(z.get("kind", ""))
        units += int(z.get("units", 0))
        kinds[kind] = kinds.get(kind, 0) + int(z.get("units", 0))
    r.write(f"PREFIX_TARGET_{horizon}_ALL_FROZEN.json", {
        "version": f"legacy_aligned_threecenter_prefix_{horizon}_v1",
        "created": time.time(),
        "files": {str(p.relative_to(r.ROOT)): r.sha(p) for p in files},
        "source_freeze": r.sha(r.ROOT / "SOURCE_FREEZE.json"),
        "prefix_source_freeze": r.sha(r.ROOT / "PREFIX_SOURCE_FREEZE_V2.json"),
        "target_fit_freeze": r.sha(r.ROOT / f"PREFIX_TARGET_FIT_FREEZE_{horizon}.json"),
        "roles": r.sha(r.ROOT / "ROLE_FREEZE.json"),
        "code": r.sha(r.ROOT / "train_target_legacy.py"),
        "mode": mode, "horizon_seconds": horizon,
        "budgets": ["FULL", "C20"], "targets": list(TARGETS), "seeds": list(SEEDS),
        "fit_units": units, "ledger_kinds": kinds,
        "external_test_evaluated": False, "test_labels_read": False,
        "selected_source_config_reused": True,
    })
    r.write(f"RESOURCE_LEDGER_PREFIX_TARGET_{horizon}.json", {"fit_units": units, "kinds": kinds, "resources": r.guard()})


def main() -> None:
    if not (r.ROOT / "SOURCE_FREEZE.json").exists():
        raise RuntimeError("SOURCE_FREEZE.json is required before target-side fits")
    role = _load_role()
    _freeze_before_fit(role, None)
    D = _load_data(None)
    files = []
    for target in TARGETS:
        for seed in SEEDS:
            for budget in ("20", "FULL"):
                run_one(target, seed, budget, mode="T900", source_path=None, data_path=None)
                label = "FULL" if _budget_label(budget) == "FULL" else "20"
                files.append(r.ROOT / "target15" / f"{target}_C{label}_seed{seed}.joblib")
    if len(files) != 120 or not all(p.exists() for p in files):
        raise RuntimeError("target15 output count mismatch")
    _write_final_freeze(files)
    print("TARGET15_COMPLETE", len(files), flush=True)


def main_prefix(mode: str) -> None:
    if mode not in {"T300", "T600"}:
        raise ValueError(mode)
    if not (r.ROOT / "SOURCE_FREEZE.json").exists() or not (r.ROOT / "PREFIX_SOURCE_FREEZE_V2.json").exists():
        raise RuntimeError("SOURCE_FREEZE and PREFIX_SOURCE_FREEZE_V2 are required before prefix target fits")
    role = _load_role()
    data_path = r.ROOT / "PREFIX_X.npz"
    source_path = r.ROOT / "prefix_source_v2"
    files = []
    for target in TARGETS:
        for seed in SEEDS:
            for budget in ("20", "FULL"):
                run_one(target, seed, budget, mode=mode, source_path=source_path, data_path=data_path)
                label = "FULL" if _budget_label(budget) == "FULL" else "20"
                files.append(r.ROOT / f"targetprefix_{mode[1:]}" / f"{target}_C{label}_seed{seed}.joblib")
    if len(files) != 120 or not all(p.exists() for p in files):
        raise RuntimeError(f"{mode} prefix output count mismatch")
    _write_prefix_final_freeze(mode, files)
    print(f"{mode}_PREFIX_COMPLETE", len(files), flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in {"T300", "T600"}:
        main_prefix(sys.argv[1])
    else:
        main()
