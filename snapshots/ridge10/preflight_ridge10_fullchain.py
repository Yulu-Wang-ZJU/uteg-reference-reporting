"""Read-only identity preflight for the fixed Ridge-10 fullchain.

This command never fits, writes historical paths, or reads test labels.  It
checks that a source payload's Ridge-10 full-C object is the same object as the
candidate full model file and that the role/OOF contracts are complete.  The
result is written only below this run root.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import runtime as r  # sets cache/TMP/thread variables before scientific imports
import joblib
import numpy as np


SEEDS = tuple(range(1000, 1020))
BASE_DATA = r.BASE / "DATA15.npz"
ROLE_PATH = r.BASE / "ROLE_FREEZE.json"
SPLIT_PATH = r.NINE / "NESTED_SPLIT_MANIFEST.json"
AUDIT_DIR = r.OUT / "mapper_followup_audit_v1"
# The authoritative source revision table is in the dedicated revision
# folder; the earlier follow-up folder is a superseded snapshot.
CONTEXTS_PATH = r.OUT / "source_revision" / "SOURCE_REVISION_CONTEXTS.csv"
EXP_DATA = r.EXP / "EXP_DATA.npz"

SOURCE_ROWS = {
    "Huzhou": None,
    "SAHZU": None,
    "Hubei": None,
}


def _digest_array(a: object) -> str:
    x = np.asarray(a)
    h = hashlib.sha256()
    h.update(str(x.dtype).encode())
    h.update(repr(tuple(x.shape)).encode())
    h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()


def _model_digest(model: dict) -> str:
    h = hashlib.sha256()
    for key in ("name", "family", "config", "train", "validation", "xm", "xs", "ym", "ys", "coef", "intercept", "seed", "train_z", "dual"):
        h.update(key.encode())
        value = model.get(key)
        if isinstance(value, np.ndarray):
            h.update(_digest_array(value).encode())
        else:
            h.update(json.dumps(value, sort_keys=True, default=str).encode())
    return h.hexdigest()


def _paths(direction: str, seed: int) -> dict[str, Path | str]:
    if direction == "HZ+HB_to_SA":
        return {
            "target": "SAHZU", "source_kind": "dual_base",
            "full": r.BASE / "source_models" / f"SAHZU_seed{seed}_RIDGE_full.joblib",
            "bundle": r.BASE / "source" / f"SAHZU_seed{seed}.joblib",
        }
    if direction == "SA+HB_to_HZ":
        return {
            "target": "Huzhou", "source_kind": "dual_base",
            "full": r.BASE / "source_models" / f"Huzhou_seed{seed}_RIDGE_full.joblib",
            "bundle": r.BASE / "source" / f"Huzhou_seed{seed}.joblib",
        }
    if direction == "SA+HZ_to_HB":
        return {
            "target": "Hubei", "source_kind": "dual_base",
            "full": r.BASE / "source_models" / f"Hubei_seed{seed}_RIDGE_full.joblib",
            "bundle": r.BASE / "source" / f"Hubei_seed{seed}.joblib",
        }
    if direction in {"HZ_to_SA", "HZ_to_HB"}:
        return {
            "target": "Huzhou", "source_kind": "single_exp_D2",
            "full": r.EXP / "source_models" / f"D2_seed{seed}_RIDGE-10_full.joblib",
            "bundle": r.EXP / "source" / f"D2_seed{seed}.joblib",
        }
    if direction in {"SA_to_HZ", "SA_to_HB"}:
        return {
            "target": "SAHZU", "source_kind": "single_exp_D1",
            "full": r.EXP / "source_models" / f"D1_seed{seed}_RIDGE-10_full.joblib",
            "bundle": r.EXP / "source" / f"D1_seed{seed}.joblib",
        }
    if direction in {"HB_to_SA", "HB_to_HZ"}:
        return {
            "target": "Hubei", "source_kind": "single_hubei",
            "full": r.NINE / "models" / "shared_hubei_source" / f"Hubei_seed{seed}_RIDGE_full.joblib",
            "bundle": r.NINE / "models" / "shared_hubei_source" / f"Hubei_seed{seed}.joblib",
        }
    raise ValueError(direction)


DIRECTIONS = (
    "HZ+HB_to_SA", "SA+HB_to_HZ", "SA+HZ_to_HB",
    "HZ_to_SA", "HB_to_SA", "SA_to_HZ", "HB_to_HZ", "SA_to_HB", "HZ_to_HB",
)


def _load_base() -> dict[str, np.ndarray]:
    with np.load(BASE_DATA, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _canonical_source_rows(data: dict[str, np.ndarray]) -> dict[str, list[int]]:
    centers = data["centers"].astype(str)
    qc = np.asarray(data["input_qc"], dtype=bool)
    return {
        "Huzhou": sorted(np.flatnonzero(centers == "Huzhou").astype(int).tolist()),
        "SAHZU": sorted(np.flatnonzero(centers == "SAHZU").astype(int).tolist()),
        "Hubei": sorted(np.flatnonzero((centers == "Hubei") & qc).astype(int).tolist()),
    }


def _expected_rows(direction: str, rows: dict[str, list[int]]) -> list[int]:
    if direction == "HZ+HB_to_SA":
        return sorted(rows["Huzhou"] + rows["Hubei"])
    if direction == "SA+HB_to_HZ":
        return sorted(rows["SAHZU"] + rows["Hubei"])
    if direction == "SA+HZ_to_HB":
        return sorted(rows["SAHZU"] + rows["Huzhou"])
    if direction in {"HZ_to_SA", "HZ_to_HB"}:
        return rows["Huzhou"]
    if direction in {"SA_to_HZ", "SA_to_HB"}:
        return rows["SAHZU"]
    if direction in {"HB_to_SA", "HB_to_HZ"}:
        return rows["Hubei"]
    raise ValueError(direction)


def _expected_fit_rows(source_kind: str, expected_rows: list[int]) -> list[int]:
    """Exact row IDs in the source dataset used by a full deployment fit."""
    if source_kind == "single_exp_D1":
        return list(range(304))
    if source_kind == "single_exp_D2":
        return list(range(304, 399))
    return list(expected_rows)


def _row_hash(rows: list[int]) -> str:
    return hashlib.sha256("\n".join(str(v) for v in rows).encode()).hexdigest()


def _model_ok(model: object, seed: int, expected_n: int,
              expected_train_rows: list[int] | None = None) -> tuple[bool, list[str]]:
    problems: list[str] = []
    if not isinstance(model, dict):
        return False, ["model_not_dict"]
    if model.get("name") != "RIDGE-10":
        problems.append(f"name={model.get('name')!r}")
    if model.get("family") != "RIDGE":
        problems.append(f"family={model.get('family')!r}")
    if model.get("config", {}).get("alpha") != 10:
        problems.append(f"alpha={model.get('config', {}).get('alpha')!r}")
    # ``model['seed']`` is the deterministic fold/family fit seed in the
    # historical mapper, whereas the outer source identity is the payload's
    # seed (1000..1019).  Require the former to be a finite integer but do not
    # conflate these two namespaces.
    try:
        int(model.get("seed", -1))
    except Exception:
        problems.append(f"model_seed={model.get('seed')!r}")
    for key in ("xm", "xs", "ym", "ys", "coef", "intercept"):
        if key not in model:
            problems.append(f"missing_{key}")
    if problems:
        return False, problems
    for key in ("xm", "xs", "ym", "ys", "coef", "intercept"):
        if not np.isfinite(np.asarray(model[key], dtype=float)).all():
            problems.append(f"nonfinite_{key}")
    if np.asarray(model["xm"]).shape != (4,) or np.asarray(model["xs"]).shape != (4,):
        problems.append("source_scaler_shape")
    if np.asarray(model["ym"]).shape != (4,) or np.asarray(model["ys"]).shape != (4,):
        problems.append("target_scaler_shape")
    if np.asarray(model["coef"]).shape != (4, 4) or np.asarray(model["intercept"]).shape != (4,):
        problems.append("ridge_coefficient_shape")
    if len(np.asarray(model.get("train", []))) != expected_n:
        problems.append(f"train_n={len(np.asarray(model.get('train', [])))} expected={expected_n}")
    if expected_train_rows is not None:
        train = np.asarray(model.get("train", []), dtype=int).tolist()
        if train != list(expected_train_rows):
            problems.append("train_row_identity_mismatch")
    if len(np.asarray(model.get("validation", []))) != 0:
        problems.append("full_validation_not_empty")
    if not bool(model.get("success", False)):
        problems.append("success_false")
    return not problems, problems


def run() -> dict:
    data = _load_base()
    rows = _canonical_source_rows(data)
    role = json.loads(ROLE_PATH.read_text(encoding="utf-8-sig"))
    contexts_hash = r.sha(CONTEXTS_PATH) if CONTEXTS_PATH.exists() else None
    records: list[dict] = []
    all_ok = True
    for direction in DIRECTIONS:
        p = _paths(direction, 1000)
        expected_rows = _expected_rows(direction, rows)
        expected_train_rows = _expected_fit_rows(str(p["source_kind"]), expected_rows)
        for seed in SEEDS:
            p = _paths(direction, seed)
            full = Path(p["full"])
            bundle_path = Path(p["bundle"])
            rec = {
                "direction": direction, "seed": seed, "source_kind": p["source_kind"],
                "full_path": str(full), "bundle_path": str(bundle_path),
                "expected_source_n": len(expected_rows), "expected_row_ids_sha256": hashlib.sha256(
                    "\n".join(str(v) for v in expected_rows).encode()).hexdigest(),
                "expected_fit_rows_n": len(expected_train_rows),
                "expected_fit_rows_sha256": _row_hash(expected_train_rows),
                "fit_data_sha256": r.sha(EXP_DATA if p["source_kind"] in {"single_exp_D1", "single_exp_D2"} else BASE_DATA),
                "full_exists": full.exists(), "bundle_exists": bundle_path.exists(),
                "status": "NOT_CHECKED",
            }
            if not bundle_path.exists():
                rec["status"] = "FAIL"
                rec["problems"] = ["source_bundle_missing"]
                records.append(rec); all_ok = False; continue
            try:
                bundle = joblib.load(bundle_path)
                bundle_model = bundle.get("families", {}).get("RIDGE", {}).get("model")
                oof = np.asarray(bundle["source_OOF"]["RIDGE-10"], dtype=float)
                full_obj = joblib.load(full) if full.exists() else None
                ok, full_problems = _model_ok(full_obj, seed, len(expected_rows), expected_train_rows) if full_obj is not None else (False, ["full_model_missing"])
                bok, bundle_problems = _model_ok(bundle_model, seed, len(expected_rows), expected_train_rows) if bundle_model is not None else (False, ["bundle_ridge_model_missing"])
                problems = []
                if oof.shape != (len(expected_rows), 4):
                    problems.append(f"source_oof_shape={oof.shape}")
                if not np.isfinite(oof).all():
                    problems.append("source_oof_nonfinite")
                source = list(map(int, np.asarray(bundle.get("source", []), dtype=int).tolist()))
                # EXP payloads carry local rows only; verify their IDs separately.
                if p["source_kind"] in {"dual_base", "single_hubei"} and source and sorted(source) != expected_rows:
                    problems.append("bundle_source_rows_mismatch")
                model_same = bool(full_obj is not None and bundle_model is not None and _model_digest(full_obj) == _model_digest(bundle_model))
                reusable = bool(ok and bok and model_same and not problems)
                needs_fit = bool(not reusable and not problems)
                rec.update({
                    "full_sha256": r.sha(full) if full.exists() else None, "bundle_sha256": r.sha(bundle_path),
                    "full_model_digest": _model_digest(full_obj) if full_obj is not None else None,
                    "bundle_model_digest": _model_digest(bundle_model) if bundle_model is not None else None,
                    "source_oof_shape": list(oof.shape), "source_rows_in_bundle_n": len(source),
                    "full_model_ok": bool(ok), "bundle_model_ok": bool(bok),
                    "full_model_problems": full_problems, "bundle_model_problems": bundle_problems,
                    "full_scaler_digest": _digest_array(np.concatenate([np.asarray(full_obj[k], dtype=float).ravel() for k in ("xm", "xs", "ym", "ys")])) if full_obj is not None else None,
                    "bundle_scaler_digest": _digest_array(np.concatenate([np.asarray(bundle_model[k], dtype=float).ravel() for k in ("xm", "xs", "ym", "ys")])) if bundle_model is not None else None,
                    "full_coef_digest": _digest_array(full_obj["coef"]) if full_obj is not None else None,
                    "bundle_coef_digest": _digest_array(bundle_model["coef"]) if bundle_model is not None else None,
                    "reusable": reusable, "needs_source_fit": needs_fit,
                    "status": "PASS" if reusable else ("SOURCE_REFIT_REQUIRED" if needs_fit else "FAIL"), "problems": problems,
                })
                if problems:
                    all_ok = False
            except Exception as exc:
                rec["status"] = "ERROR"; rec["problems"] = [f"{type(exc).__name__}: {exc}"]; all_ok = False
            records.append(rec)

    unique_refit = sorted({(x["full_path"], x["source_kind"]) for x in records if x.get("needs_source_fit")})
    hard_fail = [x for x in records if x.get("status") == "FAIL" or x.get("status") == "ERROR"]
    reusable_n = sum(bool(x.get("reusable")) for x in records)
    result = {
        "version": "ridge10_fullchain_source_identity_preflight_v1",
        "status": "PASS" if not hard_fail and not unique_refit else ("PASS_SOURCE_REFIT_REQUIRED" if not hard_fail else "FAIL"),
        "fit_units": 0,
        "test_labels_read": False,
        "test_evaluated": False,
        "base_data_sha256": r.sha(BASE_DATA), "exp_data_sha256": r.sha(EXP_DATA), "role_sha256": r.sha(ROLE_PATH),
        "nested_split_sha256": r.sha(SPLIT_PATH), "source_contexts_sha256": contexts_hash,
        "directions": list(DIRECTIONS), "seeds": list(SEEDS),
        "source_full_candidates": len(records), "source_full_pass": reusable_n,
        "source_full_refit_contexts": len(unique_refit),
        "source_full_refit_paths": [x[0] for x in unique_refit],
        "source_full_fail": len(hard_fail),
        "records": records,
    }
    r.write("SOURCE_FULL_IDENTITY_PREFLIGHT.json", result)
    # A compact manifest is convenient for the runner and verifier.
    if not hard_fail:
        r.write("SOURCE_FULL_REUSE_MANIFEST.json", {
            "status": result["status"], "fit_units": len(unique_refit), "records": [
                {"direction": x["direction"], "seed": x["seed"], "full_path": x["full_path"],
                 "bundle_path": x["bundle_path"], "full_sha256": x["full_sha256"],
                 "bundle_sha256": x["bundle_sha256"], "model_digest": x["full_model_digest"],
                 "reusable": bool(x.get("reusable")), "needs_source_fit": bool(x.get("needs_source_fit"))}
                for x in records
            ],
        })
    return result


if __name__ == "__main__":
    out = run()
    print(json.dumps({k: out[k] for k in ("status", "source_full_candidates", "source_full_pass", "source_full_fail")}, ensure_ascii=False))
