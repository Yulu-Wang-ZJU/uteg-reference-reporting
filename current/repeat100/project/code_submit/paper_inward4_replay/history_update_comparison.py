"""Three frozen-input comparisons of revisable reporting decisions.

Does not import the mapping runner, refit MA mappings, or write manuscript assets.
Candidate selection is fitted using development records only. Reference outcomes
for evaluation records are consumed exclusively by the evaluation functions.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from . import unified_hgb_reporting as base
from .input_contract import input_provenance

ROOT = base.ROOT
OUT = ROOT / "history_update_comparison_20261105"
PRIVATE = OUT / "private"
TIMES = [5, 10, 15]
METHODS = ["A_current", "B_origin_risk", "C_current_risk"]
ARMS = base.ARMS
HGB = dict(learning_rate=.05, max_iter=100, max_leaf_nodes=4,
           max_depth=2, min_samples_leaf=10, l2_regularization=10.,
           max_bins=255, early_stopping=False, random_state=73051)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected():
    directory = base.WS / "outputs/uteg_nc_goal_20261030/manuscript"
    extensions = {".tex", ".pdf", ".pptx", ".zip", ".png", ".svg", ".xlsx"}
    return {str(p.relative_to(base.WS)): digest(p) for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions}


def initialize():
    PRIVATE.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        groups={
            "A_current": "Current-checkpoint prediction and original Table 2 HGB threshold.",
            "B_origin_risk": "All available predictions are candidates. Calibrate their ORIGINAL checkpoint HGB scores to errors on development records; choose the lowest calibrated risk, latest candidate on exact ties.",
            "C_current_risk": "Re-score every available candidate using current physical inputs, all available state predictions and HGB scores, and candidate origin/state. Choose the lowest current risk, latest candidate on exact ties.",
        },
        checkpoint_5="All three groups use identical original Table 2 HGB decisions at 5 minutes, isolating later updates.",
        input="Frozen existing MA predictions, signal features and HGB scores. No MA refitting or input modification.",
        development="Reuse frozen three development partitions. B calibration and C risk predictions are cross-fitted by record/group. Fit final calibrators/risk model on all development records. Choose release threshold from cross-fitted development choices using the original selector: agreement >=85%, >=5 reports, >=20% coverage, maximize coverage.",
        upstream="Upstream development predictions/scores are existing OOF caches; they are not regenerated within the NEW meta-model cross-fit. Thus meta cross-fit is not a fresh fully nested upstream pipeline. Outer evaluation records/groups remain excluded from all fitting and thresholds.",
        model=HGB,
        candidate_weight="Each original development record has total weight 1, divided equally among its available candidates. No class weights, FN-specific penalty, augmentation, or hyperparameter search.",
        features="Current four U-TEG parameters; each available checkpoint's predicted state one-hot, finite HGB score and validity; candidate state one-hot, original HGB score and source minute; agreement with latest state and number of earlier/current state agreements. Never reference MA or future checkpoint inputs.",
        invalid="Invalid candidates cannot be selected or released. Missing HGB features use 1 plus a validity indicator. HGB preprocessor fits development-only medians/means/scales, clips to [-10,10].",
        updating="Previous predictions remain candidates; no lock on previous releases. Every checkpoint can update, newly release or withhold a record. Current trusted coverage need not be monotone. Ever-released coverage is recorded separately.",
        protocols={"paper": "60 development / 201 evaluation, 20 frozen development partitions; 4020 record-seed events, NOT 4020 independent records.",
                   "internal301": "Frozen grouped 5-fold outer / 3-fold inner; 301 records, each predicted once. Kept separate from paper evaluation."},
        metrics="Current coverage=reported/all; agreement=correct/reported; FNR=abnormal predicted normal/released reference abnormal; FPR=normal predicted abnormal/released reference normal. Opposite abnormal states are three-class errors, not binary FN. Low<50, normal 50..70, high>70 mm.",
        fn_diagnostic="Reconstruct old first-locked 85% HGB decisions from A. Track its final FN set AFTER decisions are generated. Correct-direction release, opposite-direction abnormal, still FN, and withheld are separate. New FN versus that set and transitions from previous checkpoint are reported. Never use diagnostic subset for fitting.",
        selection="One predeclared experiment; no adjustments based on evaluation outcomes. All three mapping arms reported.",
        provenance="This analysis compares reporting decisions using saved predictions.",
        manuscript="No manuscript, figure, PDF, supplementary package or existing result modified.",
    )
    path = OUT / "PROTOCOL.json"
    if path.exists():
        assert base.read(path) == protocol, "Protocol changed after freezing"
    else:
        base.write(path, protocol)
        base.write(OUT / "PROTECTED_BEFORE.json", protected())


class Contexts:
    def __init__(self):
        self.data = base.load(ROOT / "data/DATA15_CURRENT.npz")
        self.folds = base.read(ROOT / "data/FROZEN_SPLITS.json")
        self.roles = base.read(ROOT / "data/ROLE_FREEZE.json")["roles"]["SAHZU"]
        curves = base.load(ROOT / "data/CURRENT_SCALED_CURVES.npz")
        np.testing.assert_array_equal(curves["ids"], self.data["ids"])
        self.features = {int(h)//60: curves["prefix_parameters"][i]
                         for i, h in enumerate(curves["horizons"])}
        self.features[15] = self.data["x"]
        self.lookup = {str(v): i for i, v in enumerate(self.data["ids"])}
        self.cache = {}

    def load(self, path):
        if path not in self.cache:
            self.cache[path] = base.read(path) if path.suffix == ".json" else base.load(path)
        return self.cache[path]

    def build(self, protocol, arm, repeat):
        development, query = {}, {}
        if protocol == "paper":
            partitions = self.roles["positions_full"][str(1000+repeat)][:3]
            order = np.concatenate([p["B"] for p in partitions])
            split_id = np.concatenate([np.full(len(p["B"]), i) for i, p in enumerate(partitions)])
            for t in TIMES:
                z = self.load(ROOT / "gates" / f"{arm}_T{t*60}.npz")
                v = self.load(ROOT / "gates" / f"{arm}_T{t*60}.development.json")[repeat]
                rows = np.array([self.lookup[r["record_id"]] for r in v["rows"]])
                np.testing.assert_array_equal(rows, order)
                development[t] = dict(pred=np.asarray(v["pred"]), truth=np.asarray(v["truth"]),
                    score=np.array([np.inf if a is None else a for a in v["scores"]["HGB"]]))
                query[t] = dict(prediction=z["prediction"][repeat], score=z["HGB_score"][repeat],
                                expected_action=z["HGB_action"][repeat])
                if t == 5:
                    test = z["test"]
                np.testing.assert_array_equal(test, z["test"])
        else:
            f = self.folds[repeat]
            z = self.load(ROOT / "early_subset_20261103/private" / f"all301_{arm}_fold{repeat}.npz")
            order, test = z["dev_rows"], z["test"]
            split_map = {int(i): k for k, sub in enumerate(f["inner"]) for i in sub["V"]}
            split_id = np.array([split_map[int(i)] for i in order])
            for t in [5, 10]:
                development[t] = dict(pred=z[f"dev_pred{t}"], truth=z["dev_truth"], score=z[f"dev_HGB_{t}"])
                query[t] = dict(prediction=z[f"query_ma{t}"], score=z[f"query_HGB_{t}"])
            q = self.load(ROOT / "unified_hgb_20261105/private" / f"internal15_{arm}_{repeat}.npz")
            qorder = np.concatenate([s["V"] for s in f["inner"]])
            lookup = {int(v): i for i, v in enumerate(qorder)}
            align = np.array([lookup[int(v)] for v in order])
            development[15] = dict(pred=q["dev_pred"][align], truth=q["dev_truth"][align], score=q["dev_score"][align])
            query[15] = dict(prediction=q["prediction"], score=q["score"])
            np.testing.assert_array_equal(test, q["test"])
            np.testing.assert_array_equal(test, f["B"])
            for t in TIMES:
                saved = self.load(ROOT / "all301_gates" / f"{arm}_T{t*60}.npz")
                loc = {int(v): i for i, v in enumerate(saved["local"])}
                ix = [loc[int(i)] for i in test]
                np.testing.assert_array_equal(query[t]["prediction"], saved["prediction"][ix])
                query[t]["expected_action"] = saved["HGB"][ix]
        assert len(np.unique(order)) == len(order)
        assert not set(order) & set(test)
        groups = self.data["groups"][order]
        assert not set(groups) & set(self.data["groups"][test])
        for k in range(3):
            assert not set(groups[split_id == k]) & set(groups[split_id != k])
        for t in TIMES:
            np.testing.assert_array_equal(development[t]["truth"], base.state(self.data["medical"][order, 2]))
            development[t]["features"] = self.features[t][order]
            query[t]["features"] = self.features[t][test]
            action = base.apply(query[t]["score"], query[t]["prediction"], base.threshold(development[t], .85))
            np.testing.assert_array_equal(action, query[t].pop("expected_action"))
        # Query contains no reference outcomes; selection cannot consume them.
        return dict(dev=development, query=query, split_id=split_id, train=order, test=test, groups=groups)


def choose(risks, classes):
    legal = np.isfinite(risks) & np.isin(classes, [-1, 0, 1])
    scores = np.where(legal, risks, np.inf)
    # Reverse first: prefer the latest available candidate on exact ties.
    index = scores.shape[1]-1-np.argmin(scores[:, ::-1], axis=1)
    rows = np.arange(len(index))
    return index, scores[rows, index]


def calibration_risks(development, query, folds, t):
    times = [v for v in TIMES if v <= t]
    n, nq = len(folds), len(query[t]["score"])
    dr, qr = np.full((n, len(times)), np.inf), np.full((nq, len(times)), np.inf)
    for j, source in enumerate(times):
        d = development[source]
        for k in range(3):
            train, valid = folds != k, folds == k
            subset = {key: np.asarray(d[key])[train] for key in ["score", "pred", "truth"]}
            dr[valid, j] = base.calibrated_score(subset, np.asarray(d["score"])[valid])
        qr[:, j] = base.calibrated_score(d, query[source]["score"])
    return dr, qr


def features(bank, t, is_query=False):
    times = [v for v in TIMES if v <= t]
    classes = np.column_stack([base.state(bank[v]["prediction"]) if is_query else bank[v]["pred"] for v in times])
    raw = np.column_stack([bank[v]["score"] for v in times])
    valid = np.isfinite(raw) & np.isin(classes, [-1, 0, 1])
    scores = np.where(np.isfinite(raw), raw, 1.)
    onehot = lambda a: np.column_stack([a == k for k in [-1, 0, 1]]).astype(float)
    shared = np.column_stack([bank[t]["features"]] +
        [item for j in range(len(times)) for item in [onehot(classes[:, j]), scores[:, j], valid[:, j]]])
    stacks = []
    for j, source in enumerate(times):
        stacks.append(np.column_stack([shared, onehot(classes[:, j]), scores[:, j],
            np.full(len(classes), source/15.), classes[:, j] == classes[:, -1],
            np.sum(classes == classes[:, j, None], axis=1)/len(times)]))
    return np.stack(stacks, axis=1), classes, valid


def fit_risk(x, error, valid):
    n, candidates, _ = x.shape
    weights = np.broadcast_to(1/np.maximum(valid.sum(axis=1), 1)[:, None], valid.shape)
    xx, yy, ww = x[valid], error[valid].astype(int), weights[valid]
    # One unit of total training weight per original record, never per candidate.
    assert np.isclose(ww.sum(), np.sum(valid.any(axis=1)))
    pre = {}
    for name, function, fallback in [("median", np.median, 0.), ("mean", np.mean, 0.), ("sd", np.std, 1.)]:
        pre[name] = np.array([function(col[np.isfinite(col)]) if np.isfinite(col).any() else fallback for col in xx.T])
    pre["sd"] = np.maximum(pre["sd"], 1e-6)
    model = dict(pre=pre, n_records=n, n_candidate_rows=len(yy), total_weight=float(ww.sum()))
    if len(yy) < 8 or len(np.unique(yy)) < 2 or min(np.sum(yy == 0), np.sum(yy == 1)) < 2:
        model["constant"] = float((np.sum(ww*yy)+.5)/(ww.sum()+1.))
    else:
        model["clf"] = HistGradientBoostingClassifier(**HGB).fit(transform(xx, pre), yy, sample_weight=ww)
    return model


def transform(x, pre):
    return np.clip((np.where(np.isfinite(x), x, pre["median"])-pre["mean"])/pre["sd"], -10, 10)


def score_risk(model, x, valid):
    result = np.full(valid.shape, np.inf)
    if valid.any():
        result[valid] = model["constant"] if "constant" in model else model["clf"].predict_proba(transform(x[valid], model["pre"]))[:, 1]
    return result


def current_risks(dev, query, folds, t):
    dx, dc, dv = features(dev, t)
    qx, qc, qv = features(query, t, True)
    error = dc != np.asarray(dev[t]["truth"])[:, None]
    risks = np.full(dv.shape, np.inf)
    for k in range(3):
        train, valid = folds != k, folds == k
        model = fit_risk(dx[train], error[train], dv[train])
        risks[valid] = score_risk(model, dx[valid], dv[valid])
    final = fit_risk(dx, error, dv)
    return risks, score_risk(final, qx, qv), final


def decisions(context, save_models=None):
    dev, query, folds = context["dev"], context["query"], context["split_id"]
    result, fits = {}, []
    n = len(query[5]["prediction"])
    for t in TIMES:
        cutoff = base.threshold(dev[t], .85)
        p = np.asarray(query[t]["prediction"])
        result["A_current", t] = dict(prediction=p.copy(), score=query[t]["score"].copy(),
            action=base.apply(query[t]["score"], p, cutoff), source_time=np.full(n, t))
        fits.append(dict(method="A_current", minute=t, cutoff=cutoff))
        if t == 5:
            for method in METHODS[1:]:
                result[method, t] = {k: v.copy() for k, v in result["A_current", t].items()}
            continue
        times = np.array([v for v in TIMES if v <= t])
        dclasses = np.column_stack([dev[v]["pred"] for v in times])
        qvalues = np.column_stack([query[v]["prediction"] for v in times])
        qclasses = base.state(qvalues)
        for method in METHODS[1:]:
            if method == "B_origin_risk":
                dr, qr = calibration_risks(dev, query, folds, t)
            else:
                dr, qr, model = current_risks(dev, query, folds, t)
                if save_models:
                    import joblib
                    joblib.dump(model, PRIVATE / f"{save_models}_{t}_risk.joblib", compress=3)
            di, ds = choose(dr, dclasses)
            qi, qs = choose(qr, qclasses)
            dp = dclasses[np.arange(len(di)), di]
            cutoff = base.threshold(dict(score=ds, pred=dp, truth=dev[t]["truth"]), .85)
            p = qvalues[np.arange(n), qi]
            a = base.apply(qs, p, cutoff)
            result[method, t] = dict(prediction=p, action=a, score=qs, source_time=times[qi])
            fit_action = base.apply(ds, dp.astype(float), cutoff)
            fits.append(dict(method=method, minute=t, cutoff=cutoff, dev_n=len(di),
                dev_reported=int(fit_action.sum()),
                dev_errors=int(np.sum(fit_action & (dp != dev[t]["truth"]))),
                dev_sources={str(v): int(np.sum(fit_action & (times[di] == v))) for v in times}))
    return result, fits


def locked_baseline(results):
    n = results["A_current", 5]["action"].shape
    p, a, source = np.full(n, np.nan), np.zeros(n, bool), np.full(n, -1)
    out = {}
    for t in TIMES:
        r = results["A_current", t]
        new = r["action"] & ~a
        p[new] = r["prediction"][new]
        source[new] = t
        a |= new
        out[t] = dict(prediction=p.copy(), action=a.copy(), source_time=source.copy())
    return out


def fn_mask(p, a, truth):
    return a & (truth != 0) & (base.state(p) == 0)


def trajectory(result, reference, old):
    truth = np.broadcast_to(base.state(reference), old[15]["prediction"].shape)
    frozen_fn = fn_mask(old[15]["prediction"], old[15]["action"], truth)
    diagnostics = []
    for method in METHODS:
        previous = None
        ever = np.zeros(truth.shape, bool)
        for t in TIMES:
            r = result[method, t]
            a, pc = r["action"], base.state(r["prediction"])
            fn = fn_mask(r["prediction"], a, truth)
            old_early_fn = frozen_fn & old[t]["action"]
            subset = dict(
                frozen_old_fn_count=int(frozen_fn.sum()),
                correct_direction_released=int(np.sum(frozen_fn & a & (pc == truth))),
                opposite_abnormal_released=int(np.sum(frozen_fn & a & (pc != 0) & (pc != truth))),
                still_fn=int(np.sum(frozen_fn & fn)),
                withheld=int(np.sum(frozen_fn & ~a)),
                corrected_after_old_fn_had_occurred=int(np.sum(old_early_fn & a & (pc == truth))),
                new_fn_outside_old_frozen_set=int(np.sum(fn & ~frozen_fn)),
            )
            assert sum(subset[k] for k in ["correct_direction_released", "opposite_abnormal_released", "still_fn", "withheld"]) == int(frozen_fn.sum())
            ever |= a
            row = dict(method=method, minute=t, **subset, ever_reported=int(ever.sum()),
                       using_earlier_prediction=int(np.sum(a & (r["source_time"] < t))))
            if previous is not None:
                pa, pp = previous["action"], base.state(previous["prediction"])
                pfn = fn_mask(previous["prediction"], pa, truth)
                row.update(
                    previous_fn_corrected=int(np.sum(pfn & a & (pc == truth))),
                    previous_fn_opposite_abnormal=int(np.sum(pfn & a & (pc != 0) & (pc != truth))),
                    previous_fn_withheld=int(np.sum(pfn & ~a)),
                    previous_fn_retained=int(np.sum(pfn & fn)),
                    new_fn_since_previous=int(np.sum(fn & ~pfn)),
                    new_fn_from_previously_correct_abnormal=int(np.sum(fn & pa & (pp == truth))),
                    new_fn_from_previously_withheld=int(np.sum(fn & ~pa)),
                    new_fn_from_opposite_abnormal=int(np.sum(fn & pa & (pp != truth) & (pp != 0))),
                    newly_reported=int(np.sum(a & ~pa)), newly_withheld=int(np.sum(pa & ~a)),
                    changed_report_state=int(np.sum(a & pa & (pc != pp))))
                assert row["new_fn_since_previous"] == sum(row[k] for k in ["new_fn_from_previously_correct_abnormal", "new_fn_from_previously_withheld", "new_fn_from_opposite_abnormal"])
            diagnostics.append(row)
            previous = r
    return diagnostics, frozen_fn


def smoke():
    inputs = Contexts()
    checks = []
    for protocol in ["paper", "internal301"]:
        context = inputs.build(protocol, "HZ+HB_to_SA", 0)
        result, fit = decisions(context)
        poison = {**context, "query": {t: {**v, "truth": np.full(len(context["test"]), -999),
                    "reference_ma": np.full(len(context["test"]), 9999.)} for t, v in context["query"].items()}}
        result2, fit2 = decisions(poison)
        assert fit == fit2
        for key in result:
            for field in result[key]:
                np.testing.assert_array_equal(result[key][field], result2[key][field])
        future = {**context, "query": {t: dict(v) for t, v in context["query"].items()}}
        future["query"][15] = dict(prediction=np.full(len(context["test"]), -400.),
            score=np.zeros(len(context["test"])), features=np.full((len(context["test"]), 4), 100000.))
        result3, _ = decisions(future)
        for key in result:
            if key[1] < 15:
                for field in result[key]:
                    np.testing.assert_array_equal(result[key][field], result3[key][field])
        for method in METHODS[1:]:
            for field in result[method, 5]:
                np.testing.assert_array_equal(result[method, 5][field], result["A_current", 5][field])
        checks.append(dict(protocol=protocol, development_records=len(context["train"]), query_records=len(context["test"]),
            current_HGB_actions_reproduced=True, training_query_group_disjoint=True,
            time_alignment_by_record=True, candidate_groups_stay_in_same_crossfit=True,
            poisoned_query_labels_ignored=True, future_input_change_does_not_change_earlier_decisions=True,
            all_three_groups_identical_at_5=True))
    # New prediction can replace an earlier normal call; an older one can win too.
    index, score = choose(np.array([[.6, .02], [.01, .4], [.2, .2]]), np.array([[0, -1], [1, 0], [-1, 1]]))
    np.testing.assert_array_equal(index, [1, 0, 1])
    m = base.metrics(np.array([60., 75., 45., 60.]), np.array([45., 60., 75., 45.]), np.array([1, 1, 1, 0], bool))
    assert m["fn"] == 1 and m["fn_denominator"] == 2 and m["fp"] == 1 and m["fp_denominator"] == 1
    assert m["correct"] == 0 and m["withheld"] == 1
    # Test the transition accounting independently of any fitted output.
    synthetic = {}
    for method in METHODS:
        for t, p in [(5, [60., 75., 45., 60.]), (10, [45., 60., 45., 60.]), (15, [45., 75., 75., 45.])]:
            synthetic[method, t] = dict(prediction=np.array(p), action=np.ones(4, bool), source_time=np.full(4, t))
    diag, mask = trajectory(synthetic, np.array([45., 75., 75., 45.]), locked_baseline(synthetic))
    final = [d for d in diag if d["method"] == "C_current_risk" and d["minute"] == 15][0]
    assert mask.sum() == 2 and final["correct_direction_released"] == 2 and final["new_fn_outside_old_frozen_set"] == 0
    base.write(OUT / "SMOKE.json", dict(status="PASS", checks=checks, update_selection=True,
        fn_fpr_and_withholding_denominators=True, transition_counts=True,
        protocol_sha256=digest(OUT / "PROTOCOL.json"), code_sha256=digest(Path(__file__))))
    print("SMOKE_PASS", flush=True)


def evaluate(protocol):
    smoke_result = base.read(OUT / "SMOKE.json")
    assert smoke_result["code_sha256"] == digest(Path(__file__)), "Re-run smoke after code change"
    assert smoke_result["protocol_sha256"] == digest(OUT / "PROTOCOL.json")
    inputs = Contexts()
    records, diagnostics, details, thresholds = [], [], [], []
    for arm in ARMS:
        repeats = 20 if protocol == "paper" else 1
        test = inputs.load(ROOT / "gates" / f"{arm}_T900.npz")["test"] if protocol == "paper" else np.sort(np.r_[inputs.folds[0]["A"], inputs.folds[0]["B"]])
        locations = {int(v): i for i, v in enumerate(test)}
        shape = (repeats, len(test))
        aggregate = {(method, t): dict(prediction=np.full(shape, np.nan), action=np.zeros(shape, bool),
            score=np.full(shape, np.inf), source_time=np.full(shape, -1)) for method in METHODS for t in TIMES}
        seen = np.zeros(shape, int)
        for repeat in range(20 if protocol == "paper" else 5):
            context = inputs.build(protocol, arm, repeat)
            result, fits = decisions(context, f"{protocol}_{arm}_{repeat}")
            rep = repeat if protocol == "paper" else 0
            ix = [locations[int(v)] for v in context["test"]]
            seen[rep, ix] += 1
            for key, r in result.items():
                for field, array in r.items():
                    aggregate[key][field][rep, ix] = array
                details.append(dict(protocol=protocol, arm=arm, context=repeat, method=key[0], minute=key[1],
                    **base.metrics(r["prediction"], inputs.data["medical"][context["test"], 2], r["action"])))
            thresholds.extend(dict(protocol=protocol, arm=arm, context=repeat, **f) for f in fits)
            if repeat % 5 == 0:
                print("CONTEXT", protocol, arm, repeat, flush=True)
        assert (seen == 1).all()
        reference = inputs.data["medical"][test, 2]
        old = locked_baseline(aggregate)
        diag, frozen_fn = trajectory(aggregate, reference, old)
        if protocol == "paper" and arm == "HZ+HB_to_SA":
            assert int(frozen_fn.sum()) == 312, "Historical 312 FN were not reproduced"
        for t in TIMES:
            old_file = ROOT / "unified_hgb_20261105/private" / f"{protocol}_{arm}_agreement_target_0.85_{t}.npz"
            saved = inputs.load(old_file)
            np.testing.assert_array_equal(saved["test"], test)
            for field in ["prediction", "action", "source_time"]:
                np.testing.assert_array_equal(saved[field], old[t][field])
        for (method, t), r in aggregate.items():
            metric = base.metrics(r["prediction"], reference, r["action"])
            diagnosis = next(d for d in diag if d["method"] == method and d["minute"] == t)
            row = dict(protocol=protocol, arm=arm, method=method, minute=t, **metric,
                ever_reported=diagnosis["ever_reported"], using_earlier_prediction=diagnosis["using_earlier_prediction"],
                source_counts={str(s): int(np.sum(r["action"] & (r["source_time"] == s))) for s in TIMES})
            records.append(row)
            np.savez_compressed(PRIVATE / f"{protocol}_{arm}_{method}_{t}.npz", **r, test=test,
                                old_locked_fn_mask=frozen_fn)
        diagnostics.extend(dict(protocol=protocol, arm=arm, **r) for r in diag)
        print("ARM_COMPLETE", protocol, arm, flush=True)
    base.write(OUT / f"{protocol}_METRICS.json", records)
    base.write(OUT / f"{protocol}_FN_DIAGNOSTICS.json", diagnostics)
    base.write(PRIVATE / f"{protocol}_PER_CONTEXT.json", details)
    base.write(PRIVATE / f"{protocol}_THRESHOLDS.json", thresholds)
    columns = ["protocol", "arm", "method", "minute", "events", "reported", "correct", "incorrect", "withheld",
               "coverage", "agreement", "fn", "fn_denominator", "fnr", "fp", "fp_denominator", "fpr", "mae", "r2",
               "low_recall", "high_recall", "using_earlier_prediction", "ever_reported"]
    with (OUT / f"{protocol}_METRICS.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    base.write(OUT / f"{protocol}_INPUT_MANIFEST.json", base.INPUTS)
    base.write(OUT / "INPUT_CONTRACT.json", input_provenance())
    assert base.read(OUT / "PROTECTED_BEFORE.json") == protected(), "Protected manuscript assets changed"
    print("EVALUATION_COMPLETE", protocol, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["smoke", "paper", "internal301"])
    args = parser.parse_args()
    initialize()
    with threadpool_limits(limits=1):
        smoke() if args.stage == "smoke" else evaluate(args.stage)
