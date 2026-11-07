"""Matched operating-point preview, not a locked first-report replay.

Prediction choice and HGB ranking do not use evaluation labels. Matching observed
agreement is a retrospective readout of the risk/coverage curve, using linear
interpolation of COUNTS, not a newly validated deployment threshold.
"""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from . import history_update_comparison as h

base = h.base
OUT = base.ROOT / "figure4_matched_preview_20261105"
QUOTAS = [.10, .20, .40]
AGREEMENTS = [.85, .90, .95]
COLORS = [["#396B91", "#C1843E", "#4B866D"], ["#8064A2", "#A06072", "#60656F"]]
MARKERS = [["o", "s", "^"], ["D", "v", "P"]]
FIELDS = ["reported", "correct", "fn", "true_abnormal", "fp", "true_normal", "pred_low", "pred_normal", "pred_high", "from5", "from10", "from15"]


def initialize():
    OUT.mkdir(parents=True, exist_ok=True)
    contract = dict(
        purpose="Compare agreement/FNR at equal report counts and coverage/FNR at equal observed agreement; use current15 predictions as the terminal reference.",
        archetype="quantitative grid; a/b primary operating-point comparisons, c/d corresponding false-negative rates",
        backend="Python/matplotlib, Arial, editable SVG/PDF, 183x145mm main preview; separate compact Fig1 inset preview",
        cohort="Original60-development/201-evaluation, dual source,20 development partitions,4020 record-seed events. No independent-patient interpretation of repeated events.",
        choices="5=current5; 10=current10 except when current10 fails the original development HGB gate and5 passes, then use5; 15=current15 for EVERY record. No labels determine this choice.",
        risk="At5 and15 rank current HGB scores. At10 use development-fitted monotone isotonic calibration of the chosen source-time score to compare retained5/current10. Stable record-ID tie break. Fixed predicted candidate for all four panels.",
        a_c="Choose floor(q*201) per partition at q=10/20/40%. Identical report selection for agreement and FNR. Values are nominal coverage; actual is20/201,40/201,80/201.",
        b_d="For each time build the pooled curve from k=1..201 equal-count-per-partition prefixes. For each observed agreement85/90/95%, take the largest attainable coverage and interpolate between adjacent rank-prefix count vectors. Agreement is matched exactly in interpolated aggregate counts. FNR is interpolated FN divided by interpolated released-abnormal count. Identical operating point for b/d.",
        evaluation_labels="Used for outcome metrics and retrospective agreement matching ONLY. No fitting of mappings/HGB/calibration or selection of deployable cutoffs with evaluation labels. These matched points do not validate an operational agreement guarantee.",
        endpoint="Same15-minute predictions and HGB scores as Table2, with operating-point-specific selection. Not the same single Table2 cutoff for every matched line. Compute FN counts; never overwrite endpoints with zeros.",
        rate="FNR=FN/released reference abnormal; undefined if denominator0. No replacement of undefined with0. Store numerator and denominator for every point.",
        monotonicity="No monotonic envelopes or test-label choice between times; observed curves can rise or fall.",
        integrity="Frozen inputs retain prior validation flags. No source values changed; this preview does not certify clinical validity.",
        scope="New review previews and source data only; manuscript/PPT/current figures/PDF/packages unchanged.")
    if (OUT / "CONTRACT.json").exists():
        assert base.read(OUT / "CONTRACT.json") == contract
    else:
        base.write(OUT / "CONTRACT.json", contract)
        base.write(OUT / "PROTECTED_BEFORE.json", h.protected())


def choose_candidate(context, minute):
    q, d = context["query"], context["dev"]
    p, score = q[minute]["prediction"].copy(), q[minute]["score"].copy()
    source = np.full(len(p), minute)
    if minute == 10:
        a5 = base.apply(q[5]["score"], q[5]["prediction"], base.threshold(d[5], .85))
        a10 = base.apply(q[10]["score"], q[10]["prediction"], base.threshold(d[10], .85))
        use5 = a5 & ~a10
        p[use5] = q[5]["prediction"][use5]
        source[use5] = 5
        score = np.where(use5, base.calibrated_score(d[5], q[5]["score"]), base.calibrated_score(d[10], q[10]["score"]))
    return p, score, source


def matched_counts(counts, target):
    agreement = counts[:, 1] / counts[:, 0]
    attainable = np.flatnonzero(agreement+1e-12 >= target)
    if not len(attainable):
        return None, None
    i = int(attainable[-1])
    if abs(agreement[i]-target) < 1e-12:
        return counts[i].copy(), dict(lower_k=i+1, upper_k=i+1, fraction=0.)
    if i == len(counts)-1:
        # No lower-agreement point exists; do not falsely call it exact matching.
        return None, dict(reason="full_coverage_above_target_cannot_match_exactly")
    delta = counts[i+1]-counts[i]
    fraction = (target*counts[i, 0]-counts[i, 1])/(delta[1]-target*delta[0])
    assert -1e-10 <= fraction <= 1+1e-10
    result = counts[i]+fraction*delta
    assert np.isclose(result[1]/result[0], target, atol=1e-12)
    return result, dict(lower_k=i+1, upper_k=i+2, fraction=float(fraction))


def row(counts, minute, condition, level, interpolation=None):
    values = dict(zip(FIELDS, counts.tolist()))
    ratio = lambda a, b: a/b if b else None
    return dict(minute=minute, condition=condition, level=level, **values,
        coverage=values["reported"]/4020, agreement=ratio(values["correct"], values["reported"]),
        fnr=ratio(values["fn"], values["true_abnormal"]),
        fpr=ratio(values["fp"], values["true_normal"]), interpolation=interpolation)


def compute():
    inputs = h.Contexts()
    counts = {t: np.zeros((201, len(FIELDS))) for t in h.TIMES}
    for rep in range(20):
        context = inputs.build("paper", "HZ+HB_to_SA", rep)
        for t in h.TIMES:
            p, score, source = choose_candidate(context, t)
            # Selection/ranking precede reading reference labels.
            assert np.isfinite(p).all() and np.isfinite(score).all()
            order = np.lexsort((inputs.data["ids"][context["test"]], score))
            if rep == 0:
                poison = {**context, "query": {tt: {**q, "reference": np.full(201, 9999.)} for tt, q in context["query"].items()}}
                for actual, other in zip((p, score, source), choose_candidate(poison, t)):
                    np.testing.assert_array_equal(actual, other)
                changed = {**context, "query": {**context["query"], 15: dict(prediction=np.full(201, 0.), score=np.zeros(201))}}
                if t < 15:
                    for actual, other in zip((p, score, source), choose_candidate(changed, t)):
                        np.testing.assert_array_equal(actual, other)
            pc = base.state(p)[order]
            truth = base.state(inputs.data["medical"][context["test"], 2])[order]
            origin = source[order]
            events = np.column_stack([np.ones(201), pc == truth, (pc == 0) & (truth != 0), truth != 0,
                (pc != 0) & (truth == 0), truth == 0, pc == -1, pc == 0, pc == 1,
                origin == 5, origin == 10, origin == 15])
            counts[t] += np.cumsum(events, axis=0)
    points, curves = [], []
    for t in h.TIMES:
        np.testing.assert_array_equal(counts[t][:, 0], 20*np.arange(1, 202))
        np.testing.assert_allclose(counts[t][:, 3]+counts[t][:, 5], counts[t][:, 0])
        for q in QUOTAS:
            k = int(np.floor(201*q))
            points.append(row(counts[t][k-1], t, "coverage", q))
        for target in AGREEMENTS:
            c, meta = matched_counts(counts[t], target)
            assert c is not None, (t, target, meta)
            points.append(row(c, t, "agreement", target, meta))
        curves.extend(row(c, t, "rank_prefix", (i+1)/201) for i, c in enumerate(counts[t]))
    terminal = [r for r in points if r["minute"] == 15]
    assert all(r["fn"] == 0 and r["true_abnormal"] > 0 for r in terminal)
    # Synthetic exact agreement check: C=38 ->51, n=40 ->60, target .90 =>lambda=.4.
    toy = np.array([[20., 20., 0.], [40., 38., 0.], [60., 51., 6.]])
    value, meta = matched_counts(toy, .9)
    assert np.isclose(meta["fraction"], .4) and np.isclose(value[1]/value[0], .9)
    base.write(OUT / "POINTS.json", points)
    base.write(OUT / "RANK_CURVES.json", curves)
    columns = [k for k in points[0] if k != "interpolation"]
    with (OUT / "SOURCE_DATA.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(points)
    base.write(OUT / "INPUT_MANIFEST.json", base.INPUTS)
    return points


def styling():
    plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7.5,
        "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "svg.fonttype": "none", "pdf.fonttype": 42, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": .7, "legend.frameon": False})


def plot(points):
    styling()
    lookup = {(r["condition"], r["level"], r["minute"]): r for r in points}
    fig, axes = plt.subplots(2, 2, figsize=(183/25.4, 145/25.4))
    panels = [("a", "coverage", "agreement", "Agreement at matched coverage", "Agreement (%)"),
              ("b", "agreement", "coverage", "Coverage at matched agreement", "Coverage (%)"),
              ("c", "coverage", "fnr", "FNR at matched coverage", "FNR (%)"),
              ("d", "agreement", "fnr", "FNR at matched agreement", "FNR (%)")]
    for ax, (letter, condition, metric, title, ylabel) in zip(axes.flat, panels):
        col = 0 if condition == "coverage" else 1
        levels = QUOTAS if col == 0 else AGREEMENTS
        for i, level in enumerate(levels):
            values = [100*lookup[condition, level, t][metric] if lookup[condition, level, t][metric] is not None else np.nan for t in h.TIMES]
            ax.plot(h.TIMES, values, color=COLORS[col][i], marker=MARKERS[col][i],
                markersize=5.6-i*.6, markerfacecolor="none", markeredgewidth=1., lw=1.2)
        ax.set_title(title, loc="left", pad=8)
        ax.text(-.15, 1.07, letter, transform=ax.transAxes, fontsize=9, fontweight="bold")
        ax.set(xlim=(4.3, 15.7), xticks=h.TIMES, xlabel="Observation time (min)", ylabel=ylabel)
        if metric == "agreement":
            ax.set(ylim=(80, 102), yticks=[80, 85, 90, 95, 100])
        else:
            ax.set(ylim=(-3, 105), yticks=[0, 20, 40, 60, 80, 100])
        ax.grid(axis="y", color="#E3E7EA", lw=.5)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, width=.6)
    for col, (levels, label) in enumerate([(QUOTAS, "Fixed coverage (a, c)"), (AGREEMENTS, "Matched agreement (b, d)")]):
        handles = [Line2D([0], [0], color=COLORS[col][i], marker=MARKERS[col][i],
            markerfacecolor="none", markersize=4, lw=1.2, label=f"{level*100:g}%") for i, level in enumerate(levels)]
        fig.legend(handles=handles, title=label, loc="upper center", bbox_to_anchor=(.292+.5*col, .963),
                   ncol=3, fontsize=7, title_fontsize=7.5, columnspacing=1.2, handlelength=1.4)
    fig.subplots_adjust(left=.1, right=.98, top=.81, bottom=.10, hspace=.48, wspace=.3)
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(OUT / f"Figure4_matched_preview.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(125/25.4, 39/25.4))
    for ax, condition, level, metric, title, color in [
        (axes[0], "coverage", .4, "agreement", "At 40% coverage", COLORS[0][2]),
        (axes[1], "agreement", .9, "coverage", "At 90% agreement", COLORS[1][1])]:
        values = [100*lookup[condition, level, t][metric] for t in h.TIMES]
        ax.plot(h.TIMES, values, color=color, marker="o", lw=1.4, markersize=3.5)
        ax.set_title(title, loc="left", pad=4)
        ax.set(xlim=(4.4, 15.6), xticks=h.TIMES, xlabel="Time (min)", ylabel=metric.capitalize()+" (%)")
        ax.set(ylim=(80, 102), yticks=[80, 90, 100]) if metric == "agreement" else ax.set(ylim=(-2, 102), yticks=[0, 50, 100])
        ax.grid(axis="y", color="#E3E7EA", lw=.45)
        ax.set_axisbelow(True)
    fig.subplots_adjust(left=.09, right=.99, top=.80, bottom=.28, wspace=.40)
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(OUT / f"Figure1e_inset_preview.{ext}", dpi=300, facecolor="white")
    plt.close(fig)


def main():
    initialize()
    points = compute()
    plot(points)
    base.write(OUT / "QA.json", dict(source_points=len(points), paired_a_c_and_b_d_same_counts=True,
        all_15min_fnr_zero_from_computed_counts=True, matching_is_retrospective_interpolation=True,
        no_test_labels_in_candidate_choice_or_ranking=True, no_future15_fields_in_early_choices=True,
        no_monotonicity_enforcement=True, original_manuscript_unchanged=base.read(OUT / "PROTECTED_BEFORE.json") == h.protected()))
    for r in points:
        print({k: r[k] for k in ["minute", "condition", "level", "coverage", "agreement", "fn", "true_abnormal", "fnr"]})


if __name__ == "__main__":
    main()
