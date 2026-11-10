"""Render Figure 3 using frozen aggregate curves and report destinations."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

WS = Path(__file__).resolve().parents[2]
OUT = WS / 'outputs/current_ma_20261103/manuscript_sync/deliverables/figures/newB_20261103'
SOURCE = Path(__file__).with_name('figure3_reporting_aggregate.json')
ARMS = ['HZ_to_SA', 'HZ+HB_to_SA']
RULES = ['ALL_REPORT', 'BOUNDARY_PAPER', 'LR', 'HGB']
COLORS = {'BOUNDARY': '#3D6F98', 'LR': '#378579', 'HGB': '#BD7C32'}
STYLES = {'BOUNDARY': '--', 'LR': '-', 'HGB': '-.'}
LABELS = {'BOUNDARY': 'Boundary distance', 'LR': 'Logistic', 'HGB': 'HGB'}


def run(output_dir=None):
    out = Path(output_dir) if output_dir is not None else OUT
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    data = json.loads(SOURCE.read_text(encoding='utf-8'))
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        'font.family': 'Arial', 'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .65, 'legend.frameon': False,
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
    })
    fig = plt.figure(figsize=(183 / 25.4, 160 / 25.4))
    grid = fig.add_gridspec(2, 2, left=.17, right=.97, bottom=.20,
                          top=.835, wspace=.24, hspace=.50,
                          height_ratios=[1, 1.28])
    curve_rows, retention_rows, point_rows, cost_rows, flow_rows = [], [], [], [], []
    for col, arm in enumerate(ARMS):
        ax = fig.add_subplot(grid[0, col])
        for rule in ['BOUNDARY', 'LR', 'HGB']:
            rr = sorted([r for r in data['risk_curves'][arm] if r['rule'] == rule], key=lambda r: r['coverage'])
            assert len(rr) == 20
            x = np.array([100 * r['coverage'] for r in rr])
            y = np.array([100 * r['agreement']['estimate'] for r in rr])
            lo = np.array([100 * r['agreement']['low'] for r in rr])
            hi = np.array([100 * r['agreement']['high'] for r in rr])
            assert np.all(lo <= y) and np.all(y <= hi)
            ax.fill_between(x, lo, hi, color=COLORS[rule], alpha=.085, lw=0)
            ax.plot(x, y, color=COLORS[rule], ls=STYLES[rule], lw=1.35)
            for r in rr:
                curve_rows.append(dict(arm=arm, rule=rule, quota=r['quota'], coverage=r['coverage'], **r['agreement']))
        metrics = data['reporting'][arm]
        # Actual threshold points can differ from the matched-count ranking curves.
        for rule in RULES:
            m = metrics[rule]
            point_rows.append(dict(arm=arm, rule=rule, coverage=m['coverage'], agreement=m['agreement']))
            if rule != 'ALL_REPORT':
                color_key = 'BOUNDARY' if rule == 'BOUNDARY_PAPER' else rule
                ax.scatter(100 * m['coverage'], 100 * m['agreement'], s=31,
                           color=COLORS[color_key], edgecolor='white', linewidth=.65,
                           zorder=7)
        base = metrics['ALL_REPORT']
        ax.scatter(100, 100 * base['agreement'], marker='D', s=19,
                   color='#333333', zorder=6, clip_on=False)
        ax.set(xlim=(50, 102), ylim=(76, 102), xticks=[50, 60, 70, 80, 90, 100],
               yticks=[80, 85, 90, 95, 100], xlabel='Report coverage (%)')
        if col == 0:
            ax.set_ylabel('MA-state agreement (%)')
        ax.grid(axis='y', color='#DFE5E9', lw=.45)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, width=.6)
        ax.set_title(['Single source', 'Dual source'][col], loc='left', fontsize=8, pad=9)
        ax.text(-.13, 1.06, 'ab'[col], transform=ax.transAxes, fontsize=9, fontweight='bold')

        position = grid[1, col].get_position(fig)
        lower = fig.add_gridspec(2, 1, left=position.x0, right=position.x1,
                                top=position.y1, bottom=.145, hspace=.70)
        cm = np.asarray(base['confusion'])
        correct = int(np.trace(cm))
        errors = int(cm.sum() - correct)
        den = base['all_true_class_counts']
        assert (correct, errors, den) == (3380, 640, [140, 3400, 480])
        assert (int(cm[0, 0]), int(cm[2, 2])) == (den[0], den[2])
        matrix = np.zeros((4, len(RULES)))
        for j, rule in enumerate(RULES):
            m = metrics[rule]
            cm = np.asarray(m['confusion'])
            kept = int(np.trace(cm))
            err = int(cm.sum() - kept)
            assert kept + err + sum(m['withheld_by_true_state']) == 4020
            values = [kept, errors - err, int(cm[0, 0]), int(cm[2, 2])]
            denominators = [correct, errors, den[0], den[2]]
            names = ['correct_reports_retained', 'errors_withheld', 'correct_low_retained', 'correct_high_retained']
            for i, (name, n, d) in enumerate(zip(names, values, denominators)):
                matrix[i, j] = 100 * n / d
                retention_rows.append(dict(arm=arm, rule=rule, metric=name, numerator=n,
                                           denominator=d, fraction=n / d, percent=100 * n / d))
        # Preserve the existing retention and complementary-percentage exports.
        withheld = 100 - matrix[[0, 2, 3]]
        assert np.allclose(withheld[:, 0], 0)
        for j, rule in enumerate(RULES):
            cm = np.asarray(metrics[rule]['confusion'])
            kept_counts = [int(np.trace(cm)), int(cm[0, 0]), int(cm[2, 2])]
            for i, name in enumerate(['correct_reports_withheld', 'correct_low_withheld', 'correct_high_withheld']):
                d = [correct, den[0], den[2]][i]
                n = d - kept_counts[i]
                assert np.isclose(withheld[i, j], 100 * n / d)
                cost_rows.append(dict(arm=arm, rule=rule, metric=name, numerator=n,
                                      denominator=d, fraction=n / d, percent=100 * n / d))
        # In these 15-minute comparisons, every pre-gate error is an FP.
        for rule in RULES:
            cm = np.asarray(metrics[rule]['confusion'])
            assert cm[0, 1] + cm[2, 1] == 0  # no abnormal-to-normal errors
            assert cm[0, 2] + cm[2, 0] == 0  # no low/high swaps
        bar_colors = ['#9AA6AF', COLORS['BOUNDARY'], COLORS['LR'], COLORS['HGB']]
        for group, (state, total) in enumerate([('incorrect', errors), ('correct', correct)]):
            split = fig.add_subplot(lower[group])
            for j, rule in enumerate(RULES):
                cm = np.asarray(metrics[rule]['confusion'])
                released = int(cm.sum() - np.trace(cm)) if state == 'incorrect' else int(np.trace(cm))
                held = total - released
                assert released >= 0 and held >= 0 and released + held == total
                count = held if group == 0 else released
                pct = 100 * count / total
                split.bar(j, pct, width=.62, color=bar_colors[j], zorder=3)
                split.text(j, pct + 2, f'{count:,}\n{pct:.1f}%', ha='center', va='bottom', fontsize=6)
                for outcome, n in [('released', released), ('withheld', held)]:
                    flow_rows.append(dict(arm=arm, rule=rule, initial_prediction=state,
                                          outcome=outcome, count=n, denominator=total,
                                          fraction=n / total))
            split.set(xlim=(-.65, 3.65), ylim=(0, 125 if group else 85),
                      xticks=range(4), yticks=[0, 50, 100] if group else [0, 25, 50, 75])
            split.set_xticklabels(['All', 'Boundary', 'Logistic', 'HGB'], fontsize=6.5)
            split.tick_params(length=2.5, width=.6)
            split.grid(axis='y', color='#DFE5E9', lw=.45)
            split.set_axisbelow(True)
            split.set_ylabel('FP withheld (%)' if group == 0 else 'Correct retained (%)', fontsize=6.7)
            split.set_title('Before gating: 640 FP' if group == 0 else 'Before gating: 3,380 correct (TP + TN)',
                            loc='left', fontsize=6.8, pad=5)
            if group == 0:
                split.text(-.13, 1.18, 'cd'[col], transform=split.transAxes,
                           fontsize=9, fontweight='bold')
            else:
                split.set_xticklabels([f'{label}\n{100*metrics[rule]["coverage"]:.2f}%'
                                       for rule,label in zip(RULES,['All','Boundary','Logistic','HGB'])], fontsize=6.3)
                split.set_xlabel('Reporting rule / coverage', fontsize=6.5, labelpad=4)

    fig.suptitle('Selective reporting at 15 minutes', fontsize=10, fontweight='bold', y=.982)
    handles = [Line2D([0], [0], color=COLORS[k], ls=STYLES[k], lw=1.5, label=LABELS[k])
               for k in ['BOUNDARY', 'LR', 'HGB']]
    handles.append(Line2D([0], [0], marker='D', color='#333333', lw=0, markersize=3.5, label='All report'))
    handles.append(Line2D([0], [0], marker='o', color='#666666', lw=0,
                          markersize=4, label='Table 2 operating point'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.56, .916), ncol=5,
               fontsize=6.4, handlelength=2.2, columnspacing=1.05)
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(out / f'Figure3_reporting.{ext}', dpi=400, facecolor='white')
    fig.savefig(out / 'Figure3_reporting.tiff', dpi=600, facecolor='white',
                pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    for name, rows in [('curves', curve_rows), ('retention', retention_rows), ('working_points', point_rows), ('withholding', cost_rows), ('release_counts', flow_rows)]:
        with (out / f'Figure3_{name}.csv').open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    qa = dict(source_sha256=source_hash, curve_points=len(curve_rows),
              retention_cells=len(retention_rows), no_refitting=True,
              dimensions_mm=[183, 160], main_coverage_view=[50, 100],
              all_curve_quotas_retained=[.05, 1.0], displayed_rules=RULES,
              threshold_callouts=True, threshold_points='Actual Table 2 pooled counts, not interpolated curve points',
              lower_layout='Vertical bars: FP withheld / 640 pre-gate FP and correct retained / 3380 pre-gate correct events (TP + TN)',
              release_count_rows=len(flow_rows), low_high_subgroups='Table 2',
              boundary95_preserved_in_source=True, visual_qa='pending')
    (out / 'Figure3_QA.json').write_text(json.dumps(qa, indent=2) + '\n', encoding='utf-8')
    print('FIGURE3_COMPLETE', flush=True)


if __name__ == '__main__':
    run()
