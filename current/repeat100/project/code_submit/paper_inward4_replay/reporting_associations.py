"""Aggregate exploratory MA-boundary and five-minute growth associations.

Reads existing predictions and HGB actions; does not fit models or select rules.
Outputs aggregate counts only, with the two evaluation protocols kept separate.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import numpy as np
import pandas as pd

ARMS = ['HZ_to_SA', 'HZ+HB_to_SA']


def state(x):
    return np.where(~np.isfinite(x), 9,
                    np.where(x < 50, -1, np.where(x <= 70, 0, 1)))


def aggregate(root):
    root = Path(root)
    hashes = {}

    def load(relative):
        path = root / relative
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}

    d = load('data/DATA15_CURRENT.npz')
    c = load('data/CURRENT_SCALED_CURVES.npz')
    np.testing.assert_array_equal(d['ids'], c['ids'])
    growth = np.full(len(d['medical']), np.nan)
    for i, (x, mask) in enumerate(zip(c['scaled_zeroed'][:, :300],
                                      c['observed_mask'][:, :300])):
        observed = x[mask]
        if len(observed) < 60:
            continue
        span = np.percentile(observed, 95) - np.percentile(observed, 5)
        if span > 0:
            growth[i] = (np.median(observed[-30:]) -
                         np.median(observed[-60:-30])) / span

    # Preserve the CSV-read precision used in the saved exploratory summary.
    # This keeps records exactly at a quantile boundary in their original group.
    # The intermediate is in memory only; no individual records are exported.
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(['late_relative_increment'])
    writer.writerows([[float(v)] for v in growth])
    buffer.seek(0)
    growth = pd.read_csv(buffer)['late_relative_increment'].to_numpy()

    boundary = []
    early = []
    cutpoints = {}
    for protocol in ['paper201', 'internal301']:
        folder = 'gates' if protocol == 'paper201' else 'all301_gates'
        for arm in ARMS:
            for horizon in [900, 300]:
                z = load(f'{folder}/{arm}_T{horizon}.npz')
                idx = z['test'] if protocol == 'paper201' else z['local']
                pred = np.atleast_2d(z['prediction'])
                action = np.atleast_2d(z['HGB_action'] if protocol == 'paper201' else z['HGB'])
                y = d['medical'][idx, 2]
                assert np.isfinite(pred).all() and np.isfinite(y).all()
                assert pred.shape == action.shape and pred.shape[1] == len(idx)
                assert np.array_equal(pred, np.broadcast_to(pred[0], pred.shape))
                error = state(pred) != state(y)
                unique_error = error[0]
                repeats = len(pred)
                expected = (201, 20) if protocol == 'paper201' else (301, 1)
                assert (len(idx), repeats) == expected
                base = dict(protocol=protocol, arm=arm, records=len(idx), repeats=repeats)
                if horizon == 900:
                    distance = np.minimum(abs(y - 50), abs(y - 70))
                    normal = state(y) == 0
                    boundary.append(dict(**base, errors=int(unique_error.sum()),
                        within_mm={str(t): int((unique_error & (distance <= t)).sum()) for t in [2, 3, 5]},
                        normal_near3_records=int((normal & (distance <= 3)).sum()),
                        normal_near3_errors=int((unique_error & normal & (distance <= 3)).sum()),
                        normal_far3_records=int((normal & (distance > 3)).sum()),
                        normal_far3_errors=int((unique_error & normal & (distance > 3)).sum())))
                    continue

                normal = state(y) == 0
                g = growth[idx]
                assert np.isfinite(g[normal]).all()
                normal_bands, edges = pd.qcut(g[normal], 3, labels=False, retbins=True)
                assert np.all(np.diff(edges) > 0)
                if protocol in cutpoints:
                    np.testing.assert_array_equal(cutpoints[protocol], edges)
                else:
                    cutpoints[protocol] = edges.tolist()
                bands = np.full(len(idx), -1, dtype=int)
                bands[normal] = normal_bands
                for b, label in enumerate(['Lower', 'Middle', 'Upper']):
                    m = normal & (bands == b)
                    n = int(m.sum())
                    total = repeats * n
                    wrong = int(error[:, m].sum())
                    withheld = int((~action[:, m]).sum())
                    early.append(dict(**base, growth_third=label, normal_records=n,
                        error_records=int(unique_error[m].sum()), error_events=wrong,
                        withheld_events=withheld, total_events=total,
                        error_rate=wrong/total, withholding_rate=withheld/total))

    return dict(
        table='Supplementary Table S6b',
        scope='Post hoc descriptive associations; original predictions and HGB actions are unchanged.',
        protocols=dict(paper201='201 held-out records, 20 development partitions; repeats are not independent patients.',
                       internal301='301 records, one outer-held-out prediction per record; overlaps paper201.'),
        boundary_definition='Before selection: min(abs(reference MA - 50), abs(reference MA - 70)); counts are records, each once.',
        growth_definition='First 300 time points only. Among observed points, subtract median of preceding 30 from median of last 30; divide by observed-prefix p95 minus p5.',
        growth_group_definition='Reference-normal records only (50 <= MA <= 70). Separate cohort-specific thirds; right-inclusive quantile boundaries.',
        interpretation='Growth was not a direct HGB input. No mechanism or incremental predictive value is established by these associations.',
        growth_cutpoints=cutpoints,
        boundary_15min=boundary,
        growth_5min=early,
        input_sha256=hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path,
                        default=Path(__file__).resolve().parents[2] / 'outputs/current_ma_20261103')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).with_name('reporting_associations_aggregate.json'))
    args = parser.parse_args()
    result = aggregate(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print('AGGREGATED', len(result['boundary_15min']), 'boundary rows and',
          len(result['growth_5min']), 'growth rows; no models fitted')


if __name__ == '__main__':
    main()
