"""Development-only signal neighborhoods and explicit curve kinetics.

No clinical claim is assigned to a derived feature. Distances diagnose the
chosen representations, not irreducible error or impossibility of prediction.
"""
import json
import numpy as np
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from .search import ROOT
from .representation_search import load_representations
from .functional_search import load_waveforms
from .metrics import states
from code_submit.ma_imbalance_mapping.data_contract import load_development_inputs, sha256

FEATURE_NAMES = ['end_amplitude', 'peak_amplitude', 'mean_amplitude',
                 'time_10pct', 'time_50pct', 'time_90pct', 'rise_10_to_90',
                 'peak_growth_per_min', 'time_peak_growth', 'tail_growth_per_min',
                 'amplitude_5min_relative', 'amplitude_10min_relative',
                 'mean_relative', 'tail_to_peak_growth', 'monotone_fraction',
                 'roughness_relative', 'decline_relative']


def kinetics(wave):
    wave = np.asarray(wave, float)
    if wave.shape != (90,) or not np.isfinite(wave).all():
        raise ValueError('Expected 90 observed ten-second medians')
    smooth = savgol_filter(wave, 9, 2, mode='interp')
    end = float(np.median(wave[-3:])); peak = float(np.max(smooth))
    amplitude = max(abs(end), 1e-6)
    times = []
    for fraction in (.1, .5, .9):
        above = smooth >= fraction * end
        hits = np.flatnonzero(np.convolve(above.astype(int), np.ones(3, int), mode='valid') == 3)
        times.append(float(hits[0] * 10 + 5) if len(hits) and end > 1e-6 else 900.)
    slope = np.gradient(smooth, 10) * 60
    # Exclude filter edge derivatives from the peak-growth estimate.
    j = int(np.argmax(slope[4:-4])) + 4
    peak_growth = float(slope[j]); tail_growth = float((np.median(wave[-3:])-np.median(wave[57:60])) / 5.)
    values = [end, peak, float(np.mean(wave)), *times, times[2]-times[0],
              peak_growth, j*10+5, tail_growth,
              float(np.median(wave[27:30]))/amplitude,
              float(np.median(wave[57:60]))/amplitude,
              float(np.mean(wave))/amplitude,
              tail_growth/max(abs(peak_growth), 1e-6),
              float(np.mean(np.diff(smooth)>=0)),
              float(np.sqrt(np.mean((wave-smooth)**2)))/amplitude,
              (peak-end)/max(abs(peak), 1e-6)]
    result = np.asarray(values, float)
    assert len(result) == len(FEATURE_NAMES) and np.isfinite(result).all()
    return result


def load_inputs(data, parts):
    original, more = load_representations(data, parts)
    raw, shape, wave_hashes = load_waveforms(data, parts); more.update(wave_hashes)
    pool = next(iter(parts.values()))['pool']
    kinetic = np.full((len(data['x']), len(FEATURE_NAMES)), np.nan)
    for i in pool: kinetic[i] = kinetics(raw[i])
    matrices = {'parameters4': original['parameters4'], 'curve90': raw,
                'shape90': shape, 'kinetics17': kinetic,
                'parameters_plus_kinetics': np.c_[original['parameters4'], kinetic]}
    return matrices, more


def distances(train, query, kind):
    if kind in ('curve90', 'shape90'):
        # One global amplitude scale; no label-dependent normalization.
        mean = train.mean(0); scale = max(float(train.std()), 1e-6)
        a, b = (train-mean)/scale, (query-mean)/scale
    else:
        mean = train.mean(0); scale = np.maximum(train.std(0), 1e-6)
        a, b = (train-mean)/scale, (query-mean)/scale
    return np.sqrt(np.mean((a-b)**2, axis=1))


def run():
    out = ROOT/'outputs/uteg_mapping_target_runs/goal_signal_neighborhood_v1'
    out.mkdir(parents=True, exist_ok=True); (out/'private').mkdir(exist_ok=True)
    cfg = json.loads((ROOT/'code_submit/ma_imbalance_mapping/config.json').read_text(encoding='utf-8')); cfg['seeds'] = [1000]
    data, parts, hashes = load_development_inputs(ROOT, cfg); matrices, more = load_inputs(data, parts); hashes.update(more)
    pool = next(iter(parts.values()))['pool']; labels = states(data['y'][pool, 2]); rows = []
    rng = np.random.default_rng(20261102)
    for rep, values in matrices.items():
        for i, index in enumerate(pool):
            eligible = data['groups'][pool] != data['groups'][index]
            training = pool[eligible]; train_labels = labels[eligible]
            d = distances(values[training], values[index], rep); order = np.argsort(d)
            k = int(labels[i]); same = train_labels == k; normal = train_labels == 1
            entry = {'row': int(index), 'representation': rep, 'true_class': k,
                     'reference_ma': float(data['y'][index, 2]),
                     'nearest_row': int(training[order[0]]), 'nearest_class': int(train_labels[order[0]]),
                     'nearest_reference_ma': float(data['y'][training[order[0]], 2]),
                     'nearest_distance': float(d[order[0]]),
                     'same_class_min_distance': float(d[same].min()) if same.any() else None,
                     'normal_min_distance': float(d[normal].min()),
                     'nearest5_class_counts': np.bincount(train_labels[order[:5]], minlength=3).tolist()}
            if k != 1:
                count = min(3, int(same.sum()), int(normal.sum())); wins = []
                for _ in range(200):
                    a = rng.choice(np.flatnonzero(same), count, replace=False)
                    b = rng.choice(np.flatnonzero(normal), count, replace=False)
                    wins.append(float(d[b].min() < d[a].min()))
                entry['normal_closer_balanced_reference_fraction'] = float(np.mean(wins))
            rows.append(entry)
    summary = {'scope': 'representation diagnostic, not a classifier, not a proof of insufficient signal',
               'records': len(pool), 'counts': np.bincount(labels, minlength=3).tolist(), 'representations': {}}
    for rep in matrices:
        by_class = {}
        for k, name in enumerate(['low', 'within', 'high']):
            r = [v for v in rows if v['representation']==rep and v['true_class']==k]
            by_class[name] = {'n': len(r), 'nearest_same_class': sum(v['nearest_class']==k for v in r),
                              'nearest_normal': sum(v['nearest_class']==1 for v in r),
                              'median_nearest_ma_gap': float(np.median([abs(v['nearest_reference_ma']-v['reference_ma']) for v in r]))}
            if k != 1:
                by_class[name]['median_balanced_normal_closer_fraction'] = float(np.median([v['normal_closer_balanced_reference_fraction'] for v in r]))
        summary['representations'][rep] = by_class
    (out/'PROTOCOL.json').write_text(json.dumps({'input_hashes': hashes, 'code_hash': sha256(__file__),
        'distance_scale': 'fitted separately on other-group permitted development records only',
        'query_label': 'used only for descriptive nearest-class accounting, never distance/feature fitting',
        'balanced_reference': '200 repeat samples of up to3 train records from own class and normal class; diagnostic only',
        'kinetics_features': FEATURE_NAMES}, indent=2), encoding='utf-8')
    (out/'private/RECORD_NEIGHBORHOODS.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    (out/'SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': run()
