"""Train-only source feature alignment for continuous MA regression.

The CORAL covariance transform follows Sun, Feng & Saenko (AAAI 2016).
Mean matching, labeled target co-training, center loss masses and the two
regressors are this project's evaluation choices, not claims from that paper.
No target validation/test features enter alignment (inductive adaptation).
"""
from dataclasses import dataclass
import hashlib
import itertools
import json
import time
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from .models import weights
from .metrics import assess
from .search import ROOT, make_summary
from .representation_search import load_representations
from code_submit.ma_imbalance_mapping.data_contract import load_development_inputs, sha256


def matrix_power(cov, power):
    values, vectors = np.linalg.eigh(cov)
    if np.min(values) <= 0:
        raise ValueError('Covariance must be positive definite')
    return (vectors * values ** power) @ vectors.T


def align_source(source, target, method):
    sm, tm = source.mean(0), target.mean(0)
    if method == 'center_scale':
        transform = np.diag(np.maximum(target.std(0), 1e-6) /
                            np.maximum(source.std(0), 1e-6))
    elif method == 'coral':
        # The paper's +I regularization in globally standardized coordinates.
        cs = np.cov(source, rowvar=False) + np.eye(source.shape[1])
        ct = np.cov(target, rowvar=False) + np.eye(target.shape[1])
        transform = matrix_power(cs, -.5) @ matrix_power(ct, .5)
    else:
        raise ValueError(method)
    result = (source - sm) @ transform + tm
    return result, {'source_mean': sm.tolist(), 'target_mean': tm.tolist(),
                    'transform': transform.tolist()}


@dataclass
class AlignedModel:
    mean: np.ndarray
    scale: np.ndarray
    estimator: object
    diagnostics: dict

    def predict(self, x):
        return np.maximum(60 + 10 * self.estimator.predict((x - self.mean) / self.scale), 0)


def fit_arrays(target_x, target_y, sources, candidate):
    """Only training arrays are accepted; no evaluation arrays or full data."""
    arrays = [target_x] + [v[0] for v in sources.values()]
    all_x = np.vstack(arrays)
    mean, scale = all_x.mean(0), np.maximum(all_x.std(0), 1e-6)
    target_z = (target_x - mean) / scale
    blocks, labels = [target_z], [target_y]
    masses = [4. * weights(target_y, 20.) / len(target_y)]
    alignments = {}
    for name, (x, y) in sorted(sources.items()):
        z, diag = align_source((x - mean) / scale, target_z, candidate['alignment'])
        blocks.append(z); labels.append(y)
        masses.append(weights(y, 20.) / len(y)); alignments[name] = diag
    x = np.vstack(blocks); y = np.concatenate(labels); w = np.concatenate(masses)
    w /= w.sum()
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Non-finite training arrays')
    if candidate['learner'] == 'kernel':
        # Sum-normalized weights give the same loss scale for different arms.
        estimator = KernelRidge(alpha=.01, kernel='rbf', gamma=1. / x.shape[1])
        estimator.fit(x, (y - 60) / 10, sample_weight=w)
    elif candidate['learner'] == 'tree':
        estimator = HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=7,
                    min_samples_leaf=1, l2_regularization=1., learning_rate=.08,
                    early_stopping=False, random_state=20261101)
        estimator.fit(x, (y - 60) / 10, sample_weight=w / w.mean())
    else:
        raise ValueError(candidate['learner'])
    diag = {'n_train': len(y), 'alignments': alignments,
            'training_data_hash': hashlib.sha256(x.tobytes() + y.tobytes() + w.tobytes()).hexdigest()}
    return AlignedModel(mean, scale, estimator, diag)


def fit(data, A, candidate):
    source_rows = data['source_rows'].get(candidate['arm'], np.array([], int))
    sources = {str(name): (data['x'][source_rows[data['centers'][source_rows] == name]],
                          data['y'][source_rows[data['centers'][source_rows] == name], 2])
               for name in np.unique(data['centers'][source_rows])}
    return fit_arrays(data['x'][A], data['y'][A, 2], sources, candidate)


def grid():
    result = []
    for arm, representation, learner in itertools.product(
            ['target_only', 'HZ_to_SA', 'HZ+HB_to_SA'],
            ['parameters4', 'parameters_plus_curve'], ['kernel', 'tree']):
        for alignment in (['none'] if arm == 'target_only' else ['center_scale', 'coral']):
            c = dict(family='source_alignment', arm=arm, representation=representation,
                     learner=learner, alignment=alignment)
            c['id'] = hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:12]
            result.append(c)
    return result


def run():
    out = ROOT / 'outputs/uteg_mapping_target_runs/goal_alignment_v1'
    out.mkdir(parents=True, exist_ok=True); (out / 'private').mkdir(exist_ok=True)
    cfg = json.loads((ROOT / 'code_submit/ma_imbalance_mapping/config.json').read_text(encoding='utf-8'))
    cfg['seeds'] = list(range(1000, 1005))
    data, parts, hashes = load_development_inputs(ROOT, cfg)
    matrices, more = load_representations(data, parts); hashes.update(more)
    pool = next(iter(parts.values()))['pool']; pos = {int(v): i for i, v in enumerate(pool)}
    candidates = grid()
    protocol = {'candidates': candidates, 'seeds': cfg['seeds'], 'input_hashes': hashes,
                'code_hash': sha256(__file__), 'scope': 'development-only, continuous numerical MA',
                'reference': 'https://doi.org/10.1609/aaai.v30i1.10306',
                'alignment': 'source-specific centered whitening/recoloring; cov+I in global training scale; target training A only',
                'weights': 'inverse class capped20 per center, source center mass1, target mass4; train only',
                'budget': 'same target A and same permitted sources per arm; no additional labels or unknown signal channels',
                'inductive': 'neither validation nor test inputs/labels used to fit any alignment, transform, weight or model'}
    content = json.dumps(protocol, indent=2)
    if (out / 'PROTOCOL.json').exists() and (out / 'PROTOCOL.json').read_text(encoding='utf-8') != content:
        raise RuntimeError('Changed protocol')
    (out / 'PROTOCOL.json').write_text(content, encoding='utf-8')
    path = out / 'CANDIDATES.jsonl'
    records = [json.loads(v) for v in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    done = {r['candidate']['id'] for r in records}
    for number, c in enumerate(candidates):
        if c['id'] in done: continue
        data['x'] = matrices[c['representation']]; start = time.time(); diagnostics = []
        try:
            predictions = []
            for seed, part in parts.items():
                pred = np.full(len(pool), np.nan)
                for fi, fold in enumerate(part['folds']):
                    A, B = fold['A'], fold['B']
                    model = fit(data, A, c)
                    pred[[pos[int(v)] for v in B]] = model.predict(data['x'][B])
                    diagnostics.append({'seed': seed, 'fold': fi, **model.diagnostics})
                predictions.append(pred)
            p = np.asarray(predictions); y = data['y'][pool, 2]
            np.savez_compressed(out / 'private' / f"{c['id']}.npz", prediction=p, reference=y, pool=pool, seeds=cfg['seeds'])
            (out / 'private' / f"{c['id']}_audit.json").write_text(json.dumps(diagnostics), encoding='utf-8')
            row = {'candidate': c, 'status': 'complete', 'metrics': assess(np.broadcast_to(y, p.shape), p), 'seconds': time.time()-start}
        except Exception as error:
            row = {'candidate': c, 'status': 'failed', 'error': str(error)}
        records.append(row)
        with path.open('a', encoding='utf-8') as stream: stream.write(json.dumps(row, allow_nan=False)+'\n')
        summary = make_summary(records, out)
        print(json.dumps({'done': number+1, 'total': len(candidates), 'failed': summary['failed'], 'feasible': summary['feasible_count']}), flush=True)
    return make_summary(records, out)


if __name__ == '__main__':
    with threadpool_limits(limits=1): run()
