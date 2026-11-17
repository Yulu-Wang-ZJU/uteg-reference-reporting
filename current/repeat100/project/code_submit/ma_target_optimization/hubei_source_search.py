"""Same-target-budget Hubei-only source ablation, development selection only."""
import hashlib
import itertools
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from .search import ROOT, training_view, make_summary
from .models import fit_candidate
from .center_tree_search import fit as fit_tree
from .alignment_search import fit as fit_alignment
from .representation_search import load_representations
from .metrics import assess
from code_submit.ma_imbalance_mapping.models import fit_mapping, project_readouts
from code_submit.ma_imbalance_mapping.data_contract import load_development_inputs, sha256


def add_hubei_arm(data, parts):
    dual = data['source_rows']['HZ+HB_to_SA']
    hb = dual[data['centers'][dual] == 'Hubei']
    assert len(hb) == 79 and set(hb) <= set(data['allowed_training_indices'])
    for part in parts.values():
        assert not (set(hb) & set(np.r_[part['pool'], part['test']]))
        assert not (set(data['groups'][hb]) & set(data['groups'][np.r_[part['pool'], part['test']]]))
    data['source_rows'] = {**data['source_rows'], 'HB_to_SA': hb}
    return data


def grid():
    result = []
    for alpha, cap in itertools.product([.1, 10.], [1., 5.]):
        result.append(dict(family='source_unadapted', arm='HB_to_SA', source_alpha=alpha,
                           source_cap=cap, representation='parameters4'))
        for target_cap in [1., 5., 20.]:
            result.append(dict(family='transfer', arm='HB_to_SA', mode='joint',
                               source_alpha=alpha, source_cap=cap, cap=target_cap,
                               adapter_penalty=10., representation='parameters4'))
    for rep, scale, cap in itertools.product(['parameters4', 'parameters_plus_curve'], [False, True], [5., 20.]):
        result.append(dict(family='center_tree', arm='HB_to_SA', representation=rep,
                           center_scale=scale, loss='squared_error', leaf=1, target_mass=4., cap=cap))
    for rep, method, learner in itertools.product(['parameters4', 'parameters_plus_curve'],
                                                 ['center_scale', 'coral'], ['kernel', 'tree']):
        result.append(dict(family='source_alignment', arm='HB_to_SA', representation=rep,
                           alignment=method, learner=learner))
    for c in result:
        c['id'] = hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:12]
    return result


def fit_predict(data, A, B, c, cache):
    if c['family'] == 'source_unadapted':
        key = ('HB_to_SA', c['source_alpha'], c['source_cap'])
        if key not in cache:
            rows = data['source_rows']['HB_to_SA']
            cache[key] = fit_mapping(data['x'][rows], data['y'][rows],
                           method='class_weighted' if c['source_cap'] > 1 else 'unweighted',
                           parameter=1., ridge_alpha=c['source_alpha'], weight_ratio_cap=c['source_cap'])
        return project_readouts(cache[key].predict(data['x'][B]))[:, 2]
    if c['family'] == 'transfer': model = fit_candidate(c, training_view(data, A), cache)
    elif c['family'] == 'center_tree': model = fit_tree(data, A, c)
    elif c['family'] == 'source_alignment': model = fit_alignment(data, A, c)
    else: raise ValueError(c['family'])
    return model.predict(data['x'][B])


def run():
    out = ROOT/'outputs/uteg_mapping_target_runs/goal_hubei_source_v1'
    out.mkdir(parents=True, exist_ok=True); (out/'private').mkdir(exist_ok=True)
    cfg = json.loads((ROOT/'code_submit/ma_imbalance_mapping/config.json').read_text(encoding='utf-8'))
    cfg['seeds'] = list(range(1000, 1005))
    data, parts, hashes = load_development_inputs(ROOT, cfg); add_hubei_arm(data, parts)
    matrices, more = load_representations(data, parts); hashes.update(more)
    pool = next(iter(parts.values()))['pool']; pos = {int(v): i for i, v in enumerate(pool)}
    candidates = grid()
    protocol = {'candidates': candidates, 'seeds': cfg['seeds'], 'input_hashes': hashes,
                'code_hashes': {str(p.relative_to(ROOT)): sha256(p) for p in
                    [ROOT/'code_submit/ma_target_optimization'/v for v in ['hubei_source_search.py', 'models.py', 'center_tree_search.py', 'alignment_search.py']]},
                'scope': 'Hubei source-only ablation, same target60 development labels/folds, test masked',
                'source': '79 previously permitted and quality-passing Hubei rows; no new data or labels',
                'comparison': 'fixed settings used in prior Huzhou/dual runs, no source-arm selection using test results',
                'unadapted': 'zero local labels in fit, target labels only for development scoring; repeated seeds are identical predictions, not independent data',
                'source_center_counts': {'low': 14, 'within': 37, 'high': 28}}
    text = json.dumps(protocol, indent=2)
    if (out/'PROTOCOL.json').exists() and (out/'PROTOCOL.json').read_text(encoding='utf-8') != text:
        raise RuntimeError('Changed protocol')
    (out/'PROTOCOL.json').write_text(text, encoding='utf-8')
    path = out/'CANDIDATES.jsonl'
    records = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    done = {r['candidate']['id'] for r in records}; cache = {}
    for number, c in enumerate(candidates):
        if c['id'] in done: continue
        data['x'] = matrices[c['representation']]; start = time.time()
        try:
            predictions = []
            for seed, part in parts.items():
                pred = np.full(len(pool), np.nan)
                for fold in part['folds']:
                    A, B = fold['A'], fold['B']
                    pred[[pos[int(v)] for v in B]] = fit_predict(data, A, B, c, cache)
                predictions.append(pred)
            p = np.asarray(predictions); y = data['y'][pool, 2]
            np.savez_compressed(out/'private'/f"{c['id']}.npz", prediction=p, reference=y, pool=pool, seeds=cfg['seeds'])
            row = {'candidate': c, 'status': 'complete', 'metrics': assess(np.broadcast_to(y, p.shape), p), 'seconds': time.time()-start}
        except Exception as error: row = {'candidate': c, 'status': 'failed', 'error': str(error)}
        records.append(row)
        with path.open('a', encoding='utf-8') as stream: stream.write(json.dumps(row, allow_nan=False)+'\n')
        summary = make_summary(records, out)
        print(json.dumps({'done': number+1, 'total': len(candidates), 'failed': summary['failed'], 'feasible': summary['feasible_count']}), flush=True)
    return make_summary(records, out)


if __name__ == '__main__':
    with threadpool_limits(limits=1): run()
