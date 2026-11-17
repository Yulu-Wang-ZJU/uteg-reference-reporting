"""Weighted abnormal detection with separate low/high direction diagnostics.

Low/high are abnormal positives. Outputs are class decisions, never fabricated
continuous MA values. Original MA mapping, clinical cutoffs and test set stay
unchanged. Weighted sklearn logistic/SVC/forest and nested training-side threshold
selection implement an explicit false-negative-cost experiment.
"""
import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
import time
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
import sklearn
from threadpoolctl import threadpool_limits
from .models import Features
from .metrics import states
from .search import ROOT, training_view
from .voltage_scale import load_inputs
from .hubei_source_search import add_hubei_arm
from code_submit.ma_imbalance_mapping.data_contract import load_development_inputs, sha256


def binary_weights(positive, balanced, cost=1.):
    positive = np.asarray(positive, bool)
    counts = np.bincount(positive.astype(int), minlength=2)
    if not counts.min():
        raise ValueError('Both training classes are required')
    w = len(positive)/(2*counts[positive.astype(int)]) if balanced else np.ones(len(positive))
    w = w*np.where(positive, cost, 1.)
    return w/w.mean()


def assemble(train, arm):
    tx, ty = np.asarray(train['x'], float), np.asarray(train['y'], float)[:, 2]
    if arm == 'target_only':
        return tx, ty, np.zeros((len(tx), 0)), np.zeros(len(tx), int)
    if arm != 'HB_to_SA':
        raise ValueError('Only the two predeclared arms are supported')
    sx, sy = train['source_centers']['Hubei']; sy = np.asarray(sy, float)[:, 2]
    return np.vstack([sx, tx]), np.r_[sy, ty], np.r_[np.ones(len(sx)), np.zeros(len(tx))][:, None], np.r_[np.zeros(len(sx), int), np.ones(len(tx), int)]


def pooled_weights(labels, blocks, balanced, cost=1.):
    w = np.empty(len(labels))
    codes = np.unique(blocks)
    for code in codes:
        keep = blocks == code
        w[keep] = binary_weights(labels[keep], balanced, cost)/keep.sum()/len(codes)
    # Fixed mean1 preserves the solver's regularization scale across costs.
    return w/w.mean()


def estimator(kind):
    if kind == 'logistic':
        return LogisticRegression(C=1., solver='lbfgs', max_iter=5000, tol=1e-8)
    if kind == 'rbf_svc':
        return SVC(C=1., kernel='rbf', gamma='scale', probability=False, tol=1e-6, cache_size=200)
    if kind == 'forest':
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=2, max_features='sqrt',
                                      bootstrap=True, n_jobs=1, random_state=20261101)
    raise ValueError('Unknown learner')


@dataclass
class Detector:
    transform: Features
    source_columns: int
    detection: object
    direction: object
    constant_direction: object
    diagnostics: dict

    def design(self, x):
        z = self.transform.transform(x)
        if not np.isfinite(z).all():
            raise ValueError('Nonfinite query input')
        return np.c_[z, np.zeros((len(z), self.source_columns))]

    def score(self, x):
        z = self.design(x)
        if hasattr(self.detection, 'decision_function'):
            return np.asarray(self.detection.decision_function(z), float)
        return np.asarray(self.detection.predict_proba(z)[:, 1]-.5, float)

    def tail(self, x):
        z = self.design(x)
        return np.full(len(z), self.constant_direction, int) if self.direction is None else self.direction.predict(z).astype(int)


def model_digest(model):
    keys = ('coef_', 'intercept_', 'support_', 'support_vectors_', 'dual_coef_', 'classes_')
    payload = b''.join(np.asarray(getattr(model, k)).tobytes() for k in keys if hasattr(model, k))
    if hasattr(model, 'estimators_'):
        for tree in model.estimators_:
            nodes = tree.tree_.__getstate__()['nodes']
            # Hash named fields, excluding uninitialized structured-array padding.
            payload += b''.join(nodes[name].tobytes() for name in nodes.dtype.names)
            payload += tree.tree_.value.tobytes()
    return hashlib.sha256(payload).hexdigest()


def fit(train, c):
    x, y, codes, blocks = assemble(train, c['arm'])
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Finite training values required')
    label = states(y); positive = label != 1
    transform = Features.fit(x, 'linear'); z = np.c_[transform.transform(x), codes]
    weight = pooled_weights(positive, blocks, c['balanced'], c['positive_cost'])
    detection = estimator(c['learner'])
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        detection.fit(z, positive.astype(int), sample_weight=weight)
    # Direction learner is fixed across detector weight settings. It only sees
    # actual abnormal TRAINING rows; queried cases need no true MA label.
    tails = np.unique(label[positive]); direction = None; constant = None
    if len(tails) == 1:
        constant = int(tails[0])
        direction_weight = np.ones(positive.sum())
    else:
        direction_weight = np.empty(positive.sum()); tail_blocks = blocks[positive]
        for block in np.unique(tail_blocks):
            keep = tail_blocks == block; side = label[positive][keep] == 2
            # A rare fold with only one tail still has valid labels; equal
            # weights in that block do not invent the missing tail.
            local = binary_weights(side, True) if np.unique(side).size == 2 else np.ones(keep.sum())
            direction_weight[keep] = local/keep.sum()/len(np.unique(tail_blocks))
        direction_weight /= direction_weight.mean()
        direction = estimator(c['learner'])
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            direction.fit(z[positive], label[positive], sample_weight=direction_weight)
    diagnostic = {'training_n': len(y), 'target_n': len(train['x']), 'detection_hash': model_digest(detection),
                  'direction_hash': model_digest(direction) if direction is not None else str(constant),
                  'feature_mean': transform.mean.tolist(), 'feature_scale': transform.scale.tolist(),
                  'weight_by_block': []}
    for block in np.unique(blocks):
        mask = blocks == block
        diagnostic['weight_by_block'].append({'block': int(block), 'n': int(mask.sum()),
            'normal_n': int(np.sum(mask & ~positive)), 'abnormal_n': int(np.sum(mask & positive)),
            'loss_mass': float(weight[mask].sum()/weight.sum()),
            'abnormal_to_normal_per_record_weight': float(weight[mask & positive][0]/weight[mask & ~positive][0])})
    return Detector(transform, codes.shape[1], detection, direction, constant, diagnostic)


def high_recall_threshold(score, reference, max_fnr=.05):
    score = np.asarray(score, float); reference = np.asarray(reference, float)
    if score.shape != reference.shape or not np.isfinite(score).all():
        raise ValueError('Invalid inner-OOF scores')
    positive = states(reference) != 1; values = np.sort(score[positive])
    if not len(values):
        raise ValueError('No abnormal inner validation records')
    # Strict FNR<5%, with positives defined by score>=threshold. Ties remain
    # positive. For8 abnormal training records this permits zero misses.
    max_misses = max(0, int(np.ceil(max_fnr*len(values)))-1)
    threshold = float(values[max_misses])
    fn = int(np.sum(positive & (score < threshold)))
    assert fn/positive.sum() < max_fnr
    return threshold, {'abnormal_n': int(positive.sum()), 'normal_n': int((~positive).sum()),
                       'false_negatives': fn, 'false_positives': int(np.sum(~positive & (score >= threshold))),
                       'threshold': threshold, 'scope': 'inner target OOF only; no guarantee for outer or future data'}


def classification_metrics(reference, score, direction, threshold):
    y = np.asarray(reference, float).ravel(); score = np.asarray(score, float).ravel()
    direction = np.asarray(direction, int).ravel()
    cut = np.broadcast_to(threshold, np.asarray(reference).shape).ravel()
    if y.shape != score.shape or y.shape != direction.shape or not np.isfinite(score).all() or not np.isin(direction, [0, 2]).all():
        raise ValueError('Misaligned class predictions')
    true = states(y); actual = true != 1; predicted = score >= cut
    selected = np.where(predicted, direction, 1)
    cm2 = np.zeros((2, 2), int); np.add.at(cm2, (actual.astype(int), predicted.astype(int)), 1)
    cm3 = np.zeros((3, 3), int); np.add.at(cm3, (true, selected), 1)
    tn, fp, fn, tp = cm2.ravel(); safe = lambda a,b: float(a/b) if b else None
    accuracy3 = float(np.mean(selected == true)); fnr, fpr = safe(fn, fn+tp), safe(fp, fp+tn)
    result = {'n': len(y), 'coverage': 1., 'three_class_agreement': accuracy3,
              'binary_accuracy': float((tn+tp)/len(y)), 'fnr': fnr, 'fpr': fpr,
              'binary_confusion_rows_normal_abnormal': cm2.tolist(), 'three_class_confusion_rows_low_normal_high': cm3.tolist(),
              'opposite_abnormal_count': int(np.sum(actual & predicted & (selected != true))),
              'classification_joint_pass': bool(accuracy3 > .8 and fnr is not None and fpr is not None and fnr < .05 and fnr < fpr),
              'continuous_MA_evaluated': False, 'gate_evaluated': False}
    for k, name in [(0, 'low'), (1, 'within'), (2, 'high')]:
        mask = true == k
        result[name+'_state_recall'] = safe(np.sum(mask & (selected == k)), mask.sum())
        if k != 1: result[name+'_abnormal_detection_recall'] = safe(np.sum(mask & predicted), mask.sum())
    return result


def grid(learners=('logistic', 'rbf_svc')):
    conditions = [(False, 1.), (True, 1.), (True, 4.), (True, 16.)]
    result = []
    for arm, learner, (balanced, cost) in itertools.product(('target_only', 'HB_to_SA'), learners, conditions):
        c = {'family': 'binary_cost_sensitive', 'arm': arm, 'learner': learner, 'balanced': balanced, 'positive_cost': cost}
        c['id'] = hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:12]; result.append(c)
    return result


def outer_prediction(data, fold, c):
    A, B = fold['A'], fold['B']; a_pos = {int(v):i for i,v in enumerate(A)}
    assert not set(data['groups'][A]) & set(data['groups'][B])
    inner_score = np.full(len(A), np.nan)
    for inner in fold['inner']:
        T, V = inner['T'], inner['V']
        assert set(T) <= set(A) and set(V) <= set(A)
        assert not set(data['groups'][T]) & set(data['groups'][np.r_[V,B]])
        inner_model = fit(training_view(data,T),c)
        inner_score[[a_pos[int(v)] for v in V]] = inner_model.score(data['x'][V])
    threshold, info = high_recall_threshold(inner_score, data['y'][A,2])
    model = fit(training_view(data,A),c)
    return model.score(data['x'][B]), model.tail(data['x'][B]), threshold, {'threshold_selection':info,'fit':model.diagnostics}


def run(out, seeds, ids=None, learners=('logistic', 'rbf_svc')):
    out.mkdir(parents=True, exist_ok=True); (out/'private').mkdir(exist_ok=True)
    cfg = json.loads((ROOT/'code_submit/ma_imbalance_mapping/config.json').read_text(encoding='utf-8'))
    cfg['seeds'] = list(range(1000, 1000+seeds))
    data, parts, hashes = load_development_inputs(ROOT, cfg); add_hubei_arm(data, parts)
    matrices, input_audit, more = load_inputs(data, parts); hashes.update(more); data['x'] = matrices[(300, 'p5')]
    candidates = grid(learners)
    if ids is not None:
        candidates = [c for c in candidates if c['id'] in ids]; assert len(candidates) == len(ids)
    protocol = {'candidates': candidates, 'seeds': cfg['seeds'], 'input_hashes': hashes,
                'code_hash': sha256(__file__), 'sklearn': sklearn.__version__,
                'scope': 'same60 target development records, approved79 Hubei only when pooled; grouped outer/inner folds; original201 test X/Y excluded',
                'user_confirmation': 'retain three-class agreement>80%; abnormal binary FNR<5% and FNR<FPR; post-gate coverage>75% remains a later requirement',
                'labels': 'lowMA<50 or highMA>70 are abnormal positive;50<=MA<=70 is negative; frozen boundaries and labels',
                'representation': 'current best18 common300 P5 voltage features, shared training-only standardization; source indicator appended',
                'learners': list(learners),
                'estimator_parameters': 'sklearn LogisticRegression C1 lbfgs tol1e-8 max_iter5000; SVC RBF C1 gamma=scale tol1e-6; forest200 trees min_samples_leaf2 max_features=sqrt bootstrapTrue seed20261101 n_jobs1; no clinical probability calibration claim',
                'weights': 'within each source/target block ordinary or n/(2*class_count), then abnormal multiplier1/4/16; normalize within block; source and target total mass.5 each; final mean1 fixes regularization scale',
                'thresholds': 'fixed margin0 (forest score=Pabnormal-.5); or highest inner-target-OOF cutoff with FNR strictly<5%; frozen before predicting outer B; B labels score only',
                'direction': 'separate fixed learner of low vs high on abnormal training rows only, balanced within each source/target block; unchanged across detector weight conditions',
                'outputs': 'class decisions only; no class-mean pseudo-MA, no new numerical mapping or gate claim',
                'comparison': f'{len(candidates)} base configurations with2 threshold policies; no feature/model hyperparameter sweep',
                'references': ['https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.LogisticRegression.html',
                               'https://scikit-learn.org/1.8/modules/generated/sklearn.svm.SVC.html',
                               'https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestClassifier.html',
                               'https://cseweb.ucsd.edu/~elkan/rescale.pdf'],
                'evaluation_limit': 'adaptive development screening, not untouched external validation; repeated seeds do not add independent cases'}
    serialized = json.dumps(protocol, indent=2, allow_nan=False)
    if (out/'PROTOCOL.json').exists() and (out/'PROTOCOL.json').read_text(encoding='utf-8') != serialized:
        raise RuntimeError('Changed frozen protocol')
    (out/'PROTOCOL.json').write_text(serialized, encoding='utf-8'); (out/'INPUT_AUDIT.json').write_text(json.dumps(input_audit, indent=2), encoding='utf-8')
    path = out/'CLASSIFICATION_RESULTS.jsonl'; records = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    pool = next(iter(parts.values()))['pool']; pos = {int(v):i for i,v in enumerate(pool)}
    for c in candidates:
        done = {r['candidate']['id'] for r in records}
        if all(c['id']+'_'+p in done for p in ('fixed0', 'inner_fnr5')): continue
        start = time.time(); scores, directions, cuts, audits = [], [], [], []
        for seed, part in parts.items():
            score = np.full(len(pool), np.nan); direction = np.full(len(pool), -1); cut = np.full(len(pool), np.nan)
            for number, fold in enumerate(part['folds']):
                s,d,t,audit = outer_prediction(data,fold,c); ix=[pos[int(v)] for v in fold['B']]
                score[ix],direction[ix],cut[ix] = s,d,t
                audits.append({'seed':seed,'fold':number,**audit})
            assert np.isfinite(score).all() and np.isfinite(cut).all(); scores.append(score); directions.append(direction); cuts.append(cut)
        score,direction,cut=np.asarray(scores),np.asarray(directions),np.asarray(cuts); y=data['y'][pool,2]
        np.savez_compressed(out/'private'/f"{c['id']}.npz",score=score,direction=direction,threshold=cut,reference=y,pool=pool,seeds=cfg['seeds'])
        for policy, threshold in [('fixed0',0.),('inner_fnr5',cut)]:
            candidate={**c,'base_id':c['id'],'threshold_policy':policy,'id':c['id']+'_'+policy}
            if candidate['id'] in done: continue
            m=classification_metrics(np.broadcast_to(y,score.shape),score,direction,threshold)
            row={'candidate':candidate,'status':'complete','metrics':m}; records.append(row)
            with path.open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
            print(json.dumps({'arm':c['arm'],'learner':c['learner'],'balanced':c['balanced'],'cost':c['positive_cost'],'policy':policy,
                              'seconds':round(time.time()-start,1),**{k:m[k] for k in ('three_class_agreement','binary_accuracy','fnr','fpr')}}),flush=True)
        (out/'private'/f"{c['id']}_audit.json").write_text(json.dumps(audits,indent=2,allow_nan=False),encoding='utf-8')
    qualified=[r for r in records if r['metrics']['three_class_agreement']>.8]
    summary={'complete':len(records),'classification_joint_passes':sum(r['metrics']['classification_joint_pass'] for r in records),
             'best_accuracy_qualified':sorted(qualified,key=lambda r:r['metrics']['fnr'])[:5],
             'lowest_fnr':sorted(records,key=lambda r:(r['metrics']['fnr'],-r['metrics']['three_class_agreement']))[:5],
             'continuous_mapping_or_gate_verified':False}
    (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8');return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seeds',type=int,default=5)
    parser.add_argument('--output',default='goal_cost_sensitive_detection_v1');parser.add_argument('--ids',nargs='*')
    parser.add_argument('--learners',nargs='+',choices=('logistic','rbf_svc','forest'),default=['logistic','rbf_svc'])
    args=parser.parse_args()
    with threadpool_limits(limits=1):run(ROOT/'outputs/uteg_mapping_target_runs'/args.output,args.seeds,args.ids,args.learners)
