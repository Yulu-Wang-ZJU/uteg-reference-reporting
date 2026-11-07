"""Transfer and reporting numerical helpers used by the current evaluation."""
from pathlib import Path
import argparse, ast, json, importlib.util, math, os, subprocess, sys, time
from typing import Any
import numpy as np
import joblib
from threadpoolctl import threadpool_limits

WORKSPACE = Path(__file__).resolve().parents[2]
HISTORY = WORKSPACE / 'outputs/uteg_supplement_completion_20261027_203140'
ROOT = WORKSPACE / 'outputs/paper_B085_hospital_replay_20261102/transfer'
DATA = WORKSPACE / 'outputs/uteg_B085_by_hospital_20261102/private/DATA15_B085_BY_HOSPITAL.npz'
LEGACY_PRED = WORKSPACE / 'outputs/uteg_signal_model_goal_20261026_000929/ridge10_fullchain_20261026_122551/eval/predictions'
SEEDS = list(range(1000, 1020))
DIRECTIONS = ['HZ+HB_to_SA','SA+HB_to_HZ','SA+HZ_to_HB','HZ_to_SA','HB_to_SA','SA_to_HZ','HB_to_HZ','SA_to_HB','HZ_to_HB']
TARGET = {'HZ_to_SA':'SAHZU','HB_to_SA':'SAHZU','HZ+HB_to_SA':'SAHZU','SA_to_HZ':'Huzhou','HB_to_HZ':'Huzhou','SA+HB_to_HZ':'Huzhou','SA_to_HB':'Hubei','HZ_to_HB':'Hubei','SA+HZ_to_HB':'Hubei'}
FIT_MA_GATE_CODE = WORKSPACE / 'outputs/uteg_nine_direction_ma_20261025_195548/code/fit_ma_gates.py'


def _load_original_select_threshold():
    """Load the historical pure selector without importing its runtime."""
    tree = ast.parse(FIT_MA_GATE_CODE.read_text(encoding='utf-8'))
    names = {'summary', 'select_threshold'}
    namespace = {'np': np, 'math': math, 'Any': Any, 'MA_CLASSES': (-1, 0, 1)}
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(FIT_MA_GATE_CODE), 'exec'), namespace)
    return namespace['select_threshold']


ORIGINAL_SELECT_THRESHOLD = _load_original_select_threshold()


def load_batch():
    spec=importlib.util.spec_from_file_location('b085_transfer_batch', HISTORY/'batch.py')
    b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    b.ROOT=ROOT
    def data_at(h):
        assert h==900
        with np.load(DATA,allow_pickle=False) as z:
            return {k:z[k].copy() for k in z.files}
    b.data_at=data_at
    return b


def initialize():
    ROOT.mkdir(parents=True,exist_ok=True)
    b=load_batch()
    plan_path=ROOT/'PLAN.json'
    if plan_path.exists():
        plan=b.read(plan_path);assert plan['dataset_sha256']==b.sha(DATA)
        if plan.get('wrapper_sha256') != b.sha(__file__):
            plan.setdefault('wrapper_hash_history', []).append(plan.get('wrapper_sha256'))
            plan['wrapper_sha256']=b.sha(__file__);plan['wrapper_repaired_without_fit']=True;b.write(plan_path,plan)
        return b
    historical=b.read(HISTORY/'PLAN.json')
    tasks=[dict(id='prefix_900__'+d,arm='prefix_900',horizon=900,budget='FULL',direction=d,model='RIDGE-10',cap=16000) for d in DIRECTIONS]
    plan=dict(version=1,seeds=SEEDS,directions=DIRECTIONS,tasks=tasks,source_rows=historical['source_rows'],
        protocol_sha=None,runner_sha=b.sha(HISTORY/'batch.py'),wrapper_sha256=b.sha(__file__),dataset_path=str(DATA),dataset_sha256=b.sha(DATA),
        immutable_hashes=historical['immutable_hashes'],B_from_original_data=True,valid_UTEG_only_validation=False,static_revised_B=True,
        task_seeds_total=len(tasks)*len(SEEDS),scope='paper T900 Ridge transfer only; prefix 300/600 are retained historical outputs and are not modified')
    protocol=('Re-run paper T900 Ridge transfer on the static B085 revised DATA15. Nine directions, twenty seeds, '
              'same roles, target folds, Huber/adaptation/MA-error gate and dual report gate. '
              'The study authors report that experimenters at the participating hospitals identified the supplied U-TEG MA values and curves as original instrument measurements. Temporal results retain offline replay conditions.\n')
    (ROOT/'PROTOCOL.md').write_text(protocol,encoding='utf-8');plan['protocol_sha']=b.sha(ROOT/'PROTOCOL.md');b.write(plan_path,plan)
    return b


def train_task(task_id,limit=None):
    b=initialize();plan=b.read(ROOT/'PLAN.json');task=next(x for x in plan['tasks'] if x['id']==task_id)
    b.train_task(task_id,limit)


def _base_predict(b,seq,runner,target_mod,obj,D):
    role=b.read(b.BASE/'ROLE_FREEZE.json');target=TARGET[obj['direction']];test=np.asarray(role['roles'][target]['test'],int)
    src=obj['source_model'];x=D['x'][test];raw=target_mod.m.predict(src,x);base_obj=obj['cache'][obj['positions'][3]['base']]
    H=target_mod.m.calibrated(base_obj['cal'],raw);med={'H':H}
    for a,mod in base_obj['adapters'].items():med[a]=target_mod.m.ans['adapt'](mod,x,raw,H,src)
    selected=obj['final_selection']['adapter'];j=target_mod.m.ci(med[selected]);pred=runner.ma_class(med[selected][:,2])
    gate_name=obj['final_selection'].get('name');gate_model=obj['gate_models'][3]
    if 'model' in gate_model:
        F=runner.ma_features(x,med[selected],obj['source_model']['ym'],obj['source_model']['ys'])
        score=runner.score_ma_gate(gate_model['model'],F,pred)
        allowed=np.isfinite(score)&(pred!=9)
        action=np.zeros(len(test),dtype=bool)
    elif target=='Huzhou':
        gate_name=gate_name or gate_model.get('name');sm=obj['scalar_models'][3][gate_name];evidence=target_mod.m.ci(raw) if gate_name=='SOURCE-CI-ORDINAL' else j
        score=target_mod.m.scalar['score'](sm,evidence,pred);allowed=np.isfinite(score)&(pred!=9);action=target_mod.m.st['actions'](score,pred,obj['final_selection'].get('threshold'))
    else:
        F=target_mod.m.gm.features(x,med[selected],src);score,allowed,_=target_mod.m.gm.score(gate_name,gate_model['models'][gate_name],F);action=target_mod.m.gs['actions'](score,allowed,obj['final_selection'].get('threshold'))
    bad=(target=='Hubei')&(~np.asarray(D['input_qc'][test],bool));action[bad]=False;allowed[bad]=False
    return dict(test=test,raw=raw,pure=H,med=med[selected],pred_class=pred,score=score,allowed=allowed,base_action=action,reference_class=runner.ma_class(D['medical'][test,2]),target=target,selected_adapter=selected,selected_gate=gate_name)


def _dual_action(th,pred,base,dual,qc):
    out=np.zeros(len(pred),dtype=bool)
    if not th or th.get('all_withhold'): return out
    if th.get('normal') is not None:
        out |= (pred==0) & np.isfinite(dual) & (dual<=float(th['normal']))
    if th.get('tail') is not None:
        out |= np.isin(pred,[-1,1]) & np.isfinite(base) & (base<=float(th['tail']))
    return out & np.asarray(qc,dtype=bool)


def _development_lr_threshold(outer, target):
    """Select the all-class MA-error-LR threshold on pooled outer-B rows."""
    score = np.asarray([np.nan if row.get('base_score') is None else row['base_score'] for row in outer], dtype=float)
    pred = np.asarray([row['pred_class'] for row in outer], dtype=np.int8)
    truth = np.asarray([row['truth_class'] for row in outer], dtype=np.int8)
    max_risk = 0.10 if target == 'Huzhou' else 0.15
    selected = ORIGINAL_SELECT_THRESHOLD(score, pred, truth, max_risk)
    threshold = selected.get('threshold') if selected.get('eligible', False) else None
    return (None if threshold is None else float(threshold)), selected


def evaluate_task(b,task_id,seeds=None,force=False):
    plan=b.read(ROOT/'PLAN.json');task=next(x for x in plan['tasks'] if x['id']==task_id);outdir=ROOT/'results';outdir.mkdir(exist_ok=True)
    dest=outdir/(task_id+'.json');pred_path=ROOT/'predictions'/ (task_id+'.npz')
    if dest.exists() and not force:return
    r,runner,target_mod,seq,c,sel=b.env(task_id,task['cap']);D=b.data_at(900);rows=[]
    seeds = SEEDS if seeds is None else list(seeds)
    development_thresholds=[]
    lr_thresholds=[]
    lr_selections=[]
    for seed in seeds:
        fr=b.read(r.ROOT/'frozen'/f'{seed}.json');obj=joblib.load(fr['target_path']);q=_base_predict(b,seq,runner,target_mod,obj,D);dual=joblib.load(fr['dual_path'] if 'dual_path' in fr else r.ROOT/'dual'/f'{task["direction"]}_{seed}.joblib');threshold=fr.get('threshold')
        if threshold is None:threshold=seq.choose(sel,dual['outer'])
        development_thresholds.append(threshold)
        lr_threshold,lr_selection=_development_lr_threshold(dual['outer'],q['target'])
        lr_thresholds.append(np.nan if lr_threshold is None else float(lr_threshold));lr_selections.append(lr_selection)
        h=dual['heads'][3];med=q['med'];low=seq.score_head(c,h.get('low'),D['x'][q['test'],1],med[:,2]);high=seq.score_head(c,h.get('high'),D['x'][q['test'],1],med[:,2]);dual_score=np.maximum(low,high)
        qc=np.asarray(D['input_qc'][q['test']],bool)&np.isfinite(q['med']).all(1)&np.isfinite(q['score'])
        base_action=np.zeros(len(q['test']),dtype=bool) if lr_threshold is None else qc&(q['score']<=float(lr_threshold))
        a=_dual_action(threshold,q['pred_class'],q['score'],dual_score,D['input_qc'][q['test']]);
        rows.append(dict(seed=seed,test=q['test'].tolist(),raw_med4=q['raw'].tolist(),pure_huber_med4=q['pure'].tolist(),final_med4=q['med'].tolist(),pred_ma=q['med'][:,2].tolist(),pred_class=q['pred_class'].tolist(),reference_class=q['reference_class'].tolist(),base_action=base_action.tolist(),dual_action=a.tolist(),base_score=q['score'].tolist(),dual_score=dual_score.tolist(),development_threshold=threshold,lr_threshold=lr_threshold,lr_selection=lr_selection,input_qc=D['input_qc'][q['test']].tolist(),target=q['target'],selected_adapter=q['selected_adapter'],selected_gate=q['selected_gate']))
    pred_path.parent.mkdir(exist_ok=True);np.savez_compressed(pred_path,seed=seeds,raw_med4=np.asarray([x['raw_med4'] for x in rows]),pure_huber_med4=np.asarray([x['pure_huber_med4'] for x in rows]),final_med4=np.asarray([x['final_med4'] for x in rows]),pred_ma=np.asarray([x['pred_ma'] for x in rows]),pred_class=np.asarray([x['pred_class'] for x in rows]),reference_class=np.asarray([x['reference_class'] for x in rows]),base_action=np.asarray([x['base_action'] for x in rows]),dual_action=np.asarray([x['dual_action'] for x in rows]),base_score=np.asarray([x['base_score'] for x in rows]),dual_score=np.asarray([x['dual_score'] for x in rows]),lr_threshold=np.asarray(lr_thresholds,float),input_qc=np.asarray([x['input_qc'] for x in rows]),ids=D['ids'][np.asarray(rows[0]['test'])],groups=D['groups'][np.asarray(rows[0]['test'])],test=np.asarray(rows[0]['test']),B_from_original_data=np.asarray(True),valid_UTEG_only_validation=np.asarray(False))
    y=np.asarray(rows[0]['reference_class']);p=np.asarray([x['pred_class'] for x in rows]);a=np.asarray([x['dual_action'] for x in rows],bool);known=y!=9;ab=known&(y!=0);norm=known&(y==0);pm=p;yb=np.broadcast_to(y,pm.shape);Yma=D['medical'][np.asarray(rows[0]['test']),2];finite=np.isfinite(Yma)&np.asarray(rows[0]['input_qc'],bool)
    labels=(-1,0,1);cm=[[int(np.sum((y[None,:]==i)&(pm==j))) for j in labels] for i in labels];summary=dict(task=task,seeds=len(rows),target=rows[0]['target'],records=int(known.sum()),agreement=float(np.mean(pm==yb)),confusion_labels=['low','normal','high'],confusion=cm,mae=float(np.mean(np.abs(np.asarray([x['pred_ma'] for x in rows])[:,finite]-Yma[None,finite])),),all_report_fnr=float(np.mean(pm[:,ab]==0)) if ab.any() else None,all_report_fpr=float(np.mean(pm[:,norm]!=0)) if norm.any() else None,coverage=float(a.mean()),post_gate_agreement=float(np.mean(pm[a]==yb[a])) if a.any() else None,post_gate_fnr=float(np.mean(pm[a&(ab[None,:])]==0)) if (a&ab[None,:]).any() else None,post_gate_fpr=float(np.mean(pm[a&(norm[None,:])]!=0)) if (a&norm[None,:]).any() else None,B_from_original_data=True,valid_UTEG_only_validation=False,ma_denominator_records=int(finite.sum()))
    base=np.asarray([x['base_action'] for x in rows],bool)
    summary.update(base_coverage=float(base.mean()),base_agreement=float(np.mean(pm[base]==yb[base])) if base.any() else None,base_fnr=float(np.mean(pm[(base)&ab[None,:]]==0)) if (base&ab[None,:]).any() else None,base_fpr=float(np.mean(pm[(base)&norm[None,:]]!=0)) if (base&norm[None,:]).any() else None,lr_thresholds=[None if not np.isfinite(x) else float(x) for x in lr_thresholds],lr_selections=lr_selections,development_thresholds=development_thresholds)
    summary['evaluated_seeds']=list(seeds);b.write(dest,summary);b.write(ROOT/'seals'/ (task_id+'.json'),dict(prediction_sha256=b.sha(pred_path),dataset_sha256=b.sha(DATA),reference_scoring_started=True,base_threshold_source='original_fit_ma_gates.select_threshold_on_pooled_development_outer_B',ma_class_contract='low=-1,normal=0,high=1'))


def smoke():
    b=initialize();task=next(t for t in b.read(ROOT/'PLAN.json')['tasks'] if t['direction']=='HZ_to_SA');
    # One seed smoke keeps the fit contract lightweight and verifies the complete T900 chain.
    if not (ROOT/'runs'/task['id']/'frozen'/'1000.json').exists(): b.train_task(task['id'],limit=1)
    evaluate_task(b,task['id'],seeds=[1000]);b.write(ROOT/'SMOKE_VERIFIED.json',dict(task=task['id'],seed=1000,B_from_original_data=True,reference_scores_not_used_for_fit=True))


def launch(workers=2):
    b=initialize();plan=b.read(ROOT/'PLAN.json');pending=[t for t in plan['tasks'] if not (ROOT/'runs'/t['id']/'COMPLETE.json').exists()]
    active=[];failed=[];(ROOT/'logs').mkdir(exist_ok=True)
    # Representative smoke is allowed before wide launch; each task resumes from its own run root.
    while pending or active:
        while pending and len(active)<workers and not failed:
            t=pending.pop(0);h=(ROOT/'logs'/(t['id']+'.log')).open('a',encoding='utf-8');p=subprocess.Popen([sys.executable,'-B',str(__file__),'task','--id',t['id']],cwd=WORKSPACE,stdout=h,stderr=subprocess.STDOUT);active.append((p,t,h))
        for p,t,h in active[:]:
            if p.poll() is not None:
                h.close();active.remove((p,t,h));
                if p.returncode:failed.append(t['id'])
                elif (ROOT/'runs'/t['id']/'COMPLETE.json').exists():evaluate_task(b,t['id'])
        b.write(ROOT/'BATCH_STATUS.json',dict(running=[t['id'] for p,t,h in active],remaining=len(pending),failed=failed,completed=len(list((ROOT/'results').glob('*.json')))))
        if failed and not active:raise RuntimeError(str(failed))
        time.sleep(2)
    b.write(ROOT/'COMPLETE.json',dict(tasks=len(plan['tasks']),results=len(list((ROOT/'results').glob('*.json'))),B_from_original_data=True,original_default_data_unchanged=True))


