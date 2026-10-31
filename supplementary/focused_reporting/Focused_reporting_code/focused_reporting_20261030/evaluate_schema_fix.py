"""Evaluation-only schema repair; frozen training script/models remain unchanged.

The historical action archive has no seed column. Confirm its seed-axis ordering
against the matched prediction archive, which explicitly stores the seed vector,
and compare all MA values and class assignments before the reference comparison.
No fit or threshold-selection function is changed or invoked here.
"""
import ast
import run as r

source=r.Path(r.__file__).read_text(encoding='utf-8')
old="assert np.array_equal(z['test_ids'].astype(str),D['ids'][idx].astype(str)) and np.array_equal(z['seed'],SEEDS)"
new="""assert np.array_equal(z['test_ids'].astype(str),D['ids'][idx].astype(str))
        prediction_path=OLD/'eval/predictions'/f'{d}.npz'
        evalhash[str(prediction_path)]=sha(prediction_path)
        q=dict(np.load(prediction_path,allow_pickle=False))
        assert np.array_equal(q['seed'],SEEDS)
        assert np.array_equal(q['test_ids'].astype(str),z['test_ids'].astype(str))
        assert np.allclose(q['final_med4'][:,:,2],z['final_ma'],rtol=0,atol=1e-12)
        assert np.array_equal(ma_class(q['final_med4'][:,:,2]),z['final_ma_class'])"""
assert source.count(old)==1
tree=ast.parse(source.replace(old,new))
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
exec(compile(ast.Module(body=[node],type_ignores=[]),__file__,'exec'),r.__dict__)

if __name__=='__main__':
    if not (r.ROOT/'EVALUATION_SCHEMA_REPAIR.json').exists():
        r.write('EVALUATION_SCHEMA_REPAIR.json',dict(reason='Historical action NPZ has no seed column; validate against explicit seed-labelled prediction NPZ',frozen_run_sha=r.sha(r.__file__),evaluation_wrapper_sha=r.sha(__file__),training_changes=False,threshold_changes=False,initial_evaluation_error='KeyError: seed; no aggregate outputs written'))
    r.evaluate()
