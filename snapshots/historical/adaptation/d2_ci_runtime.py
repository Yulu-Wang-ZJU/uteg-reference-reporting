from pathlib import Path
BASE=Path(__file__).resolve().parent.parent/'uteg_d2_adapt60_trust_20261007_005909'
exec(compile((BASE/'runtime.py').read_text(encoding='utf-8'),str(BASE/'runtime.py'),'exec'))
def ledger(key,kind,**kw):
 rows=list((ROOT/'ledger').glob('*.json'));assert len(rows)<5500;assert not (ROOT/f'ledger/{key}.json').exists();write(f'ledger/{key}.json',dict(kind=kind,state='started',time=time.time(),**kw))
