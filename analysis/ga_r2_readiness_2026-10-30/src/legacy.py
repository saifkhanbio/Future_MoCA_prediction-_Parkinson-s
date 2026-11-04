"""Evaluation-only matched comparisons with frozen MAE-selected predictions."""
from support import *
ARMS=['core','enhanced','full','GA_selected','random_selected','exhaustive_selected']
def compare_rows(new,old,horizon,synthetic=False):
    ni=[r['PATNO'] for r in new];oi=[r['PATNO'] for r in old]
    if len(set(ni))!=len(ni) or len(set(oi))!=len(oi) or set(ni)!=set(oi):raise IntegrityError('Legacy participant mismatch')
    om={r['PATNO']:r for r in old};old=[om[i] for i in ni]
    if synthetic and not all(str(i).startswith('SYNTH_') for i in ni):raise IntegrityError('Nonsynthetic legacy record')
    if len({r['component_id'] for r in new})!=len(new):raise IntegrityError('Legacy component duplication')
    for n,o in zip(new,old):
        if n['component_id']!=o['component_id'] or n['outer_fold']!=o['outer_fold'] or n['y']!=o['y']:raise IntegrityError('Legacy endpoint/fold/component mismatch')
    y=np.array([r['y'] for r in new])
    return [{'horizon':horizon,'arm':a,'contrast':'R2-selected minus frozen MAE-selected','frozen_model_refits':0,
             **frozen.interval(y,np.array([r[a] for r in new]),np.array([r[a] for r in old]),seed=20261030+horizon)} for a in ARMS]
def compare_frozen(out):
    spec=json.loads((ROOT/'config/Legacy_Comparison.json').read_text());rows=[]
    for h in (12,24,36):
        info=spec['OOF'][str(h)]
        if file_sha(info['path'])!=info['sha256']:raise IntegrityError('Legacy prediction hash changed')
        rows.extend(compare_rows(json.loads((out/f'restricted/OOF_{h}m.json').read_text()),json.loads(Path(info['path']).read_text()),h))
    info=spec['selections']
    if file_sha(info['path'])!=info['sha256']:raise IntegrityError('Legacy selection hash changed')
    old=json.loads(Path(info['path']).read_text());new=json.loads((out/'results/Selections.json').read_text())
    om={(r['horizon'],r['outer'],r['arm']):r for r in old}
    if len(new)!=90 or len(om)!=90:raise IntegrityError('Legacy model count mismatch')
    counts=[]
    for r in new:
        o=om[(r['horizon'],r['outer'],r['arm'])]
        counts.append({'horizon':r['horizon'],'outer':r['outer'],'arm':r['arm'],'R2_instruments':r['instruments'],'MAE_instruments':o['instruments'],'difference':r['instruments']-o['instruments']})
    write_json(out/'results/Frozen_MAE_Contrasts.json',rows)
    write_json(out/'results/Frozen_MAE_Burden_Comparison.json',counts)
