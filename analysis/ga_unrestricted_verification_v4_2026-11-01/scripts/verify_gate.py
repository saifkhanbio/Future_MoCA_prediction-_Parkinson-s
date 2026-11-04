from common import *
def main():
    base,profile,b,by,cohorts=data()
    auth=read(R/'restricted/Execution_Authorization.json');lock=read(E/'EXECUTION_LOCK.json')
    assert auth['participant_training_authorized'] is True
    assert sha(R/'restricted/Execution_Authorization.json')==lock['authorization_sha256']
    for key,path in [('plan_lock_sha256',A/'PLAN_LOCK.json'),('launcher_lock_sha256',A/'LAUNCHER_LOCK.json'),('profile_sha256',A/'config/r2_expanded_v1.json')]:
        assert auth[key]==sha(path)
    assert auth['study_id']==base['study_id'] and auth['profile_id']==profile['profile_id']
    gate=read(R/'restricted/Convergence_Gate.json');run=read(R/'RUN.json')
    assert gate['status']=='PASS' and gate['binding']==run['binding'] and len(gate['selection_digests'])==30
    gate_time=(R/'restricted/Convergence_Gate.json').stat().st_mtime
    deltas=[];outer_counts=[]
    for p in (R/'restricted').glob('*/Result.json'):
        result=read(p);ready=read(p.parent/'Convergence.json')
        assert ready['ledger']['refits']==0
        assert ready['selection_digest']==gate['selection_digests'][result['context_id']]
        assert result['start_unix']>=gate_time
        assert result['ledger']['refits']==2
        deltas.append(result['start_unix']-gate_time);outer_counts.append(result['ledger']['refits'])
    assert len(deltas)==30 and sum(outer_counts)==60
    # Small independent implementation checks on mask accounting and intercept-only designs.
    for arm in ['unrestricted28','moca27']:
        c=arm_contract(base,arm);ns,cs=layout(c);row={k:'' for k in ns};row.update({k:'unknown' for k in cs})
        train=[dict(row),dict(row)];test=[dict(row)]
        x,t,labels,state=full_design(train,test,c)
        assert np.isfinite(x).all() and np.isfinite(t).all()
        assert all(v['median']==v['mean']==0 and v['sd']==1 and v['training_observed_n']==0 for v in state['numeric'].values())
        zcost,zcount=burden_count([0],c);assert int(zcost[0])==int(zcount[0])==(1 if arm=='moca27' else 0)
        motor=next(f['bit'] for f in c['features']['optional'] if f['name']=='motor_state')
        selected=[labels[j] for j in indices(1<<motor,c,labels)]
        assert sum(label.startswith('motor_state==') for label in selected)==5
        assert all('baseline_hvlt_form==' not in label and 'baseline_sdmt_form==' not in label for label in selected)
        g1=np.random.default_rng(773);g2=np.random.default_rng(773);space=1<<len(c['features']['optional'])
        assert [int(g1.integers(0,space)) for _ in range(1000)]==g2.integers(0,space,size=1000).tolist()
        assert g1.bit_generator.state==g2.bit_generator.state
    report={'status':'PASS','execution_authorization_hash':'PASS','all_context_gate_bindings':'PASS','contexts':30,'outer_refits_after_gate':60,
            'minimum_recorded_seconds_between_gate_write_and_refit_task_start':min(deltas),'readiness_records_have_no_outer_refits':True,
            'independent_helper_checks':['all-missing preprocessing','mandatory-MoCA count','zero-mask count','categorical motor-state parent','form-column triggers','random scalar/vector stream equivalence']}
    write(V/'evidence/Gate_And_Authorization.json',report);print(json.dumps(report))
if __name__=='__main__':main()
