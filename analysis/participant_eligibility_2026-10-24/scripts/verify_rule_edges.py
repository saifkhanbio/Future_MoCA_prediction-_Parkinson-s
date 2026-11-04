from audit_common import *
checks=[]
def test(name,v):checks.append({'check':name,'passed':bool(v)})
r={k:'0' for k in ITEMS};r.update(MCATOT='0',MCAVFNUM='0')
a=moca(r,12,'selected');test('Education <=12 adds exactly one',a['raw']==0 and a['adjusted']==1 and not a['reasons'])
r['MCATOT']='1';a=moca(r,12,'selected');test('Reported corrected total is not adjusted twice',a['adjusted']==1 and not a['reasons'])
r['MCATOT']='2';test('Discordant total is held','reported_total_discordance' in moca(r,12,'selected')['reasons'])
r['MCATOT']='';test('Complete items allow flagged total reconstruction','reported_total_missing_reconstructed' in moca(r,13,'selected')['flags'])
r['MCAVFNUM']='11';test('Fluency wordcount mismatch held','fluency_words_score_discordance' in moca(r,13,'selected')['reasons'])
r['MCAVFNUM']='0';test('Unknown education cannot be inferred from total','missing_or_invalid_education' in moca(r,None,'selected')['reasons'])
ceiling={k:str(v) for k,v in ITEMS.items()};ceiling.update(MCATOT='30',MCAVFNUM='11');a=moca(ceiling,None,'selected');test('Ceiling score is education-invariant',a['adjusted']==30 and not a['reasons'])
ceiling['MCAALTTM']='';test('Missing item is not silently filled','missing_or_invalid_items' in moca(ceiling,16,'selected')['reasons'])
test('Dates require exact month/year',month('1/2020') is None and month('13/2020') is None and month('01/2020')==24241)
idx=month('01/2020');records=[{'_row':2,'EVENT_ID':'SC','INFODT':'12/2019','VALUE':'valid'},{'_row':3,'EVENT_ID':'BL','INFODT':'01/2020','VALUE':'invalid'}]
a=select(records,idx,['VALUE']);test('Closest baseline chosen without score fallback',a[0]['VALUE']=='invalid')
records.append({'_row':4,'EVENT_ID':'BL','INFODT':'01/2020','VALUE':'other'});a=select(records,idx,['VALUE']);test('Conflicting selected records hold instead of selecting older value',a[0] is None and a[3]=='conflicting_selected_records')
future=[{'_row':2,'EVENT_ID':'V04','INFODT':'12/2020','VALUE':'A'},{'_row':3,'EVENT_ID':'V04','INFODT':'02/2021','VALUE':'B'}];test('Follow-up equal-distance tie chooses earlier month',select(future,idx,['VALUE'],h=12)[0]['VALUE']=='A')
invalid_exam=[{'_row':2,'EVENT_ID':'BL','INFODT':'01/2020','EXAMDT':'invalid','PDSTATE':'OFF','VALUE':'1'}];test('Invalid supplied motor date cannot fall back to valid form date',select(invalid_exam,idx,['VALUE'],motor=True)[0] is None)
future_pred=[{'_row':2,'EVENT_ID':'BL','INFODT':'02/2020','VALUE':'1'}];test('Future predictor held',select(future_pred,idx,['VALUE'])[0] is None)
result={'passed':all(x['passed'] for x in checks),'checks':checks,'models_fit':0};save('qa/Rule_Edge_Case_Checks.json',result);print(json.dumps({'passed':result['passed'],'checks':len(checks)}));assert result['passed']