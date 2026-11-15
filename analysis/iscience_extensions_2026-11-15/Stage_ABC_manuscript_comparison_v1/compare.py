"""Summarize verified aggregate results. No models or participant records are loaded."""
import csv,json,hashlib,os
from pathlib import Path
O=Path(__file__).resolve().parent
os.umask(0o077)
os.environ.setdefault('MPLCONFIGDIR',str(O/'.mplcache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=O.parent
ab=P/'Stage_AB_verified_results_v1'; cc=P/'Stage_C_verified_results_v1'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def j(p):return json.loads(p.read_text())
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
seals={**{ab/k:v for k,v in j(ab/'Result_manifest.json')['files'].items()},**{cc/k:v for k,v in j(cc/'RESULTS_LOCK.json').items()}}
def check():
    assert all(digest(k)==v for k,v in seals.items())
check()
assert j(ab/'Verification.json')['status']==j(cc/'verification.json')['status']=='PASS'
AB=read(ab/'Verified_results.csv');C=read(cc/'verified_estimates.csv')
assert len(AB)==813 and len(C)==546
idx={(r['stage'],r['view'],int(r['horizon']),r['measure']):r for r in AB}
idx.update({('C',r['view'],int(r['horizon']),r['measure']):r for r in C})
rows=[]
for stage,view,model in [('C','expanded','compact4'),('A','compact','compact4'),('B','nominal','reduced3')]:
    for h in (12,24,36):
        def get(key):return idx[stage,view,h,key]
        r=get(model+'-persistence/delta_MAE')
        metric=lambda key:float(get(key)['estimate'])
        lo,hi=('lower95','upper95') if stage=='C' else ('low95','high95')
        row=dict(stage=stage,view=view,model=model,months=h,n=int(r['n']),model_MAE=metric(model+'/MAE'),persistence_MAE=metric('persistence/MAE'),delta_MAE=float(r['estimate']),lower95=float(r[lo]),upper95=float(r[hi]),model_R2=metric(model+'/R2'),model_bias=metric(model+'/bias'),calibration_slope=metric(model+'/calibration_slope'))
        assert abs(row['model_MAE']-row['persistence_MAE']-row['delta_MAE'])<1e-10
        rows.append(row)
with (O/'stage_comparison.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
plt.rcParams.update({'pdf.fonttype':42,'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(9.6,3.55),sharex=True,sharey=True)
for ax,stage,title,color,mark in zip(axs,['C','A','B'],['C   Expanded PPMI\nFour scores, refitted','A   Later PPMI\nFour scores, frozen','B   Independent PDBP\nMoCA, age and sex, frozen'],['#19685E','#85455E','#674F27'],['o','s','D']):
    ax.axvline(0,color='#7c7774',ls='--',lw=1)
    ax.set_title(title,loc='left',fontsize=10,pad=11)
    for i,r in enumerate(r for r in rows if r['stage']==stage):
        y=2-i
        ax.errorbar(r['delta_MAE'],y,xerr=[[r['delta_MAE']-r['lower95']],[r['upper95']-r['delta_MAE']]],fmt=mark,color=color,capsize=3,ms=5)
        ax.text(-.61,y+.19,'n = '+str(r['n']),fontsize=8,color='#555555')
    ax.set_xlim(-.64,.49);ax.set_ylim(-.45,2.55);ax.set_xticks([-.4,-.2,0,.2,.4]);ax.set_yticks([0,1,2]);ax.set_yticklabels(['36 months','24 months','12 months'])
    ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
fig.subplots_adjust(left=.092,right=.985,top=.77,bottom=.23,wspace=.13)
fig.text(.53,.07,'Difference in mean absolute error (model minus baseline MoCA), points',ha='center')
fig.savefig(O/'paired_error_comparison.pdf',bbox_inches='tight')
fig.savefig(O/'paired_error_comparison.png',bbox_inches='tight',dpi=360)
plt.close(fig)
report=['# What Stages A, B and C add to the iScience manuscript','',
'**Recommended central claim.** In expanded PPMI, four baseline cognitive scores improved prediction of future MoCA beyond both carrying baseline MoCA forward and fitting MoCA alone. In later PPMI recruits, the earlier frozen four-score model had similar average absolute error to carrying baseline MoCA forward, with uncertainty in both directions. In an independent PDBP cohort, a separate MoCA–age–sex model explained some variation in future scores, while its advantage over the same benchmark remained uncertain.','',
'| Analysis | Months | People | Model MAE | Baseline carried forward MAE | Paired difference (95% interval) | Model R² | Mean signed error |','|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    report.append('| {stage} | {months} | {n} | {model_MAE:.3f} | {persistence_MAE:.3f} | {delta_MAE:+.3f} ({lower95:+.3f} to {upper95:+.3f}) | {model_R2:.3f} | {model_bias:+.3f} |'.format(**r))
report+=['','All comparisons in the table are within participants at each horizon. Negative paired differences favor the model. Negative signed error means the forecast was lower than the observed score. Intervals are pointwise paired participant-bootstrap intervals conditional on the fitted or out-of-fold forecasts. Each stage prespecified 12 months as its leading horizon for this question. Earlier contracts retain their original horizon priorities.','',
'**C — contribution of additional assessments.** Stage C used grouped, nested cross-validation with regression penalties tuned on training folds. The four scores were baseline MoCA, processing speed (SDMT), and immediate and delayed verbal recall (HVLT). It included 839 unique people without requiring an RNA sample and excluded all 227 Stage A identities. The 12-month gain versus persistence was about 11% of the benchmark error. Against a fitted MoCA-only model, the reduction was 0.126 points (95% interval 0.066 to 0.186). This comparison supports incremental information in the three detailed scores. The historical sensitivity also favored the panel. On fully observed records, four-versus-28 differences included zero at each horizon. The enhanced clinical model had a small 24-month advantage in the expanded sample (0.041 points, 0.010 to 0.073). There is no consistent uniquely best panel across settings.','',
'**A — temporal transport of the four scores.** Stage A applied the earlier frozen ensemble in 2023–2025 PPMI recruits. Stage C refitted the same clinical panel under its own protocol. A and C have disjoint assessment identities, but this comparison cannot establish how the new Stage C coefficients perform prospectively. At 12 months, the panel had similar absolute error across A and C (1.661 and 1.634), while the baseline benchmark had lower error in A (1.620 versus 1.842). This helps explain how the two comparisons yield different conclusions. Their difference is descriptive. Cohort composition, model fitting and visit selection also differ. A’s mean underprediction of 0.680 points at 12 months and 0.708 at 24 months warrants reporting alongside R². The 36-month estimate rests on 39 participants. The secondary 12-month subset with center/year collection support favored four scores over 28, while the complete-case comparison included zero. Neither comparison establishes a transport advantage over persistence.','',
'**B — external transport of a different model.** Stage B applied the PPMI equations based on baseline MoCA, age and sex to 201 eligible PDBP identities at nominal annual visits. The 12-month R² of 0.549 describes squared error relative to using the target sample mean. Carrying each individual’s baseline MoCA forward already achieved MAE 1.856. The model’s paired MAE difference of +0.130 had an interval spanning zero. The visit dates were unavailable and the scoring and concordance rules restricted inclusion to participants with more than 12 years of education. The stricter education sensitivity also had intervals crossing zero. Present PDBP alongside LCC as external evidence for the shared three-input model. The four-score panel has temporal evidence within PPMI, and its independent-cohort performance remains open.','',
'**Next manuscript step.** Integrate the completed evaluations now. Lead the Summary and Results with the 12-month added value in expanded PPMI and the temporal persistence comparison in the same paragraph. Present PDBP in a separate sentence naming its three inputs. Replace the Summary’s emphasis on the peak GA R² with these results, retaining the original feature search and RNA-era results as the discovery that motivated the panel. Preserve the original primary analysis and label A/B/C as subsequent prespecified exploratory analyses.','',
'Update Figure 1 to show which models were trained, carried forward and refitted, and which cohorts evaluated each. Use the paired-error figure with its zero line for the central comparison. Add PDBP to the manuscript’s external-cohort tables alongside LCC with distinct denominators, education restrictions and nominal timing. Keep the other LCC equation with depressive symptoms identified separately. In STAR Methods give each stage’s source, outcome, benchmark, group exclusions and uncertainty procedure. Show absolute MAE and signed error alongside paired differences. No pooled estimate or total sample size across stages is warranted.','',
'**Clinical significance.** The results help investigators decide whether collecting additional cognitive tests is worthwhile when the intended endpoint is a future MoCA score. The roughly 0.2-point internal reduction is an average error difference. It does not establish detection of meaningful decline, dementia risk, a clinical action threshold or trial sample-size savings. If one further analysis is pursued, give priority to a separately prespecified independent-cohort test of the frozen four-score panel when dated SDMT, HVLT and comparable MoCA are available. The manuscript revision can proceed with the existing evidence.','',
'Figure caption. Each symbol shows the difference in mean absolute error between the model and carrying each participant’s baseline MoCA forward. Bars show pointwise paired 95% bootstrap intervals. Stage C refitted the four-score panel using grouped cross-validation. Stage A applied the earlier four-score model to later PPMI recruits. Stage B applied the separate MoCA, age and sex equations to PDBP at nominal annual visits. Negative values favor the model.','',
'Sources: ../Stage_AB_verified_results_v1/Verified_results.csv, ../Stage_C_verified_results_v1/verified_estimates.csv, their verification records, the locked protocols, and the existing iScience manuscript and Summary.']
(O/'Manuscript_recommendation.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
check()
(O/'comparison_manifest.json').write_text(json.dumps({'source_hashes':{str(k.relative_to(P)):v for k,v in seals.items()},'main_comparisons':len(rows),'new_fits':0,'paid_queries':0,'files':{f.name:digest(f) for f in O.iterdir() if f.is_file() and f.name!='comparison_manifest.json'}},indent=2)+'\n')
