from pathlib import Path
import csv, io, zipfile, os
root=Path(__file__).resolve().parent
base=root.parent
out=base.parent
os.environ['MPLCONFIGDIR']=str(base/'QA'/'mplcache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.patches import Rectangle
from PIL import Image
def read(p):
    with p.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))
comp=read(out/'Stage_ABC_manuscript_comparison_v1/stage_comparison.csv')
cal=read(out/'Stage_J_diagnostics_v3/results/calibration.csv')
with zipfile.ZipFile(base/'Data_S1.zip') as z:
    lcc=list(csv.DictReader(io.StringIO(z.read('Table_S11_LCC_Metrics.csv').decode('utf-8-sig'))))
labels=[('Expanded PPMI','Four scores, cross-validation'),('Later PPMI','Four scores, frozen ensemble'),('PDBP','MoCA, age and sex'),('LCC','MoCA, age and sex'),('LCC','MoCA, age, sex and GDS15')]
rows=[]
for i,(cohort,model) in enumerate(labels):
    for h in (12,24,36):
        if i<3:
            stage='CAB'[i]
            r,=[r for r in cal if r['stage']==stage and int(r['h'])==h and r['metric']=='bias']
            s,=[r for r in comp if r['stage']==stage and int(r['months'])==h]
            values=[r['estimate'],r['lo'],r['hi'],s['n']]
            assert abs(float(values[0])-float(s['model_bias']))<1e-12
        else:
            m='reduced3' if i==3 else 'reduced4'
            r,=[r for r in lcc if int(r['horizon_months'])==h and r['model']==m and r['metric']=='bias']
            values=[r[k] for k in ('estimate','CI95_lower','CI95_upper','n')]
        rows.append([cohort,model,h,*values])
with (root/'Figure_S4_source.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['cohort','model','months','mean_signed_error','lower95','upper95','participants']);w.writerows(rows)
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(7.1,5.05))
fig.subplots_adjust(left=.035,right=.99,top=.82,bottom=.135)
ax.set(xlim=(-1.64,3),ylim=(5.10,-.43));ax.axis('off')
fig.text(.039,.945,'Do forecasts run high or low?',fontsize=16,weight='bold',color='#23303b')
fig.text(.039,.89,'Mean predicted minus observed MoCA, in points',fontsize=10)
cmap=LinearSegmentedColormap.from_list('signed',['#9ccac9','#f8f7f2','#deb3d1'])
norm=Normalize(-1,1)
def fmt(v):
    n=round(float(v),2)
    return ('0.00' if n==0 else f'{n:+.2f}').replace('-','−')
for k,h in enumerate((12,24,36)):ax.text(k+.5,-.2,f'{h} months',ha='center',weight='bold',fontsize=9)
for i,(cohort,model) in enumerate(labels):
    ax.text(-1.6,i+.38,cohort,fontsize=10,weight='bold')
    ax.text(-1.6,i+.67,model,fontsize=7.8)
    for j in range(3):
        a,b,h,v,lo,hi,n=rows[3*i+j]
        ax.add_patch(Rectangle((j,i+.015),.985,.955,facecolor=cmap(norm(float(v))),lw=0))
        for y,s,sz,bold in [(.25,fmt(v),11,True),(.51,f'[{fmt(lo)}, {fmt(hi)}]',8,False),(.76,f'n = {n}',7.8,False)]:
            ax.text(j+.493,i+y,s,ha='center',va='center',fontsize=sz,color='#1a2532',weight='bold' if bold else 'normal')
fig.text(.04,.085,'Negative = forecasts too low     |     Positive = forecasts too high',fontsize=8.5,weight='bold')
fig.text(.04,.038,'Brackets show pointwise 95% intervals. PDBP and LCC visits use nominal annual timing.',fontsize=8)
fig.savefig(root/'Figure_S4.pdf');fig.savefig(root/'Figure_S4.png',dpi=210)
buf=io.BytesIO();fig.savefig(buf,format='png',dpi=600);buf.seek(0)
with Image.open(buf) as im:im.convert('RGB').save(root/'Figure_S4.tiff',compression='tiff_lzw',dpi=(600,600))
