exec(compile((__import__('pathlib').Path(__file__).parent/'render_current.py').read_text(), 'render_current.py', 'exec'))
# Matched mapping controls across the nine configurations.
controls=load('nine_mapping_and_calibration_controls');maps=load('nine_mapping');methods=['RIDGE-10','KRR-1','PLS-2','SVR-1','MLP-8','TT-SMALL'];short=['Ridge','KRR','PLS','SVR','MLP','TT'];keys=['R_MAE','K_MAE','MA_MAE','angle_MAE']
arm9=list(dict.fromkeys(r['arm'] for r in controls if r['mode']=='full_local'))
for page in range(3):
 fig,axs=plt.subplots(3,4,figsize=(9,7));fig.subplots_adjust(hspace=.65,wspace=.4,left=.10,right=.98,bottom=.10,top=.95)
 for row,a in enumerate(arm9[page*3:page*3+3]):
  for col,k in enumerate(keys):
   rr=[one(controls,mode='full_local',arm=a,panel='mapping',method=m) for m in methods];ax=axs[row,col];ax.errorbar(range(6),[r[k]['mean'] for r in rr],yerr=[r[k]['sd'] for r in rr],fmt='o',capsize=2,ms=3);ax.set(xticks=range(6),xticklabels=short,ylabel=['R MAE (min)','K MAE (min)','MA MAE (mm)','Angle MAE (deg)'][col]);ax.tick_params(axis='x',rotation=60,labelsize=6)
   if col==0:ax.set_title(a.replace('_to_',' → '),loc='left',fontsize=8)
 save(fig,f'S3_mapping_{page+1}')
fig,axs=plt.subplots(1,4,figsize=(10,4.5),sharey=True)
for col,k in enumerate(keys):
 ax=axs[col]
 for i,a in enumerate(arm9):
  rr=[one(maps,mode='full_local',arm=a,stage=st) for st in ['unadapted',None]];v=[r[k]['mean'] for r in rr];sd=[r[k]['sd'] for r in rr];ax.plot(v,[i,i],color='#aaaaaa');ax.errorbar(v[0],i,xerr=sd[0],fmt='o',color='#b57b31',capsize=2);ax.errorbar(v[1],i,xerr=sd[1],fmt='o',color='#3978A8',capsize=2)
 ax.set(yticks=range(9),yticklabels=[a.replace('_to_',' → ') for a in arm9],xlabel=['R MAE (min)','K MAE (min)','MA MAE (mm)','Angle MAE (deg)'][col]);ax.invert_yaxis()
fig.text(.60,.99,'Orange: source prediction    Blue: after local correction',ha='center',fontsize=9);fig.tight_layout(rect=[0,0,1,.96]);save(fig,'S4_paired_adaptation')
