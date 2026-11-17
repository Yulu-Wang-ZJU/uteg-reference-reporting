import os
import os
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
# Text changes: current repeated results versus explicitly retained fixed-cohort figures.
p=M/'supplement_current.tex';s=p.read_text(encoding='utf8')
s=s.replace('All stages, readouts, seeds and marginal intervals are provided in Supplementary Data~2 and~5.','Points and error bars show mean and SD over 100 repeats with full local development pools. All readouts and repeats are provided in Supplementary Data~2 and~5.').replace('All retained stages, readouts, seeds and marginal intervals are provided in Supplementary Data~2 and~5.','Points and error bars show mean and SD over 100 repeats with full local development pools. All stages and repeats are provided in Supplementary Data~2 and~5.')
import re
rr=[(one(maps,mode='full_local',arm=a,stage='unadapted')['MA_MAE']['mean'],one(maps,mode='full_local',arm=a,stage=None)['MA_MAE']['mean']) for a in arm9]
gains=[a-b for a,b in rr]
new=f'Local correction reduced mean MA error by {min(gains):.2f}--{max(gains):.2f}~mm across the nine configurations. We compare source predictions and corrected predictions within each matched repeat. Figure~'+r'\ref{sfig:current_S4_paired_adaptation}'+' shows the mean errors and SD; Supplementary Data~2 retains the complete repeat-level results.'
s=re.sub(r'Local correction reduced MA error in all nine configurations,.*?electronic table retains paired intervals\.',lambda m:new,s)
s=s.replace('Lines connect paired aggregate errors for the same records and seeds.','Lines connect mean errors from matched repeats; error bars show SD across 100 repeats.').replace('Paired MA intervals and all class transitions remain in the electronic tables.','All repeated results and the original fixed-cohort class transitions remain separately labelled in the electronic tables.')
s=s.replace('Supplementary Data~2 (``Figure 2 mapping\'\') also provides counts pooled over 20 repeats; rates and error metrics are identical.','Supplementary Data~2 also provides the separate 100-repeat prediction summaries. We do not count identical repeated predictions as independent records.')
s=s.replace('All seeds remain paired within each sampled record group.','Comparisons remain paired within each repeat.').replace('Counts are record--seed events, including both beneficial and harmful transitions.','This retained fixed-cohort transition analysis counts record--seed events, including both beneficial and harmful transitions.')
for centre in ['SAHZU','Huzhou','Hubei']:
 s=s.replace('All-candidate score-ranking curves with '+centre+' as target','Retained fixed-cohort score-ranking curves with '+centre+' as target')
s=s.replace('B95 first-report trajectories in all nine source configurations.','Retained fixed-cohort B95 first-report trajectories in all nine source configurations.').replace('Matched signal-refit comparison.','Retained fixed-cohort signal-refit comparison.')
s=s.replace('Existing adaptation budgets with fixed source models and test cohorts. Full budgets are Huzhou 40, SAHZU 60 and Hubei 40 records. Zero denotes source prediction without target adaptation. Errors are evaluated before report selection.','MA errors at matched local budgets of 20, 40 and 60 records. Points and error bars show mean and SD over 100 repeats; prediction is assessed before report selection. The original all-centre budget analysis remains separately labelled in Supplementary Data~5.')
s=s.replace('Source models and test cohorts are fixed. FULL uses Huzhou C40, SAHZU C60 and Hubei C40. Zero denotes source prediction without target adaptation. Errors are evaluated before report selection.','Current 20/40/60-record comparisons use matched sampled source and test records within each repeat. The original fixed-cohort all-centre budgets retain their own labels.')
lin=load('linear');lf=[one(lin,budget=60,arm=a,rule='All report')['FNR'] for a in arms];fp=[one(lin,budget=60,arm=a,rule='All report')['FPR'] for a in arms]
old=re.search(r'We compared one-input linear and RBF models.*?sheet ``Linear-RBF comparison\'\'\.',s).group()
new=r'We compared one-input linear and RBF models with matched splits, class weights and asymmetric loss (fixed penalty 0.01). With 60 development records, neither source RBF model produced an abnormal-to-normal error across the 100 repeats. Linear regression had an FNR of $13.07\pm3.16\%$ for both source configurations. The corresponding FPRs were $18.64\pm1.11\%$ and $18.70\pm1.16\%$ for single- and dual-source RBF, versus $6.95\pm0.91\%$ for linear regression. This comparison separates the reduction in FN from its FP cost. Supplementary Data~2 retains both development budgets; the original 301-record comparison remains separately labelled in Supplementary Data~8.'
# Derive exact SD rather than copy rounded assumptions.
new=new.replace('13.07\\pm3.16',f"{lf[0]['mean']*100:.2f}\\pm{lf[0]['sd']*100:.2f}").replace('6.95\\pm0.91',f"{fp[0]['mean']*100:.2f}\\pm{fp[0]['sd']*100:.2f}")
s=s.replace(old,new);p.write_text(s,encoding='utf8')
print('S3/S4 and remaining evaluation scopes updated')
