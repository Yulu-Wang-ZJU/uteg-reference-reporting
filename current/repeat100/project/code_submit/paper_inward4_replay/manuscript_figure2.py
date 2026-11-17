"""MA prediction, state confusion and paired signal examples from frozen outputs.

No models or gates are fitted here. The B–MA OLS line is descriptive only.
Individual signals and prediction arrays stay in private local analysis storage.
"""
from pathlib import Path
import json,hashlib,csv,string
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

WS=Path(__file__).resolve().parents[2]
ROOT=WS/'outputs/current_ma_20261103'
SYNC=ROOT/'manuscript_sync'
OUT=SYNC/'deliverables/figures/newB_20261103'
REVIEW=ROOT/'figure2_mapping_20261103'
CURVE_REVIEW=ROOT/'figure2_curve_revision_20261103'
ARMS=['target_only_SA','HZ_to_SA','HZ+HB_to_SA']
TITLES=['Target-only','Single-source transfer','Dual-source transfer']
MODEL_COLORS=['#4477AA','#CC6677','#228833']
MODEL_STYLES=['--',':','-.']
EXAMPLE_COLORS=['#4477AA','#EE6677','#228833','#CCBB44','#66CCEE',
                '#AA3377','#666666','#332288','#44AA99','#882255']
STYLE={'font.family':'Arial','font.size':7,'axes.titlesize':7.5,'axes.labelsize':7,
       'xtick.labelsize':6.5,'ytick.labelsize':6.5,'axes.spines.top':False,
       'axes.spines.right':False,'axes.linewidth':.65,'pdf.fonttype':42,
       'svg.fonttype':'none','legend.frameon':False,'savefig.facecolor':'white'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def states(y):return np.where(y<50,0,np.where(y>70,2,1))

def metrics(y,p):
    y=np.broadcast_to(y,np.asarray(p).shape).ravel();p=np.asarray(p).ravel()
    assert np.isfinite(y).all() and np.isfinite(p).all()
    t,h=states(y),states(p);cm=np.bincount(3*t+h,minlength=9).reshape(3,3)
    fn=int(cm[0,1]+cm[2,1]);fp=int(cm[1,0]+cm[1,2]);support=cm.sum(1)
    return dict(events=len(y),mae=float(np.mean(abs(p-y))),
        r2=float(1-np.sum((p-y)**2)/np.sum((y-y.mean())**2)),
        rmse=float(np.sqrt(np.mean((p-y)**2))),agreement=float(np.trace(cm)/len(y)),
        fn=fn,fn_denominator=int(support[0]+support[2]),fnr=float(fn/(support[0]+support[2])),
        fp=fp,fp_denominator=int(support[1]),fpr=float(fp/support[1]),
        low_recall=float(cm[0,0]/support[0]),high_recall=float(cm[2,2]/support[2]),confusion=cm.tolist())

def record_metrics(y,predictions):
    """Count every test record once after verifying identical repeated predictions."""
    result=[]
    for p in predictions:
        p=np.asarray(p)
        assert p.shape==(20,201)
        np.testing.assert_array_equal(p,np.broadcast_to(p[0],p.shape))
        m=metrics(y,p[0]);pooled=metrics(y,p)
        np.testing.assert_array_equal(np.asarray(m['confusion'])*20,pooled['confusion'])
        for key in ['mae','r2','rmse','agreement','fnr','fpr']:
            np.testing.assert_allclose(m[key],pooled[key],rtol=0,atol=1e-14)
        result.append(m)
    np.testing.assert_array_equal(states(predictions[1]),states(predictions[2]))
    return result


def paired_signals(d):
    """Align an existing reference-curve archive using stable, de-identified IDs."""
    p=REVIEW/'private/PAIRED_SIGNALS.npz'
    if p.exists():
        z=load(p);np.testing.assert_array_equal(z['ids'],d['ids'][:304]);return z
    index=read(WS/'outputs/uteg_cohort399_20261006_201800/INDEX_MAP.json')[:304]
    raw=load(WS/'outputs/uteg_signal_stability_abnormal_20261005_225819/private/canonical_signals.npz')
    ii=np.array([r['original_row'] for r in index])
    np.testing.assert_array_equal([r['record_id'] for r in index],d['ids'][:304])
    np.testing.assert_array_equal(raw['ids'][ii],d['ids'][:304])
    z=dict(ids=d['ids'][:304],teg=raw['teg_mm'][ii],teg_time=raw['teg_time'][ii],
           teg_mask=raw['teg_observed'][ii])
    np.testing.assert_array_equal(np.nanmax(z['teg'],axis=1),d['medical'][:304,2])
    p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**z)
    return z

EXAMPLE_SELECTION = ('Nine examples retain the nearest distinct records to equally spaced reference-MA targets. '
    'For illustration, the lowest-MA example is replaced by the nearest unused record within 1 mm '
    'with lower curve RMSE for all three models over their common observed interval; seed 1000 is fixed. '
    'This post hoc display selection does not change whole-cohort evaluation.')

def choose_examples(rows,y,readouts,teg):
    """MA-spaced examples with one explicitly post hoc illustrative replacement."""
    targets=np.linspace(y.min(),y.max(),10);chosen=[]
    for target in targets:
        ordered=np.lexsort((rows,abs(y-target)))
        pick=next(int(i) for i in ordered if int(i) not in chosen);chosen.append(pick)
    original=chosen[0]
    candidates=[int(i) for i in np.lexsort((rows,abs(y-y[original])))
                if int(i) not in chosen and abs(y[i]-y[original])<=1.0]
    audit=[]
    for candidate in candidates:
        pair=[original,candidate];observed=[]
        for idx in pair:
            row=rows[idx];mask=teg['teg_mask'][row]
            observed.append((teg['teg_time'][row,mask]/60,teg['teg'][row,mask]))
        start=max(t[0] for t,v in observed);end=min(t[-1] for t,v in observed)
        common=np.arange(np.ceil(start*12),np.floor(end*12)+1)/12
        recon,_=reconstruct_teg(readouts[:,pair,:],common)
        actual=np.array([np.interp(common,t,v) for t,v in observed])
        rmse=np.sqrt(np.mean((recon-actual[None,:,:])**2,axis=-1))
        better=bool(np.all(rmse[:,1]<rmse[:,0]))
        audit.append(dict(original_row=int(rows[original]),candidate_row=int(rows[candidate]),
            reference_ma=[float(y[i]) for i in pair],common_interval_minutes=[float(start),float(end)],
            curve_rmse_mm=rmse.tolist(),lower_for_all_three_models=better))
        if better:
            chosen[0]=candidate
            break
    assert chosen[0]!=original,'No similar-amplitude example improves all three reconstructions.'
    write(REVIEW/'private/FIRST_EXAMPLE_REPLACEMENT.json',dict(selection=EXAMPLE_SELECTION,
        comparison=audit,original_selected_row=int(rows[original]),replacement_row=int(rows[chosen[0]]),
        evaluation_predictions_changed=False))
    return rows[np.array(chosen)]

def axis_style(ax):
    ax.tick_params(length=3,width=.65,pad=2)
    ax.grid(False)
def letter(ax,value):ax.text(-.23,1.10,value,transform=ax.transAxes,fontsize=9,fontweight='bold',va='top')

def prediction_panels(fig,axes,y,pred,stats,labels='abcdef'):
    for j in range(3):
        a,b=axes[0,j],axes[1,j];p=pred[j]
        ref=np.broadcast_to(y,p.shape).ravel();flat=p.ravel()
        assert min(ref.min(),flat.min())>=30 and max(ref.max(),flat.max())<=90
        a.axvspan(50,70,color='#f1f3f4',zorder=0)
        a.plot([30,90],[30,90],color='#3E4A53',lw=.8,zorder=1)
        for cut in [50,70]:
            a.axvline(cut,color='#ADB4BA',lw=.55,ls='--');a.axhline(cut,color='#ADB4BA',lw=.55,ls='--')
        a.scatter(ref,flat,s=5,alpha=.32,color='#396B91',edgecolors='none',rasterized=False)
        a.set(xlim=(30,90),ylim=(30,90),xticks=[30,50,70,90],yticks=[30,50,70,90],
              xlabel='Reference MA (mm)',ylabel='Predicted MA (mm)',title=TITLES[j])
        a.text(.04,.96,f"MAE = {stats[j]['mae']:.2f} mm\n$R^2$ = {stats[j]['r2']:.3f}",
               transform=a.transAxes,va='top',fontsize=7,bbox=dict(facecolor='white',edgecolor='none',alpha=.88,pad=1.5))
        cm=np.array(stats[j]['confusion']);norm=cm/cm.sum(1)[:,None]
        b.imshow(norm,cmap='Blues',vmin=0,vmax=1,aspect='auto',interpolation='nearest')
        for r in range(3):
            for c in range(3):
                b.text(c,r,f'{cm[r,c]:,}\n({norm[r,c]*100:.1f}%)',ha='center',va='center',fontsize=6.5,
                       color='white' if norm[r,c]>.58 else '#17283E')
        b.set(xticks=[0,1,2],yticks=[0,1,2],xticklabels=['Low','Normal','High'],
              yticklabels=['Low','Normal','High'],xlabel='Predicted state',ylabel='Reference state')
        b.tick_params(length=0)
        for spine in b.spines.values():spine.set_visible(False)
        b.text(.5,-.35,f"Agreement {100*stats[j]['agreement']:.2f}%\nFNR {100*stats[j]['fnr']:.2f}%  |  FPR {100*stats[j]['fpr']:.2f}%",
               ha='center',va='top',transform=b.transAxes,fontsize=7,linespacing=1.45)
        letter(a,labels[j]);letter(b,labels[3+j])
        axis_style(a)


def reconstruct_teg(readouts,time_minutes):
    """Invert the manuscript's delayed exponential using predicted R, K and MA.

    MA is the asymptote A. R and R+K are the 2 and 20 mm crossings.
    The three-parameter model has no independent angle or lysis parameter.
    No measured reference trajectory enters this function.
    """
    v=np.asarray(readouts,dtype=float);R,K,A=v[...,0],v[...,1],v[...,2]
    assert np.isfinite(v).all() and np.all(K>0) and np.all(A>20)
    beta=K/np.log((A-2)/(A-20))
    tau=R+beta*np.log1p(-2/A)
    assert np.all(tau>=0) and np.all(beta>0)
    t=np.asarray(time_minutes,dtype=float)
    curve=A[...,None]*(-np.expm1(-np.maximum(t-tau[...,None],0)/beta[...,None]))
    return curve,np.stack([tau,beta,A],axis=-1)


def prediction_readouts(rows,primary_predictions):
    """Read frozen outputs for the first seed, with no fitting or seed selection."""
    folder=ROOT/'manuscript_sync/experiments/auxiliary';all_readouts=[];files=[]
    for j,arm in enumerate(ARMS):
        p=folder/('target_ridge_B60.npz' if j==0 else f'{arm}_B0.npz')
        z=load(p);files.append(p);np.testing.assert_array_equal(z['test'],rows)
        assert int(z['seed'][0])==1000
        v=(z['prediction'] if j==0 else z['complete'])[0].copy()
        if j:np.testing.assert_array_equal(v[:,2],primary_predictions[j][0])
        v[:,2]=primary_predictions[j][0]
        all_readouts.append(v)
    return np.array(all_readouts),files


def curve_examples(rows,examples,primary_predictions):
    """Use the first fixed seed for every example and every model."""
    all_readouts,files=prediction_readouts(rows,primary_predictions)
    indices=np.array([list(rows).index(int(row)) for row in examples])
    readouts=all_readouts[:,indices,:];time=np.arange(0,2701,5)/60
    curves,params=reconstruct_teg(readouts,time)
    # The inversion must retain the predicted crossing times exactly.
    R,K=readouts[...,0],readouts[...,1];tau,beta,A=params.transpose(2,0,1)
    np.testing.assert_allclose(A*(-np.expm1(-(R-tau)/beta)),2,atol=1e-12)
    np.testing.assert_allclose(A*(-np.expm1(-(R+K-tau)/beta)),20,atol=1e-12)
    assert np.isfinite(curves).all() and np.all(np.diff(curves,axis=-1)>=-1e-12)
    p=CURVE_REVIEW/'private';p.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p/'RECONSTRUCTED_EXAMPLES.npz',rows=examples,time_minutes=time,
        readouts=readouts,parameters=params,curves=curves,arms=np.array(ARMS),seed=1000)
    write(CURVE_REVIEW/'RECONSTRUCTION.json',dict(
        model='A*(1-exp(-(t-tau)+/beta)), the existing manuscript Eq. teg_exponential',
        parameter_conversion='A=predicted MA; beta=K/log((A-2)/(A-20)); tau=R+beta*log(1-2/A)',
        units='R,K,tau,beta and t in minutes; A and curves in mm',
        example_seed=1000,seed_selection='First prespecified seed for all ten records and all three models',
        readout_sources='Primary amplitude-based MA; target-only four-input Ridge R/K; source four-input Ridge plus joint Huber R/K',
        reference_trajectories_used_in_reconstruction=False,models_refitted=False,
        exact_2_and_20_mm_crossings=True,positive_onset_and_timescale=True,
        interpretation='Curves reconstructed from predicted readouts under a fixed three-parameter shape; MA is the asymptote. Angle and lysis are not fitted. Not a separately trained trajectory predictor.',
        source_files={str(p.relative_to(WS)):sha(p) for p in files}))
    return time,curves,files

def save(fig,name):
    for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{name}.{ext}',dpi=400)
    plt.close(fig)


def measured_with_display_tail(ax,t,v,color,lw=1.,ls='-',zorder=4,tail_alpha=.40):
    """Keep measured values intact; mark a last-value display extension explicitly."""
    ax.plot(t,v,color=color,lw=lw,ls=ls,zorder=zorder)
    if t[-1]<45:
        ax.plot([t[-1],45],[v[-1],v[-1]],color=color,lw=lw,ls=ls,
                alpha=tail_alpha,zorder=zorder)

def main_figure(d,rows,y,pp,ms,teg,examples,B,coef,assoc_r2,time,curves,signals):
    """Six groups: paired signals, amplitude association, cases, mapping and states."""
    fig=plt.figure(figsize=(183/25.4,194/25.4))
    fig.text(.5,.988,'U-TEG mapping and MA-state classification',
             ha='center',va='top',fontsize=9.3,fontweight='bold')
    # A–C: introduce the paired signal observations and descriptive association.
    g,h,i=[fig.add_axes([x,.842,.235,.105]) for x in [.085,.405,.725]]
    for k,row in enumerate(examples):
        col=EXAMPLE_COLORS[k];ls='-' if k<5 else '--'
        um=signals['observed_mask'][row];tm=teg['teg_mask'][row]
        g.plot((np.arange(900)[um]+1)/60,signals['scaled_zeroed'][row,um],lw=.9,color=col,ls=ls)
        measured_with_display_tail(h,teg['teg_time'][row,tm]/60,teg['teg'][row,tm],col,lw=.9,ls=ls,tail_alpha=1.)
    g.set(xlabel='Time (min)',ylabel='U-TEG (V)',xlim=(0,15),xticks=[0,5,10,15])
    h.set(xlabel='Time (min)',ylabel='TEG (mm)',xlim=(0,45),xticks=[0,15,30,45],ylim=(0,85))
    i.scatter(B,y,s=6,color='#A8B3BC',alpha=.55,edgecolors='none')
    grid=np.linspace(B.min(),B.max(),100);i.plot(grid,coef[0]+coef[1]*grid,color='#333333',lw=.9)
    for k,row in enumerate(examples):
        i.scatter(d['x'][row,1],d['medical'][row,2],s=12,color=EXAMPLE_COLORS[k],edgecolors='white',linewidths=.3,zorder=3)
    i.set(xlabel='U-TEG MA (V)',ylabel='TEG MA (mm)')
    i.text(.04,.96,f'OLS $R^2$ = {assoc_r2:.3f}',transform=i.transAxes,va='top',fontsize=6.5)
    for ax,ch in zip([g,h,i],'ABC'):
        letter(ax,ch);axis_style(ax);ax.tick_params(labelsize=6.)
        ax.xaxis.label.set_fontsize(6.3);ax.yaxis.label.set_fontsize(6.3)
    handles=[Line2D([],[],color=col,lw=1.2,ls='-' if k<5 else '--',label=str(k+1)) for k,col in enumerate(EXAMPLE_COLORS)]
    fig.legend(handles=handles,loc='center',bbox_to_anchor=(.5,.793),ncol=10,
               handlelength=1.5,columnspacing=1.05,handletextpad=.4,fontsize=6.2)
    # D: one group letter and one shared legend for all ten cases.
    fig.text(.018,.765,'D',fontsize=9,fontweight='bold',va='top')
    fig.text(.5,.765,'Measured and reconstructed TEG',ha='center',va='top',fontsize=7.5)
    handles=[Line2D([],[],color='#252525',lw=1.1,label='Measured')]
    handles += [Line2D([],[],color=col,ls=ls,lw=1.2,label=title.replace(' transfer',''))
                for col,ls,title in zip(MODEL_COLORS,MODEL_STYLES,TITLES)]
    fig.legend(handles=handles,loc='center',bbox_to_anchor=(.5,.740),ncol=4,
               handlelength=2.5,columnspacing=1.3,fontsize=6.5)
    for k,row in enumerate(examples):
        rr,cc=divmod(k,5)
        ax=fig.add_axes([.075+cc*.187,.605 if rr==0 else .450,.140,.095])
        mask=teg['teg_mask'][row]
        measured_with_display_tail(ax,teg['teg_time'][row,mask]/60,teg['teg'][row,mask],'#252525',lw=1.05,tail_alpha=1.)
        for j in range(3):ax.plot(time,curves[j,k],color=MODEL_COLORS[j],ls=MODEL_STYLES[j],lw=1.)
        ax.set(xlim=(0,45),ylim=(0,85),xticks=[0,15,30,45],yticks=[0,40,80],
               xlabel='Time (min)',ylabel='TEG (mm)')
        ax.set_title(f'{k+1} | MA {d["medical"][row,2]:.1f}',fontsize=6.3,pad=3)
        ax.tick_params(labelsize=5.6);axis_style(ax)
        ax.xaxis.label.set_size(5.8);ax.yaxis.label.set_size(5.8)
        ax.xaxis.labelpad=2;ax.yaxis.labelpad=2
    # E retains three continuous-prediction panels; F shares identical source states.
    axes=np.empty((2,3),dtype=object)
    for j,x in enumerate([.085,.405,.725]):
        axes[0,j]=fig.add_axes([x,.266,.235,.117])
        axes[1,j]=fig.add_axes([x,.076,.235,.103])
    prediction_panels(fig,axes,y,pp,ms,labels='      ')
    for j in range(3):
        a,b=axes[:,j]
        a.set_xlabel('');b.set_xlabel('')
        if j:a.set_ylabel('');b.set_ylabel('')
        for ax in [a,b]:
            ax.xaxis.label.set_fontsize(6.4);ax.yaxis.label.set_fontsize(6.4)
            ax.tick_params(labelsize=6.)
            for t in list(ax.texts):
                if t.get_text().startswith('Agreement'):t.remove()
                elif len(t.get_text().strip())>1:t.set_fontsize(6.4)
        a.set_title(TITLES[j].replace(' transfer',''),fontsize=7.3,pad=5)
        b.remove()
    unique=record_metrics(y,pp)
    for x,j,title in [(.085,0,'Target-only'),(.565,1,'Single-source and dual-source')]:
        b=fig.add_axes([x,.076,.235,.103]);cm=np.asarray(unique[j]['confusion']);norm=cm/cm.sum(1)[:,None]
        b.imshow(norm,cmap='Blues',vmin=0,vmax=1,aspect='auto',interpolation='nearest')
        for r in range(3):
            for c in range(3):
                b.text(c,r,f'{cm[r,c]:,}\n({norm[r,c]*100:.1f}%)',ha='center',va='center',fontsize=6.4,
                       color='white' if norm[r,c]>.58 else '#17283E')
        b.set(xticks=[0,1,2],yticks=[0,1,2],xticklabels=['Low','Normal','High'],
              yticklabels=['Low','Normal','High'],ylabel='Reference state' if j==0 else '')
        b.tick_params(length=0,labelsize=6.)
        b.yaxis.label.set_fontsize(6.4)
        for spine in b.spines.values():spine.set_visible(False)
        b.set_title(title,fontsize=7.3,pad=5)
        fig.text(x+.1175,.030,
                 f"Agreement {100*unique[j]['agreement']:.2f}%\nFNR {100*unique[j]['fnr']:.2f}% | FPR {100*unique[j]['fpr']:.2f}%",
                 ha='center',va='top',fontsize=6.5,linespacing=1.3)
    fig.text(.018,.399,'E',fontsize=9,fontweight='bold',va='top')
    fig.text(.5,.231,'Reference MA (mm)',ha='center',fontsize=6.5)
    fig.text(.018,.196,'F',fontsize=9,fontweight='bold',va='top')
    fig.text(.5,.048,'Predicted state',ha='center',fontsize=6.5)
    save(fig,'Figure2_translation')


def run():
    OUT.mkdir(parents=True,exist_ok=True);REVIEW.mkdir(exist_ok=True)
    d=load(ROOT/'data/DATA15_CURRENT.npz');signals=load(ROOT/'data/CURRENT_SCALED_CURVES.npz')
    np.testing.assert_array_equal(d['ids'],signals['ids'])
    artifacts=[ROOT/'data/DATA15_CURRENT.npz',ROOT/'data/CURRENT_SCALED_CURVES.npz',ROOT/'all301/T900.npz']
    pp=[];ms=[];rows=None
    for arm in ARMS:
        f=ROOT/'primary'/f'{arm}_T900_B0.npz';artifacts.append(f);z=load(f)
        if rows is None:rows=z['test']
        else:np.testing.assert_array_equal(rows,z['test'])
        assert z['prediction'].shape==(20,201)
        y=d['medical'][rows,2];p=z['prediction'];m=metrics(y,p)
        expected=read(f.with_suffix('.metrics.json'))['all_report']
        for key in ['mae','r2','agreement','fnr','fpr']:np.testing.assert_allclose(m[key],expected[key],rtol=0,atol=1e-12)
        assert m['confusion']==expected['confusion']
        pp.append(p);ms.append(m)
    z301=load(ROOT/'all301/T900.npz');y301=d['medical'][z301['local'],2]
    np.testing.assert_array_equal(y301,z301['reference']);assert list(z301['directions'])==ARMS
    ms301=[metrics(y301,p) for p in z301['prediction']]
    expected301=read(ROOT/'all301/T900.metrics.json')
    for arm,m in zip(ARMS,ms301):
        for key in ['mae','r2','agreement','fnr','fpr']:np.testing.assert_allclose(m[key],expected301[arm]['all_report'][key],rtol=0,atol=1e-12)
    teg=paired_signals(d);all_readouts,_=prediction_readouts(rows,pp)
    examples=choose_examples(rows,y,all_readouts,teg)
    assert len(set(examples))==10 and set(examples)<=set(rows)
    B=d['x'][rows,1];coef=np.linalg.lstsq(np.c_[np.ones(len(B)),B],y,rcond=None)[0]
    fitted=coef[0]+coef[1]*B;assoc_r2=float(1-np.sum((y-fitted)**2)/np.sum((y-y.mean())**2))
    association=dict(records=201,ols_intercept=float(coef[0]),ols_slope=float(coef[1]),r2=assoc_r2,
        scope='Descriptive B-MA association; not held-out prediction accuracy.',
        examples=10,selection=EXAMPLE_SELECTION)
    time,curves,curve_files=curve_examples(rows,examples,pp)
    artifacts.extend(curve_files)
    with plt.rc_context(STYLE):
        main_figure(d,rows,y,pp,ms,teg,examples,B,coef,assoc_r2,time,curves,signals)
        fig,axes=plt.subplots(2,3,figsize=(183/25.4,128/25.4))
        fig.subplots_adjust(left=.085,right=.985,bottom=.22,top=.87,wspace=.50,hspace=.60)
        fig.text(.5,.982,'MA prediction: separate 301-record internal evaluation',ha='center',va='top',fontsize=9,fontweight='bold')
        prediction_panels(fig,axes,y301,list(z301['prediction']),ms301)
        save(fig,'S13_MA_mapping_internal301')
    summary=dict(paper={a:m for a,m in zip(ARMS,ms)},internal301={a:m for a,m in zip(ARMS,ms301)},association=association)
    write(REVIEW/'SUMMARY.json',summary)
    # Counts and metrics only in public source tables; no record-level values.
    records=[]
    for protocol,vals in [('paper',ms),('paper_unique_records',record_metrics(y,pp)),('internal301',ms301)]:
        for arm,m in zip(ARMS,vals):
            row=dict(protocol=protocol,arm=arm,**{k:v for k,v in m.items() if k!='confusion'})
            row.update({f'true_{a}_pred_{b}':m['confusion'][r][s] for r,a in enumerate(['low','normal','high']) for s,b in enumerate(['low','normal','high'])})
            records.append(row)
    with (REVIEW/'AGGREGATE_METRICS.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    write(REVIEW/'private/EXAMPLE_SELECTION.json',dict(selected_rows=examples.tolist(),selected_ids=d['ids'][examples].tolist(),
        reference_ma=d['medical'][examples,2].tolist(),B=d['x'][examples,1].tolist(),selection=association['selection']))
    books=read(SYNC/'WORKBOOK_TABLES.json')
    books['2']['Figure 2 mapping']=[v for v in records if v['protocol'] in ['paper','paper_unique_records']]
    books['2']['Figure 2 association']=[association]
    count_definition={'metric': 'Figure 2 counting unit', 'definition': 'Figure 2F counts each of the 201 test records once. In Figure 2 mapping, paper_unique_records gives these counts; paper pools 20 identical predictions per record. Rates are unchanged.'}
    books['2']['Definitions']=[v for v in books['2']['Definitions'] if v.get('metric')!='Figure 2 counting unit']+[count_definition]
    books['8']['Figure S13 mapping']=[v for v in records if v['protocol']=='internal301']
    write(SYNC/'WORKBOOK_TABLES.json',books)
    write(REVIEW/'INPUT_MANIFEST.json',dict(files={str(p.relative_to(WS)):sha(p) for p in artifacts},
        private_signal_bundle_sha256=sha(REVIEW/'private/PAIRED_SIGNALS.npz'),reference_curve_MA_exact=True,
        input_version='original U-TEG B values',models_refitted=False,
        protocol_separation=True,scatter_and_confusion_use_same_predictions=True,
        paper_prediction_events=4020,internal_prediction_records=301,
        descriptive_association_records=201,public_tables_contain_individual_records=False))
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':run()
