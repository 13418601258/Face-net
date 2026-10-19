"""Plot S3.5 tables without recomputing experiments (PNG/PDF, every state)."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STATES=['au0','angry','disgust','fear','happy','sad','surprise']


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def panel(title):
    fig,axs=plt.subplots(2,4,figsize=(15,7),constrained_layout=True)
    fig.suptitle(title,fontsize=14)
    for ax,state in zip(axs.flat,STATES):
        ax.set_title(state);ax.grid(alpha=.2)
    axs.flat[-1].axis('off')
    return fig,list(axs.flat)


def save(fig,out,name,manifest,source):
    for ext in ['png','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=160)
    plt.close(fig);manifest.append(dict(name=name,sources=source,states=STATES,formats=['png','pdf']))


def curve(ax,rows,xkey,ykey,label):
    xs=sorted({float(r[xkey]) for r in rows})
    yy=[[float(r[ykey]) for r in rows if float(r[xkey])==x] for x in xs]
    means=[np.mean(v) for v in yy];sds=[np.std(v,ddof=1) if len(v)>1 else 0 for v in yy]
    ax.errorbar(xs,means,yerr=sds,marker='o',markersize=3,capsize=2,label=label)


def plot_all(out):
    out=Path(out);manifest=[]
    t2=read(out/'t2_results.csv');t3=read(out/'t3_results.csv');t4=read(out/'t4_results.csv');t5=read(out/'t5_results.csv')
    fig,axes=panel('T2: smooth deformation / native K (error bars: seed SD)')
    for ax,state in zip(axes,STATES):
        for c in ['base15','new5','all20']:curve(ax,[r for r in t2 if r['cohort']==c and r['state']==state],'strength','delta_per_k',c)
        ax.axhline(0,color='black',lw=.6);ax.set_xlabel('deformation / bounding-box diagonal');ax.set_ylabel('delta E_COMP / K')
    axes[-1].legend(*axes[0].get_legend_handles_labels(),loc='center')
    save(fig,out,'t2_smooth_all_states',manifest,['t2_results.csv'])
    fig,axes=panel('T2: geometry on all 20 identities (error bars: seed SD)')
    for ax,state in zip(axes,STATES):
        rr=[r for r in t2 if r['cohort']=='all20' and r['state']==state]
        curve(ax,rr,'strength','coverage_mean','coverage')
        ax.set_ylim(0,1.02);ax.set_ylabel('mean surface coverage');ax.set_xlabel('deformation strength')
    axes[-1].text(.05,.6,'2048 area samples per surface\nAll 20 identities, every seed\nTolerance = 0.001 x diagonal',transform=axes[-1].transAxes)
    save(fig,out,'t2_geometry_all_states',manifest,['t2_results.csv','geometry_subjects.csv'])
    fig,axes=panel('T2: geometry distance and normal reversal (all 20 identities)')
    for ax,state in zip(axes,STATES):
        rr=[r for r in t2 if r['cohort']=='all20' and r['state']==state]
        curve(ax,rr,'strength','distance_over_tolerance_mean','surface distance / tolerance')
        curve(ax,rr,'strength','normal_reversal_rate_mean','normal reversal rate')
        ax.set_ylabel('normalized geometry metrics');ax.set_xlabel('deformation strength')
    axes[-1].legend(*axes[0].get_legend_handles_labels(),loc='center',fontsize=9)
    save(fig,out,'t2_geometry_distance_normals',manifest,['t2_results.csv','geometry_subjects.csv'])
    for family in ['global_index','local_cycle','adjacent_swap']:
        fig,axes=panel(f'T3: {family} (error bars: seed SD; exact surface identity certified)')
        for ax,state in zip(axes,STATES):
            rr=[r for r in t3 if r['family']==family and r['state']==state]
            curve(ax,rr,'changed_fraction','delta_per_k',family)
            ax.set_xscale('log');ax.set_xlabel('actual changed vertex fraction');ax.set_ylabel('delta E_COMP / K')
        axes[-1].text(.05,.6,'Scores: all20, K=19\nIndependent per-identity permutations\nGeometry certificate: before GPA\nNot a continuous gradient test',transform=axes[-1].transAxes,fontsize=10)
        save(fig,out,'t3_'+family,manifest,['t3_results.csv'])
    fig,axes=panel('T4: nested restoration (each seed shown; all20 K=19)')
    for ax,state in zip(axes,STATES):
        for seed in range(3):
            rr=sorted([r for r in t4 if r['state']==state and int(r['seed'])==seed],key=lambda r:float(r['t']))
            ax.plot([float(r['t']) for r in rr],[float(r['delta_per_k']) for r in rr],marker='o',markersize=3,label=f'seed {seed}')
        ax.set_xlabel('restored fraction');ax.set_ylabel('delta E_COMP / K')
    axes[-1].legend(*axes[0].get_legend_handles_labels(),loc='center')
    save(fig,out,'t4_restoration_all_states',manifest,['t4_results.csv','t4_seed_checks.csv'])
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    fig.suptitle('T5: shrink in frozen aligned frame (all20 K=19; 2048 geometry samples)')
    metrics=[('delta_per_k','delta E_COMP / K'),('coverage_mean','surface coverage'),
             ('distance_over_tolerance_mean','surface distance / tolerance'),('normal_reversal_rate_mean','normal reversal rate'),
             ('geometry_loss','G'),('lambda_geom_min_tested_candidate','conditional lambda minimum')]
    for state in STATES:
        rr=sorted([r for r in t5 if r['state']==state and r['variant']=='shrink'],key=lambda r:float(r['t']))
        for ax,(key,label) in zip(axes.flat,metrics):
            ax.plot([float(r['t']) for r in rr],[float(r[key]) for r in rr],marker='o',markersize=3,label=state)
            ax.set_xlabel('t (0 = shared mean, 1 = baseline)');ax.set_ylabel(label);ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8,ncol=2)
    save(fig,out,'t5_collapse_joint_curves',manifest,['t5_results.csv','geometry_subjects.csv'])
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    rr=[next(r for r in t5 if r['state']==s and r['variant']=='copy_subject0') for s in STATES]
    for ax,(key,label) in zip(axes,[('delta_per_k','delta E_COMP / K'),('coverage_mean','surface coverage'),('geometry_loss','G')]):
        ax.bar(STATES,[float(r[key]) for r in rr]);ax.set_ylabel(label);ax.tick_params(axis='x',rotation=35);ax.grid(axis='y',alpha=.2)
    fig.suptitle('T5: copying ast001 to every identity (same-state collapse)')
    save(fig,out,'t5_copy_collapse',manifest,['t5_results.csv'])
    (out/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',default='code/reports/stage3_s35_v2')
    plot_all(p.parse_args().input)
