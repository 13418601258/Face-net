"""Plot frozen v2 proxy-anchor endpoints and closed-loop trajectories."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARMS=['anchor_only','anchor_fixed_e','anchor_gated_e','e_only']


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def plot_all(out):
    out=Path(out);rows=read(out/'endpoints.csv');traces=read(out/'trajectories.csv');figures=[]
    def save(fig,name,sources):
        for ext in ['png','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=160)
        plt.close(fig);figures.append(dict(name=name,sources=sources))
    fig,axs=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    levels=[0,.05,.1,.5,1]
    for arm in ARMS:
        rr=[r for r in rows if r['arm']==arm]
        for ax,key in zip(axs,['final_nonanchor_nrmse','final_anchor','local_variance_ratio']):
            ax.plot(levels,[np.median([float(r[key]) for r in rr if float(r['t'])==t]) for t in levels],marker='o',label=arm)
    for ax,label in zip(axs,['Non-anchor endpoint NRMSE','Sparse anchor residual','Local identity variance / truth']):
        ax.set(xlabel='Initial corruption level t',ylabel=label);ax.grid(alpha=.2)
    axs[0].legend(fontsize=8);axs[0].set_yscale('symlog',linthresh=1e-7);axs[1].set_yscale('symlog',linthresh=1e-10)
    fig.suptitle('S3.6 v2: synthetic sparse-anchor mechanism pilot; no neural-training release')
    save(fig,'v2_endpoints',['endpoints.csv'])
    fig,axs=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for arm in ARMS:
        rr=[r for r in traces if r['state']=='happy' and int(r['seed'])==0 and r['arm']==arm and float(r['t'])==1]
        for ax,key in zip(axs,['nonanchor_nrmse','anchor']):
            ax.plot([int(r['iteration']) for r in rr],[float(r[key]) for r in rr],label=arm)
    for ax,label in zip(axs,['Non-anchor NRMSE','Sparse anchor residual']):
        ax.set(xlabel='Iteration',ylabel=label);ax.grid(alpha=.2);ax.set_yscale('symlog',linthresh=1e-10)
    axs[0].legend(fontsize=8);fig.suptitle('happy / seed 0 / t=1: full optimization paths; iteration-limit exits are not convergence')
    save(fig,'v2_closed_loop',['trajectories.csv'])
    pairs=read(out/'paired_endpoints.csv');fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
    for j,arm in enumerate(ARMS[1:]):
        rr=[r for r in pairs if r['arm']==arm and float(r['t'])>0]
        ax.scatter([float(r['t'])+(j-1)*.012 for r in rr],[float(r['delta_nrmse_vs_anchor']) for r in rr],s=12,alpha=.7,label=arm)
    ax.axhline(0,color='black',lw=.8);ax.set(xlabel='Initial corruption level t (small plotting jitter)',ylabel='Endpoint NRMSE minus anchor-only');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Paired endpoint effects: below zero beats sparse anchors alone')
    save(fig,'v2_paired_gain',['paired_endpoints.csv'])
    (out/'figure_manifest.json').write_text(json.dumps(figures,indent=2)+chr(10),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',default='code/reports/stage3_s36_v2_harmonic');plot_all(p.parse_args().input)
