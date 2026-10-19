"""Plot S3.6 first-gate diagnostics from saved tables; no model rerun."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STATES=['au0','angry','disgust','fear','happy','sad','surprise']


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def panels(title):
    fig,axs=plt.subplots(2,4,figsize=(15,7),constrained_layout=True);fig.suptitle(title)
    for ax,state in zip(axs.flat,STATES):ax.set_title(state);ax.grid(alpha=.2)
    axs.flat[-1].axis('off');return fig,list(axs.flat)


def plot_all(out):
    out=Path(out);curves=read(out/'gradient_curves.csv');updates=read(out/'projected_updates.csv');figures=[]
    def save(fig,name,sources):
        for ext in ['png','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=160)
        plt.close(fig);figures.append(dict(name=name,sources=sources,formats=['png','pdf']))
    fig,axes=panels('S3.6: E_COMP along a known on-surface slide (all20, K=19)')
    for ax,state in zip(axes,STATES):
        for seed in range(3):
            rr=sorted([r for r in curves if r['state']==state and int(r['seed'])==seed],key=lambda r:float(r['t']))
            ax.plot([float(r['t']) for r in rr],[float(r['change']) for r in rr],marker='o',markersize=3,label=f'seed {seed}')
        ax.axhline(0,color='black',lw=.6);ax.set_xlabel('slide t');ax.set_ylabel('delta E_COMP')
    axes[-1].legend(*axes[0].get_legend_handles_labels(),loc='center');save(fig,'energy_paths',['gradient_curves.csv'])
    fig,axes=panels('S3.6: signed path derivative (positive supports restoration)')
    for ax,state in zip(axes,STATES):
        for seed in range(3):
            rr=sorted([r for r in curves if r['state']==state and int(r['seed'])==seed and float(r['t'])>0],key=lambda r:float(r['t']))
            line=ax.plot([float(r['t']) for r in rr],[float(r['dE_dt']) for r in rr],label=f'seed {seed}')[0]
            ax.plot([float(r['t']) for r in rr],[float(r['regpa_fd_dE_dt']) for r in rr],ls='--',color=line.get_color(),alpha=.7)
        ax.axhline(0,color='black',lw=.8);ax.set_xlabel('slide t');ax.set_ylabel('dE/dt')
    axes[-1].text(.05,.65,'Solid: frozen alignment, analytic\nDashed: fresh GPA, finite difference\nThree perturbation seeds\nNo neural network trained',transform=axes[-1].transAxes)
    save(fig,'direction_checks',['gradient_curves.csv','finite_difference_checks.csv'])
    fig,axes=panels('S3.6: actual projected update, normalized step 0.03')
    for ax,state in zip(axes,STATES):
        for seed in range(3):
            rr=sorted([r for r in updates if r['state']==state and int(r['seed'])==seed and float(r['t'])>0 and float(r['step_fraction'])==.03],key=lambda r:float(r['t']))
            ax.plot([float(r['t']) for r in rr],[100*float(r['relative_rmse_change']) for r in rr],marker='o',markersize=3,label=f'seed {seed}')
        ax.axhline(0,color='black',lw=.8);ax.set_xlabel('initial slide t');ax.set_ylabel('RMSE change (%)')
    axes[-1].text(.05,.65,'Below zero = correspondence improves\nAbove zero = correspondence worsens\nDirection: energy gradient only\nStep scale: known corruption RMS\nControlled probe, not a deployed solver',transform=axes[-1].transAxes)
    save(fig,'actual_update_rmse',['projected_updates.csv'])
    fig,axs=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    primary=[r for r in updates if float(r['t'])>0 and float(r['step_fraction'])==.03]
    for state in STATES:
        rr=[r for r in primary if r['state']==state]
        axs[0].scatter([float(r['score_after'])-float(r['score_before']) for r in rr],[100*float(r['relative_rmse_change']) for r in rr],s=12,label=state)
        axs[1].scatter([float(r['t']) for r in rr],[float(r['local_variance_ratio']) for r in rr],s=12)
        axs[2].scatter([float(r['t']) for r in rr],[float(r['normal_reversal_rate']) for r in rr],s=12)
    axs[0].axhline(0,color='black',lw=.8);axs[0].axvline(0,color='black',lw=.8);axs[0].set(xlabel='E after - E before',ylabel='Correspondence RMSE change (%)');axs[0].legend(fontsize=8)
    axs[1].axhline(.99,color='red',ls='--');axs[1].set(xlabel='initial slide t',ylabel='Local identity variance / truth')
    axs[2].set(xlabel='initial slide t',ylabel='Affected-face normal reversal rate')
    for ax in axs:ax.grid(alpha=.2)
    fig.suptitle('S3.6: score improvement versus correspondence and geometry protection')
    save(fig,'tradeoffs',['projected_updates.csv'])
    gs=read(out/'geometry_summary.csv');fig,axs=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for variant in ['before','after']:
        rr=sorted([r for r in gs if r['variant']==variant],key=lambda r:float(r['t']))
        for ax,key in zip(axs,['coverage_mean','surface_distance_over_tolerance','normal_reversal_rate_mean']):
            ax.plot([float(r['t']) for r in rr],[float(r[key]) for r in rr],marker='o',label=variant);ax.set(xlabel='slide t',ylabel=key);ax.grid(alpha=.2)
    axs[0].legend();fig.suptitle('S3.6: reconnected mesh geometry, all 20 identities / 3 seeds / 7 states')
    save(fig,'geometry_controls',['geometry_subjects.csv','geometry_summary.csv'])
    (out/'figure_manifest.json').write_text(json.dumps(figures,indent=2)+chr(10),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',default='code/reports/stage3_s36_v1');plot_all(p.parse_args().input)
