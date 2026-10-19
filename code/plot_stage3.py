"""Redraw S3.2 from the exported curve table; never recompute experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stage3_analysis import read_csv,write_json,PANELS,TITLES


def render(folder):
    folder=Path(folder)
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    cfg=manifest['config'];curves=read_csv(folder/'curve_summary.csv')
    for r in curves:
        for key in ('strength','mean','std','minimum','maximum'):
            r[key]=float(r[key]) if r[key]!='' else None
    colors=plt.get_cmap('tab10').colors
    plt.rcParams['axes.formatter.useoffset']=False

    def draw(ax,kind,mode,metric,scale=1.):
        for i,state in enumerate(cfg['states']):
            rr=sorted([r for r in curves if r['state']==state and r['kind']==kind
                       and r['mode']==mode and r['metric']==metric and r['mean'] is not None],key=lambda r:r['strength'])
            if not rr:continue
            x=np.array([r['strength'] for r in rr])*(1 if kind=='rigid' else 100)
            ax.errorbar(x,np.array([r['mean'] for r in rr])*scale,
                yerr=np.array([r['std'] for r in rr])*scale,marker='o',markersize=3,
                linewidth=1.1,capsize=3,elinewidth=.9,label=state,color=colors[i])
            # Vertical labels preserve actual linear spacing without collisions
            # between baseline 0 and small nonzero perturbation strengths.
            ax.set_xticks(x,[f'{v:g}' for v in x],rotation=90)
        ax.grid(alpha=.22);ax.tick_params(labelsize=7)
        ax.set_xlabel('Rigid strength' if kind=='rigid' else 'Vertices (%)'
            if kind in ('index','random_missing') else 'Amplitude / box diagonal (%)',fontsize=8)

    files=[]
    def save(fig,name):
        for suffix in ('png','pdf'):
            filename=name+'.'+suffix;fig.savefig(folder/filename,dpi=180);files.append(filename)
        plt.close(fig)

    for metric,name,label in [('score','ecomp_curves','E_COMP'),('change','ecomp_delta_curves','Delta E_COMP')]:
        fig,axes=plt.subplots(2,4,figsize=(17,8.5),constrained_layout=True)
        for ax,(kind,mode),title in zip(axes.flat,PANELS,TITLES):
            ax.set_title(title,fontsize=11)
            if kind=='random_missing':
                ax.text(.5,.5,'E_COMP: not applicable\nMissing coordinates are not imputed.\nSee geometry curves.',ha='center',va='center',transform=ax.transAxes)
                ax.set_axis_off();continue
            draw(ax,kind,mode,metric);ax.set_ylabel(label)
            if metric=='change' and ((kind=='index' and mode=='common') or kind=='rigid'):ax.set_ylim(-1e-8,1e-8)
        axes.flat[0].legend(fontsize=8,ncol=2)
        fig.suptitle('15 identities per state; mean +/- SD over 3 perturbation seeds',fontsize=13)
        save(fig,name)

    for columns,name,width in [(list(range(8)),'geometry_curves',28),([5,6,7],'geometry_perturbations',13)]:
        fig,axes=plt.subplots(3,len(columns),figsize=(width,10.5),constrained_layout=True)
        for j,panel in enumerate(columns):
            kind,mode=PANELS[panel]
            for i,(metric,label,scale) in enumerate([
                ('coverage_mean','Mean subject coverage (%)',100.),
                ('distance_over_tolerance_mean','Mean distance / tolerance',1.),
                ('normal_reversal_rate_mean','Mean normal reversal (%)',100.)]):
                ax=axes[i,j];draw(ax,kind,mode,metric,scale)
                if i==0:ax.set_title(TITLES[panel],fontsize=10)
                if j==0:ax.set_ylabel(label,fontsize=9)
                if kind in ('index','rigid'):
                    ax.set_ylim((99.9,100.1) if i==0 else (-1e-7,1e-7))
                elif i==0:ax.set_ylim(0,102)
        axes[0,0].legend(fontsize=8,ncol=2)
        fig.suptitle('All 15 identities: subject mean, then mean +/- seed SD\nCoverage tolerance = 0.1% reference box diagonal',fontsize=12)
        save(fig,name)
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    write_json(folder/'figure_manifest.json',dict(
        input_csv_sha256=digest(folder/'curve_summary.csv'),
        computation_manifest_sha256=digest(folder/'manifest.json'),
        renderer_sha256=digest(Path(__file__)),
        files={n:digest(folder/n) for n in files},
        note='Presentation redraw only; same means and seed SD as curve_summary.csv'))
    print('Exported',len(files),'PNG/PDF figures to',folder)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='code/reports/stage3_v3')
    render(p.parse_args().input)
