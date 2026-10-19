"""Redraw S3.3 stability figures from exported CSV tables only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def render(folder):
    folder=Path(folder);identity=read(folder/'identity_stability.csv')
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    pairs=['new5','all20'];selected=[]
    for right in pairs:
        rows=[r for r in identity if r['right_cohort']==right and r['applicable']=='True']
        selected.append(rows)
    labels=[f"{r['state']}\n{r['kind']}" for r in selected[0]]
    vals=np.array([[float(r['spearman_rho']) for r in rows] for rows in selected]).T
    fig,axes=plt.subplots(1,2,figsize=(10, max(5,len(labels)*.18)),sharey=True,constrained_layout=True)
    for j,right in enumerate(pairs):
        image=axes[j].imshow(vals[:,j:j+1],vmin=-1,vmax=1,cmap='coolwarm',aspect='auto')
        axes[j].set_xticks([0],[f'base15 vs {right}']);axes[j].tick_params(axis='x',labelrotation=25)
        if j==0:axes[j].set_yticks(np.arange(len(labels)),labels,fontsize=7)
        else:axes[j].set_yticks(np.arange(len(labels)),[])
        for i,value in enumerate(vals[:,j]):axes[j].text(0,i,f'{value:.2f}',ha='center',va='center',fontsize=7)
    fig.colorbar(image,ax=axes.ravel().tolist(),label='Spearman rho')
    fig.suptitle('S3.3 identity curve-shape stability (delta E_COMP / native K)',fontsize=12)
    files=[]
    for suffix in ('png','pdf'):
        name='identity_stability_heatmap.'+suffix;fig.savefig(folder/name,dpi=180);files.append(name)
    plt.close(fig)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    output=dict(input_csv_sha256=sha(folder/'identity_stability.csv'),
        computation_manifest_sha256=sha(folder/'manifest.json'),renderer_sha256=sha(Path(__file__)),
        files={n:sha(folder/n) for n in files},note='Presentation redraw only; all values read from identity_stability.csv')
    (folder/'figure_manifest.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Exported',files)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',default='code/reports/stage3_stability_v1');render(p.parse_args().input)
