import numpy as np
from pathlib import Path


def read_obj(p):
    V, F = [], []
    for line in open(p, encoding='utf-8', errors='ignore'):
        if line.startswith('v '):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith('f '):
            idx = [int(t.split('/')[0]) - 1 for t in line.split()[1:]]
            for k in range(1, len(idx) - 1):
                F.append([idx[0], idx[k], idx[k + 1]])
    return np.array(V), np.array(F)


base = Path('data/ASTFace_Public')
for subj, state in [('ast001', 'au0'), ('ast007', 'disgust'), ('ast016', 'happy')]:
    obj = next(base.rglob(f'{subj}_{state}.obj'))
    txt = next((q for q in base.rglob(f'{subj}_{state}*.txt')), None)
    V, F = read_obj(obj)
    e = np.vstack([V[F[:, 0]] - V[F[:, 1]], V[F[:, 2]] - V[F[:, 0]]])
    med_edge = float(np.median(np.linalg.norm(e, axis=1)))
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    print(f'\n=== {subj}_{state}  V={len(V)} diag={diag:.4f} medEdge={med_edge:.6f}')
    print(f'    txt = {txt}')
    if txt is None:
        continue
    lm = np.loadtxt(txt, dtype=float)
    print(f'    landmark raw range = {lm.min(0).round(3)} .. {lm.max(0).round(3)}')
    best = None
    for s in [0.002, 0.004, 0.006, 0.008, 0.009, 0.0095, 0.01, 0.0105, 0.011, 0.012, 0.015, 0.02, 1.0]:
        P = lm * s
        d = np.linalg.norm(P[:, None, :] - V[None, :, :], axis=2).min(1)
        m = float(np.median(d))
        if best is None or m < best[1]:
            best = (s, m)
        print(f'    scale={s:<8} median_dist={m:.5f}  ({m / med_edge:.2f} x edge)')
    print(f'    -> best scale {best[0]} with median {best[1]:.5f} ({best[1] / med_edge:.2f} x edge)')
