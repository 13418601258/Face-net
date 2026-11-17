import numpy as np, sys
sys.path.insert(0,'code')
from audit_io import read_obj

def build_adj(F, n):
    tri_of = [[] for _ in range(n)]
    for t,(a,b,c) in enumerate(F):
        tri_of[a].append(t); tri_of[b].append(t); tri_of[c].append(t)
    return [np.array(x,dtype=np.int64) for x in tri_of]

def closest_on_tri(p, A, B, C):
    """Ericson: closest point on triangle ABC to p. Returns (point, dist2)."""
    ab=B-A; ac=C-A; ap=p-A
    d1=ab@ap; d2=ac@ap
    if d1<=0 and d2<=0: return A
    bp=p-B; d3=ab@bp; d4=ac@bp
    if d3>=0 and d4<=d3: return B
    vc=d1*d4-d3*d2
    if vc<=0 and d1>=0 and d3<=0:
        v=d1/(d1-d3+1e-30); return A+v*ab
    cp=p-C; d5=ab@cp; d6=ac@cp
    if d6>=0 and d5<=d6: return C
    vb=d5*d2-d1*d6
    if vb<=0 and d2>=0 and d6<=0:
        w=d2/(d2-d6+1e-30); return A+w*ac
    va=d3*d6-d5*d4
    if va<=0 and (d4-d3)>=0 and (d5-d6)>=0:
        w=(d4-d3)/((d4-d3)+(d5-d6)+1e-30); return B+w*(C-B)
    den=1.0/(va+vb+vc+1e-30); v=vb*den; w=vc*den
    return A+ab*v+ac*w

def surface_info(V,F,adj,L,K=14):
    out=[]
    for p in L:
        d=np.linalg.norm(V-p,axis=1)
        near=np.argsort(d)[:K]
        tris=np.unique(np.concatenate([adj[i] for i in near]))
        best=None
        for t in tris:
            q=closest_on_tri(p,V[F[t,0]],V[F[t,1]],V[F[t,2]])
            dd=np.linalg.norm(q-p)
            if best is None or dd<best[0]: best=(dd,t,q)
        dd,t,q=best
        n=np.cross(V[F[t,1]]-V[F[t,0]],V[F[t,2]]-V[F[t,0]]); n/=np.linalg.norm(n)+1e-15
        out.append((dd,float((p-q)@n),t))
    return out

for subj,st in [('ast001','happy'),('ast007','disgust'),('ast016','happy'),('ast016','surprise')]:
    m=read_obj(f'data/ASTFace_Public/{subj}/Standardized Meshes/{subj}_{st}.obj')
    V,F=m.vertices,m.faces
    e=np.vstack([V[F[:,0]]-V[F[:,1]],V[F[:,1]]-V[F[:,2]],V[F[:,2]]-V[F[:,0]]])
    el=float(np.median(np.linalg.norm(e,axis=1)))
    L=np.loadtxt(f'data/ASTFace_Public/{subj}/landmarks/{subj}_{st}.txt')*0.01
    adj=build_adj(F,len(V))
    r=surface_info(V,F,adj,L)
    d=np.array([x[0] for x in r]); s=np.array([x[1] for x in r])
    print('%-9s/%-9s 边长%.5f | 点面距离 中位%.4f=%.1f边 q90%.4f=%.1f边 max%.4f=%.1f边 | 内外 %d/%d'%(
        subj,st,el,np.median(d),np.median(d)/el,np.quantile(d,.9),np.quantile(d,.9)/el,
        d.max(),d.max()/el,(s<0).sum(),(s>0).sum()))

print()
print('=== 对照：随机点云 vs 真实 landmark（ast001/happy）===')
m=read_obj('data/ASTFace_Public/ast001/Standardized Meshes/ast001_happy.obj')
V,F=m.vertices,m.faces
e=np.vstack([V[F[:,0]]-V[F[:,1]],V[F[:,1]]-V[F[:,2]],V[F[:,2]]-V[F[:,0]]])
el=float(np.median(np.linalg.norm(e,axis=1)))
adj=build_adj(F,len(V))
L=np.loadtxt('data/ASTFace_Public/ast001/landmarks/ast001_happy.txt')*0.01
rng=np.random.default_rng(0)
def stat(P,tag):
    r=surface_info(V,F,adj,P)
    d=np.array([x[0] for x in r])
    print('  %-22s 中位%.4f=%.1f边 q90%.1f边'%(tag,np.median(d),np.median(d)/el,np.quantile(d,.9)/el))
stat(L,'真实 landmark')
lo,hi=V.min(0),V.max(0)
for k in range(3):
    stat(rng.uniform(lo,hi,size=(84,3)),'均匀随机 %d'%(k+1))
stat(V[rng.choice(len(V),84,replace=False)],'网格顶点采样')
