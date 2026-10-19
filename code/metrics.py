"""Surface health measures in a shared coordinate frame.

Distances are to triangle surfaces, not nearest vertices. Coverage is a
Monte Carlo area estimate from reference -> candidate, with explicit seed,
sample count and tolerance. Normal reversals compare corresponding triangle
rows and are only a proxy for flips, NOT a self-intersection detector.
No automatic alignment: index-based alignment would contaminate shuffle tests.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def _mesh(vertices, faces, valid_mask=None):
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces)
    if v.ndim != 2 or v.shape[1] != 3 or not len(v):
        raise ValueError('vertices must be nonempty (N,3)')
    if f.ndim != 2 or f.shape[1] != 3 or not np.issubdtype(f.dtype, np.integer):
        raise ValueError('faces must be integer (F,3)')
    if f.size and (f.min() < 0 or f.max() >= len(v)):
        raise ValueError('face indices out of bounds')
    valid = np.isfinite(v).all(axis=1)
    if valid_mask is not None:
        mask = np.asarray(valid_mask)
        if mask.shape != (len(v),) or mask.dtype != bool:
            raise ValueError('valid_mask must be boolean (N,)')
        valid &= mask
    face_valid = valid[f].all(axis=1)
    normals = np.zeros((len(f), 3))
    triangles = v[f[face_valid]]
    normals[face_valid] = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    twice_area = np.linalg.norm(normals, axis=1)
    # Relative threshold preserves behavior under a change of length units.
    extent = np.linalg.norm(np.ptp(v[valid], axis=0)) if valid.any() else 0.
    degenerate = face_valid & (twice_area <= np.finfo(float).eps * extent**2)
    usable = face_valid & ~degenerate
    return v, f, valid, usable, normals, twice_area / 2., degenerate


def _paired_triangle_distance(points, triangles):
    """Exact minimum of perpendicular interior projection and three edges."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac, ap = b-a, c-a, points-a
    dot = lambda x,y: np.einsum('ij,ij->i', x,y)
    aa, bb, cc = dot(ab,ab), dot(ab,ac), dot(ac,ac)
    determinant = aa*cc - bb*bb
    u = np.divide(cc*dot(ap,ab)-bb*dot(ap,ac), determinant,
                  out=np.zeros(len(points)), where=determinant>0)
    w = np.divide(aa*dot(ap,ac)-bb*dot(ap,ab), determinant,
                  out=np.zeros(len(points)), where=determinant>0)
    inside = (determinant>0) & (u>=0) & (w>=0) & (u+w<=1)
    projected = a + u[:,None]*ab + w[:,None]*ac
    squared = np.where(inside, dot(points-projected, points-projected), np.inf)
    for start,end in ((a,b),(b,c),(c,a)):
        edge = end-start
        t = np.divide(dot(points-start,edge),dot(edge,edge),
                      out=np.zeros(len(points)), where=dot(edge,edge)>0)
        diff = points - (start + np.clip(t,0,1)[:,None]*edge)
        squared = np.minimum(squared, dot(diff,diff))
    return np.sqrt(np.maximum(squared,0))


class TriangleSurface:
    def __init__(self, vertices, faces, valid_mask=None):
        v,f,valid,usable,_,area,_ = _mesh(vertices,faces,valid_mask)
        self.triangles = v[f[usable]]
        self.area = area[usable]
        if len(self.triangles):
            self.centers = self.triangles.mean(axis=1)
            self.tree = cKDTree(self.centers)
            self.max_radius = np.linalg.norm(self.triangles-self.centers[:,None],axis=2).max()

    def sample(self, count, seed):
        if not isinstance(count, (int,np.integer)) or count < 1:
            raise ValueError('sample count must be positive')
        if not len(self.triangles):
            raise ValueError('cannot sample an empty surface')
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(self.triangles), count, p=self.area/self.area.sum())
        uv = rng.random((count,2))
        root = np.sqrt(uv[:,0])
        weights = np.column_stack((1-root,root*(1-uv[:,1]),root*uv[:,1]))
        return np.einsum('ni,nij->nj',weights,self.triangles[chosen])

    def distances(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
            raise ValueError('query points must be finite (N,3)')
        if not len(self.triangles):
            return np.full(len(points),np.inf)
        distances = np.empty(len(points))
        for start in range(0,len(points),128):
            p = points[start:start+128]
            _,nearest = self.tree.query(p)
            upper = _paired_triangle_distance(p,self.triangles[nearest])
            # If triangle distance < upper, its centroid distance <= upper +
            # its radius <= upper + max_radius. Thus no nearer face is missed.
            radius = np.nextafter(upper+self.max_radius,np.inf)
            lists = self.tree.query_ball_point(p,radius)
            lengths = np.array([len(x) for x in lists])
            owners = np.repeat(np.arange(len(p)),lengths)
            indexes = np.concatenate(lists).astype(int)
            values = _paired_triangle_distance(p[owners],self.triangles[indexes])
            best = np.full(len(p),np.inf)
            np.minimum.at(best,owners,values)
            distances[start:start+len(p)] = best
        return distances


def evaluate_geometry(reference_vertices, reference_faces, vertices, faces, *,
                      valid_mask=None, tolerance, sample_count=2048, seed=0,
                      corresponding_faces=True):
    """Bidirectional distances, reference-area coverage, reversal proxy.

    corresponding_faces=True requires matching triangle ROW identities, not
    identical vertex numbers. A vertex shuffle must inverse-remap its faces.
    Empty candidate: coverage=0, distances=None, no comparable normals.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('tolerance must be finite and nonnegative')
    ref = _mesh(reference_vertices, reference_faces)
    cand = _mesh(vertices, faces, valid_mask)
    if not ref[2].all() or not ref[3].any():
        raise ValueError('reference needs finite vertices and a nonempty surface')
    rs = TriangleSurface(reference_vertices,reference_faces)
    cs = TriangleSurface(vertices,faces,valid_mask)
    sampled_ref = rs.sample(sample_count,seed)
    forward = cs.distances(sampled_ref)
    backward = rs.distances(cs.sample(sample_count,seed)) if len(cs.triangles) else None
    def mean_or_none(d):
        return float(np.mean(d)) if d is not None and np.isfinite(d).all() else None
    comparable = flipped = None
    if corresponding_faces:
        if ref[1].shape != cand[1].shape:
            raise ValueError('matching triangle rows required for normal comparisons')
        mask = ref[3] & cand[3]
        comparable = int(mask.sum())
        dots = np.einsum('ij,ij->i',ref[4][mask],cand[4][mask])
        flipped = int((dots < 0).sum())
    return dict(
        coverage=float(np.mean(forward<=tolerance)),
        reference_to_candidate_mean=mean_or_none(forward),
        candidate_to_reference_mean=mean_or_none(backward),
        symmetric_mean=(float((forward.mean()+backward.mean())/2) if backward is not None else None),
        reference_to_candidate_p95=(float(np.quantile(forward,.95)) if backward is not None else None),
        normal_reversals=flipped, comparable_faces=comparable,
        normal_reversal_rate=(flipped/comparable if comparable else None),
        valid_vertex_fraction=float(cand[2].mean()),
        usable_faces=int(cand[3].sum()), degenerate_faces=int(cand[6].sum()),
        reference_area=float(rs.area.sum()), candidate_area=float(cs.area.sum()),
        tolerance=float(tolerance), sample_count=sample_count, seed=seed,
        distance_method='exact triangle distance for sampled surface points',
        alignment='none; shared coordinate frame required',
    )


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',required=True,help='NPZ with vertices/faces')
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--tolerance',required=True,type=float)
    parser.add_argument('--samples',default=2048,type=int)
    parser.add_argument('--seed',default=0,type=int)
    parser.add_argument('--output')
    parser.add_argument('--unmatched-faces',action='store_true')
    args=parser.parse_args()
    with np.load(args.reference,allow_pickle=False) as r, np.load(args.candidate,allow_pickle=False) as c:
        result=evaluate_geometry(r['vertices'],r['faces'],c['vertices'],c['faces'],
            valid_mask=c.get('valid_mask'),tolerance=args.tolerance,
            sample_count=args.samples,seed=args.seed,corresponding_faces=not args.unmatched_faces)
    text=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)
    if args.output:
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(text,encoding='utf-8')
    print(text)


if __name__=='__main__':
    main()
