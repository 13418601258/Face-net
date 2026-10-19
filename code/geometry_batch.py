"""Exact sampled point-to-triangle metrics with radius-binned search.

Same metric definitions as metrics.py. Radius bins avoid letting one large
triangle inflate every query. Each bin uses a conservative triangle bound.
"""
import numpy as np
from scipy.spatial import cKDTree
from metrics import TriangleSurface, _mesh, _paired_triangle_distance


class BinnedSurface(TriangleSurface):
    def __init__(self, vertices, faces, valid_mask=None):
        super().__init__(vertices, faces, valid_mask)
        self.bins=[]
        if not len(self.triangles):
            return
        radii=np.linalg.norm(self.triangles-self.centers[:,None],axis=2).max(axis=1)
        labels=np.floor(np.log2(np.maximum(radii,self.max_radius/256)/self.max_radius)).astype(int)
        for label in np.unique(labels):
            indexes=np.flatnonzero(labels==label)
            self.bins.append((indexes,cKDTree(self.centers[indexes]),float(radii[indexes].max())))

    def distances(self,points):
        points=np.asarray(points,float)
        if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
            raise ValueError('query points must be finite (N,3)')
        if not len(self.triangles):return np.full(len(points),np.inf)
        result=np.empty(len(points))
        for start in range(0,len(points),128):
            p=points[start:start+128]
            _,nearest=self.tree.query(p)
            best=_paired_triangle_distance(p,self.triangles[nearest])
            for indexes,tree,radius in self.bins:
                # Any better triangle has centroid distance <= best + radius.
                lists=tree.query_ball_point(p,np.nextafter(best+radius,np.inf))
                lengths=np.array([len(v) for v in lists])
                if not lengths.sum():continue
                owners=np.repeat(np.arange(len(p)),lengths)
                candidates=indexes[np.concatenate(lists).astype(int)]
                values=_paired_triangle_distance(p[owners],self.triangles[candidates])
                np.minimum.at(best,owners,values)
            result[start:start+len(p)]=best
        return result


def evaluate_batch_geometry(reference_vertices,reference_faces,vertices,faces,*,
                            valid_mask=None,tolerance,sample_count=2048,seed=0):
    """Identical definitions and sampling RNG to metrics.evaluate_geometry."""
    if not np.isfinite(tolerance) or tolerance<0:raise ValueError('invalid tolerance')
    ref=_mesh(reference_vertices,reference_faces)
    cand=_mesh(vertices,faces,valid_mask)
    if not ref[2].all() or not ref[3].any():raise ValueError('invalid reference')
    if ref[1].shape!=cand[1].shape:raise ValueError('matching face rows required')
    rs=BinnedSurface(reference_vertices,reference_faces)
    cs=BinnedSurface(vertices,faces,valid_mask)
    forward=cs.distances(rs.sample(sample_count,seed))
    backward=rs.distances(cs.sample(sample_count,seed)) if len(cs.triangles) else None
    mask=ref[3]&cand[3]
    comparable=int(mask.sum())
    reversed_count=int((np.einsum('ij,ij->i',ref[4][mask],cand[4][mask])<0).sum())
    return dict(coverage=float(np.mean(forward<=tolerance)),
        reference_to_candidate_mean=float(forward.mean()) if backward is not None else None,
        candidate_to_reference_mean=float(backward.mean()) if backward is not None else None,
        symmetric_mean=float((forward.mean()+backward.mean())/2) if backward is not None else None,
        reference_to_candidate_p95=float(np.quantile(forward,.95)) if backward is not None else None,
        normal_reversals=reversed_count,comparable_faces=comparable,
        normal_reversal_rate=reversed_count/comparable if comparable else None,
        valid_vertex_fraction=float(cand[2].mean()),usable_faces=int(cand[3].sum()),
        degenerate_faces=int(cand[6].sum()),reference_area=float(rs.area.sum()),
        candidate_area=float(cs.area.sum()),tolerance=float(tolerance),sample_count=sample_count,
        seed=seed,distance_method='exact triangle distance; radius-binned conservative search',
        alignment='none; caller supplies shared frame')


def identical_geometry(vertices,faces,*,tolerance,seed):
    """Analytic identity result, used only after exact surface identity checks."""
    m=_mesh(vertices,faces)
    n=int(m[3].sum());area=float(m[5][m[3]].sum())
    return dict(coverage=1.,reference_to_candidate_mean=0.,candidate_to_reference_mean=0.,
        symmetric_mean=0.,reference_to_candidate_p95=0.,normal_reversals=0,comparable_faces=n,
        normal_reversal_rate=0.,valid_vertex_fraction=1.,usable_faces=n,
        degenerate_faces=int(m[6].sum()),reference_area=area,candidate_area=area,
        tolerance=float(tolerance),sample_count=0,seed=seed,
        distance_method='analytic identical surface; exact permutation/inverse check',alignment='native')
