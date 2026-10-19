"""Rigid GPA v3: translation + proper rotation, no scale/reflection.

Row-vector convention: aligned = (vertices - center) @ rotation.
Missing coordinates are never implicitly imputed. GPA depends on proposed
vertex correspondences; permutation regression is therefore mandatory.
"""
from dataclasses import dataclass
import numpy as np


def proper_rotation(source, target):
    """Least-squares SO(3) rotation for already centered corresponding points."""
    a, b = np.asarray(source, float), np.asarray(target, float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or len(a)<3:
        raise ValueError('need matching (V,3) shapes with V>=3')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('rotation inputs must be finite')
    u, _, vt = np.linalg.svd(a.T @ b)
    correction = np.eye(3)
    correction[-1,-1] = 1. if np.linalg.det(u @ vt)>=0 else -1.
    return u @ correction @ vt


@dataclass
class GPAResult:
    aligned: np.ndarray
    mean: np.ndarray
    rotations: np.ndarray
    centers: np.ndarray
    iterations: int
    converged: bool
    relative_change: float
    objective: float


def rigid_gpa(samples, *, max_iterations=100, tolerance=1e-9):
    """Initialize with first centered shape, iterate to mean, fix global gauge.

    Each iteration aligns ORIGINAL centered samples to the current mean.
    A common rotation fixes the mean orientation to the first-shape frame.
    Stop on relative Frobenius change of mean; report convergence explicitly.
    """
    x = np.asarray(samples, dtype=float)
    if x.ndim!=3 or x.shape[0]<2 or x.shape[1]<3 or x.shape[2]!=3 or not np.isfinite(x).all():
        raise ValueError('GPA requires finite (N>=2,V>=3,3) samples')
    if not isinstance(max_iterations,int) or max_iterations<1 or not np.isfinite(tolerance) or tolerance<=0:
        raise ValueError('invalid GPA stopping parameters')
    centers=x.mean(axis=1)
    centered=x-centers[:,None,:]
    if (np.linalg.norm(centered,axis=(1,2))<=np.finfo(float).tiny).any():
        raise ValueError('GPA cannot orient a collapsed shape')
    anchor=centered[0]
    mean=anchor.copy()
    converged=False
    for iteration in range(1,max_iterations+1):
        rotations=np.array([proper_rotation(shape,mean) for shape in centered])
        aligned=np.einsum('nvi,nij->nvj',centered,rotations)
        updated=aligned.mean(axis=0)
        gauge=proper_rotation(updated,anchor)
        aligned=aligned@gauge
        rotations=rotations@gauge
        updated=updated@gauge
        change=float(np.linalg.norm(updated-mean)/max(np.linalg.norm(mean),np.finfo(float).tiny))
        mean=updated
        if change<=tolerance:
            converged=True
            break
    objective=float(np.mean((aligned-mean)**2))
    return GPAResult(aligned,mean,rotations,centers,iteration,converged,change,objective)


def match_group_frame(aligned, reference_mean):
    """Common rigid gauge only, for geometry comparisons after independent GPA."""
    center=aligned.mean(axis=(0,1))
    target_center=reference_mean.mean(axis=0)
    r=proper_rotation(aligned.mean(axis=0)-center,reference_mean-target_center)
    return (aligned-center)@r+target_center


def apply_rigid_group(samples, identities, *, mode, seed, strength):
    """Exact common R,t or independent R,t; all computations in float64."""
    from index_protocol import sample_seed
    x=np.asarray(samples,float)
    if mode not in ('common','independent') or len(identities)!=len(x) or strength<0:
        raise ValueError('invalid rigid protocol')
    # RMS-radius scale is rigid invariant, unlike an axis-aligned box diagonal.
    centers=x.mean(axis=1)
    radius=float(np.sqrt(np.mean((x-centers[:,None])**2)))
    rotations=[];translations=[]
    for identity in identities:
        rng=np.random.default_rng(seed if mode=='common' else sample_seed(seed,identity+'/rigid-v3'))
        ax,ay,az=rng.uniform(-np.pi/12,np.pi/12,3)*strength
        cx,cy,cz=np.cos([ax,ay,az]);sx,sy,sz=np.sin([ax,ay,az])
        rx=np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
        ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
        rz=np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
        rotations.append((rz@ry@rx).T)
        translations.append(rng.uniform(-.5,.5,3)*radius*strength)
    rotations=np.asarray(rotations);translations=np.asarray(translations)
    return np.einsum('nvi,nij->nvj',x,rotations)+translations[:,None,:],rotations,translations


def score_status(samples, policy):
    """An explicit geometry-only branch: missing E_COMP is null, never zero."""
    if policy!='geometry_only':
        raise ValueError('only reviewed geometry_only missing policy is implemented')
    return 'not_applicable_missing_coordinates' if not np.isfinite(samples).all() else 'finite'
