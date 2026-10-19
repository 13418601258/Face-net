"""Three explicit index-permutation protocols for group experiments v2."""
from __future__ import annotations
import hashlib
import numpy as np


def sample_seed(seed: int, identity: str) -> int:
    """Stable across processes/order; independent streams for sample identities."""
    digest = hashlib.sha256(f'index-v2:{seed}:{identity}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'little')


def permutation(n: int, strength: float, seed: int) -> np.ndarray:
    """A cyclic permutation on a nested random subset (no fixed points in it).

    Each seed fixes an ordering of all vertices. Stronger perturbations use
    longer prefixes. Realized changed fraction is round(N*strength)/N (min 2).
    Geometry is preserved if faces are mapped by the inverse permutation.
    """
    if n < 2 or not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('need N>=2 and strength in [0,1]')
    p = np.arange(n)
    if strength == 0:
        return p
    chosen = np.random.default_rng(seed).permutation(n)[:min(n, max(2, round(n * strength)))]
    p[chosen] = np.roll(chosen, 1)
    return p


def permute_mesh(vertices, faces, p):
    inverse = np.argsort(p)
    return np.asarray(vertices)[p], inverse[np.asarray(faces)]


def perturb_group(samples, identities, *, mode, strength, seed, target=None):
    """Preserve sample membership; change vertex labels only.

    common: one permutation on every sample (invariance control).
    single: change just the specified identity (required).
    independent: derive one stream per identity from the repetition seed.
    """
    x = np.asarray(samples)
    if x.ndim != 3 or x.shape[2] != 3 or len(x) != len(identities):
        raise ValueError('expected (samples, vertices, 3) with matching identities')
    if len(set(identities)) != len(identities):
        raise ValueError('sample identities must be unique')
    if mode not in ('common', 'single', 'independent'):
        raise ValueError('unknown permutation mode')
    if mode == 'single' and target not in identities:
        raise ValueError('single mode requires an existing target identity')
    out = x.copy()
    for i, identity in enumerate(identities):
        if mode == 'single' and identity != target:
            continue
        derived = seed if mode == 'common' else sample_seed(seed, identity)
        out[i] = x[i, permutation(x.shape[1], strength, derived)]
    return out
