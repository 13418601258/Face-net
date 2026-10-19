"""Fixed-dimension, fixed-reference log-spectrum score (protocol v2).

The reference variance is estimated once from an unperturbed cohort. All
comparisons must use the same N, D, scale, delta, preprocessing and log base.
This is a regularized compactness surrogate, not a literal code length.
"""
from __future__ import annotations

import argparse
import json
import numpy as np


def covariance_spectrum(samples: np.ndarray) -> np.ndarray:
    """Return K=min(N-1,D) nonnegative eigenvalues, including zeros.

    Use the smaller Gram matrix for the wide AST-Face matrices. Tiny negative
    eigenvalues from roundoff are clipped; significant negatives are errors.
    Missing coordinates are rejected rather than implicitly imputed.
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim < 2 or x.shape[0] < 2 or not x.size or not np.isfinite(x).all():
        raise ValueError('need at least two finite nonempty samples')
    x = x.reshape(len(x), -1)
    x = x - x.mean(axis=0)
    n, d = x.shape
    gram = (x @ x.T if n <= d else x.T @ x) / (n - 1)
    eigenvalues = np.linalg.eigvalsh(gram)[::-1][:min(n - 1, d)]
    slack = 10 * np.finfo(float).eps * float(np.abs(eigenvalues).max()) * len(gram)
    if eigenvalues.min() < -slack:
        raise ArithmeticError('covariance has a significant negative eigenvalue')
    return np.where(eigenvalues <= slack, 0., eigenvalues)


def ecomp(eigenvalues, *, n_components: int, reference_variance: float,
          delta: float = 1e-9, base: float = np.e) -> float:
    """Sum log(lambda_i / reference_variance + delta) over exactly K terms.

    Short spectra are padded with zeros. Truncation is rejected. Callers must
    pass the full spectrum, not a selected subset of principal components.
    delta is dimensionless and fixed across the comparison.
    """
    values = np.asarray(eigenvalues, dtype=np.float64)
    if not isinstance(n_components, (int, np.integer)) or n_components < 1:
        raise ValueError('n_components must be a positive integer')
    if values.ndim != 1 or len(values) > n_components:
        raise ValueError('spectrum must be 1-D and cannot exceed the fixed K')
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('eigenvalues must be finite and nonnegative')
    if not np.isfinite(reference_variance) or reference_variance <= 0:
        raise ValueError('reference_variance must be finite and positive')
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError('delta must be finite and positive')
    if not np.isfinite(base) or base <= 1:
        raise ValueError('log base must be greater than one')
    scaled = np.pad(values, (0, n_components - len(values))) / reference_variance
    return float(np.log(scaled + delta).sum() / np.log(base))


def ecomp_from_pca(model, *, reference_variance: float, delta: float = 1e-9,
                   base: float = np.e) -> float:
    """Score the full spectrum even when reconstruction PCA was truncated."""
    k = min(model.n_samples_ - 1, model.mean_.size)
    return ecomp(model.full_spectrum_, n_components=k,
                 reference_variance=reference_variance, delta=delta, base=base)


def delta_sensitivity(eigenvalues, deltas, *, n_components, reference_variance,
                      base=np.e):
    ds = np.asarray(deltas, dtype=float)
    if ds.ndim != 1 or not len(ds):
        raise ValueError('deltas must be a nonempty vector')
    return np.array([ecomp(eigenvalues, n_components=n_components,
                          reference_variance=reference_variance,
                          delta=float(d), base=base) for d in ds])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eigenvalues', required=True)
    parser.add_argument('--n-components', required=True, type=int)
    parser.add_argument('--reference-variance', required=True, type=float)
    parser.add_argument('--delta', type=float, default=1e-9)
    args = parser.parse_args()
    value = ecomp([float(v) for v in args.eigenvalues.split(',')],
                  n_components=args.n_components, reference_variance=args.reference_variance,
                  delta=args.delta)
    print(json.dumps(dict(score=value, K=args.n_components,
                          reference_variance=args.reference_variance, delta=args.delta,
                          log_base='e', protocol='v2'), indent=2))


if __name__ == '__main__':
    main()
