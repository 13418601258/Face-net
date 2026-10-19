"""Linear PCA decomposition for correspondence-quality experiments.

The input keeps samples on axis 0; all remaining axes are flattened into one
feature vector. The implementation uses an SVD of the centered sample matrix,
which is numerically safer than forming a very large covariance matrix.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def flatten_samples(samples: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return a finite (n_samples, n_features) matrix and sample shape."""

    array = np.asarray(samples, dtype=np.float64)
    if array.ndim < 2:
        raise ValueError("samples must have shape (n_samples, ...)")
    if not np.isfinite(array).all():
        raise ValueError("samples contain NaN or infinite values")
    sample_shape = tuple(array.shape[1:])
    matrix = array.reshape(array.shape[0], -1)
    if matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("at least two samples are required")
    return matrix, sample_shape


@dataclass(frozen=True)
class PCADecomposition:
    """Fitted centered PCA model and its spectrum."""

    mean_: np.ndarray
    components_: np.ndarray
    eigenvalues_: np.ndarray
    sample_shape_: tuple[int, ...]
    n_samples_: int
    total_variance_: float
    singular_value_tolerance_: float
    full_spectrum_: np.ndarray

    @property
    def n_components_(self) -> int:
        return int(self.components_.shape[0])

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        if self.total_variance_ <= 0:
            return np.zeros_like(self.eigenvalues_)
        return self.eigenvalues_ / self.total_variance_

    def normalized_eigenvalues(self) -> np.ndarray:
        """Diagnostic relative spectrum only; NOT the v2 E_COMP input."""

        total = float(self.eigenvalues_.sum())
        if total <= 0 or not np.isfinite(total):
            raise ValueError("the retained PCA spectrum has no positive variance")
        return self.eigenvalues_ / total

    def transform(self, samples: np.ndarray) -> np.ndarray:
        """Project samples onto all fitted principal components."""

        matrix, sample_shape = flatten_samples(samples)
        if sample_shape != self.sample_shape_ or matrix.shape[1] != self.mean_.size:
            raise ValueError(
                f"sample shape mismatch: expected (*, {self.sample_shape_}), got {samples.shape}"
            )
        return (matrix - self.mean_) @ self.components_.T

    def inverse_transform(self, scores: np.ndarray, n_components: int | None = None) -> np.ndarray:
        """Reconstruct samples from the first n_components scores."""

        values = np.asarray(scores, dtype=np.float64)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("scores must have shape (n_samples, n_components)")
        count = self.n_components_ if n_components is None else int(n_components)
        if count < 0 or count > self.n_components_ or values.shape[1] < count:
            raise ValueError(f"n_components must be in [0, {self.n_components_}]")
        reconstructed = values[:, :count] @ self.components_[:count] + self.mean_
        return reconstructed.reshape((values.shape[0],) + self.sample_shape_)

    def reconstruct(self, samples: np.ndarray, n_components: int | None = None) -> np.ndarray:
        """Project and reconstruct samples using the selected component count."""

        return self.inverse_transform(self.transform(samples), n_components=n_components)

    def reconstruction_error(self, samples: np.ndarray, n_components: int | None = None) -> float:
        """Return mean squared reconstruction error per scalar feature."""

        matrix, sample_shape = flatten_samples(samples)
        reconstruction = self.reconstruct(samples, n_components=n_components).reshape(matrix.shape)
        if sample_shape != self.sample_shape_:
            raise ValueError("sample shape mismatch")
        return float(np.mean((matrix - reconstruction) ** 2))


def fit_pca(
    samples: np.ndarray,
    n_components: int | None = None,
    *,
    tolerance: float = 1e-10,
) -> PCADecomposition:
    """Fit PCA with SVD and discard numerically zero singular directions.

    tolerance is relative to the largest singular value. The returned
    spectrum therefore contains positive eigenvalues in descending order,
    which is the form consumed by ecomp.ecomp.
    """

    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    matrix, sample_shape = flatten_samples(samples)
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = (singular_values ** 2) / (matrix.shape[0] - 1)
    # Scoring always uses min(N-1, D) directions, including zero variance.
    dimension = min(matrix.shape[0] - 1, matrix.shape[1])
    full_spectrum = eigenvalues[:dimension].copy()
    roundoff = 10 * np.finfo(float).eps * eigenvalues[0] * min(matrix.shape)
    full_spectrum[full_spectrum <= roundoff] = 0.
    threshold = float(singular_values[0] * tolerance)
    keep = (singular_values > threshold) & (np.arange(len(singular_values)) < dimension)
    eigenvalues = eigenvalues[keep]
    components = components[keep]
    if n_components is not None:
        if not isinstance(n_components, (int, np.integer)):
            raise ValueError('n_components must be an integer')
        count = int(n_components)
        if count < 1:
            raise ValueError("n_components must be positive")
        if count > eigenvalues.size:
            raise ValueError(f"n_components={count} exceeds positive rank {eigenvalues.size}")
        eigenvalues = eigenvalues[:count]
        components = components[:count]
    total_variance = float((centered ** 2).sum() / (matrix.shape[0] - 1))
    return PCADecomposition(
        mean_=mean,
        components_=components,
        eigenvalues_=eigenvalues,
        sample_shape_=sample_shape,
        n_samples_=matrix.shape[0],
        total_variance_=total_variance,
        singular_value_tolerance_=threshold,
        full_spectrum_=full_spectrum,
    )


def _load_input(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            for key in ("samples", "vertices", "arr_0"):
                if key in archive:
                    return archive[key]
        raise ValueError("NPZ must contain samples, vertices, or arr_0")
    raise ValueError("input must be .npy or .npz")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit linear PCA to shape samples")
    parser.add_argument("--input", required=True, help="npy/npz array with samples on axis 0")
    parser.add_argument("--n-components", type=int, default=None)
    parser.add_argument("--output", default=None, help="optional npz model output")
    args = parser.parse_args()
    model = fit_pca(_load_input(Path(args.input)), args.n_components)
    print({
        "n_samples": model.n_samples_,
        "n_features": int(model.mean_.size),
        "n_components": model.n_components_,
        "total_variance": model.total_variance_,
        "eigenvalues": model.eigenvalues_.tolist(),
        "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
    })
    if args.output:
        np.savez_compressed(
            args.output,
            mean=model.mean_,
            components=model.components_,
            eigenvalues=model.eigenvalues_,
            full_spectrum=model.full_spectrum_,
            sample_shape=np.asarray(model.sample_shape_, dtype=np.int64),
            total_variance=np.asarray(model.total_variance_),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
