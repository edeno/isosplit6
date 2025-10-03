"""
Core utilities for isosplit6 implementation.

This package contains the low-level functions used by the NumPy and JAX
implementations of isosplit6.
"""

from .isocut import (
    compute_ks4,
    compute_ks5,
    find_max_index,
    find_min_index,
    isocut6,
)
from .isotonic import (
    jisotonic5,
    jisotonic5_downup,
    jisotonic5_sort,
    jisotonic5_updown,
)
from .utils import (
    compute_centroid,
    compute_centroids,
    compute_covariance,
    compute_covmats,
    mahalanobis_distance,
    matrix_inverse_stable,
)

__all__ = [
    "compute_centroid",
    "compute_centroids",
    "compute_covariance",
    "compute_covmats",
    "compute_ks4",
    "compute_ks5",
    "find_max_index",
    "find_min_index",
    "isocut6",
    "jisotonic5",
    "jisotonic5_downup",
    "jisotonic5_sort",
    "jisotonic5_updown",
    "mahalanobis_distance",
    "matrix_inverse_stable",
]
