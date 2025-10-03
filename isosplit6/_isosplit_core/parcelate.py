"""
Parcelate2 algorithm for initialization.

This module implements the parcelate2 algorithm used by the C++ implementation
for creating initial cluster parcels. Unlike KMeans, parcelate2 does NOT do
final reassignment, which preserves more natural cluster boundaries.
"""

from typing import List

import numpy as np

__all__ = ["parcelate2"]


class Parcel:
    """
    A parcel (cluster) in the parcelate2 algorithm.

    Attributes
    ----------
    indices : np.ndarray
        Indices of points in this parcel
    centroid : np.ndarray
        Centroid of the parcel
    radius : float
        Maximum distance from centroid to any point
    """

    def __init__(self, indices: np.ndarray, X: np.ndarray):
        """
        Initialize a parcel.

        Parameters
        ----------
        indices : np.ndarray
            Indices of points in this parcel
        X : np.ndarray
            Full data matrix
        """
        self.indices = indices
        if len(indices) > 0:
            self.centroid = np.mean(X[indices], axis=0)
            dists = np.linalg.norm(X[indices] - self.centroid, axis=1)
            self.radius = np.max(dists) if len(dists) > 0 else 0.0
        else:
            self.centroid = np.zeros(X.shape[1])
            self.radius = 0.0


def parcelate2(
    X: np.ndarray,
    target_parcel_size: int = 10,
    target_num_parcels: int = 200,
    final_reassign: bool = False,
) -> np.ndarray:
    """
    Parcelate2 algorithm for creating initial cluster parcels.

    Creates parcels by iteratively splitting large parcels based on distance
    to randomly selected split points. Does NOT do final reassignment by default,
    which preserves natural cluster boundaries (unlike KMeans).

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (N, M)
    target_parcel_size : int, optional
        Minimum size for parcels. Parcels larger than this may be split.
        Default is 10.
    target_num_parcels : int, optional
        Target number of parcels to create. Default is 200.
    final_reassign : bool, optional
        If True, reassign points to nearest centroid at the end (like KMeans).
        If False, keep original assignments (preserves natural boundaries).
        Default is False.

    Returns
    -------
    labels : np.ndarray
        Cluster labels of shape (N,), 1-indexed

    Notes
    -----
    Algorithm:
    1. Start with all points in one parcel
    2. Iteratively split large parcels:
       - Select split_factor=3 points from the parcel
       - Assign each point to nearest split point
       - Create new parcels from these assignments
    3. Repeat until target_num_parcels reached
    4. Optionally reassign points to nearest centroid (final_reassign)

    The key difference from KMeans: NO final reassignment by default.
    This avoids creating "hexagonal" patterns that are not conducive to
    isosplit iterations.

    Examples
    --------
    >>> X = np.random.randn(100, 2)
    >>> labels = parcelate2(X, target_parcel_size=10, target_num_parcels=20)
    >>> len(np.unique(labels))
    20

    References
    ----------
    Based on src/isosplit5.cpp:133-263
    """
    N, _M = X.shape
    split_factor = 3  # Split each parcel into up to 3 sub-parcels

    # Initialize with all points in one parcel
    labels = np.ones(N, dtype=np.int32)
    parcels: List[Parcel] = [Parcel(np.arange(N), X)]

    # Iteratively split parcels
    something_changed = True
    while len(parcels) < target_num_parcels and something_changed:
        something_changed = False

        # Check if any parcels can be split
        candidate_found = False
        for parcel in parcels:
            if len(parcel.indices) > target_parcel_size and parcel.radius > 0:
                candidate_found = True
                break

        if not candidate_found:
            break

        # Find target radius (95% of max radius among large parcels)
        target_radius = 0.0
        for parcel in parcels:
            if len(parcel.indices) > target_parcel_size:
                tmp = parcel.radius * 0.95
                if tmp > target_radius:
                    target_radius = tmp

        if target_radius == 0:
            break

        # Split parcels that are too large and have radius >= target_radius
        p_index = 0
        while p_index < len(parcels):
            parcel = parcels[p_index]
            sz = len(parcel.indices)
            rad = parcel.radius

            if sz > target_parcel_size and rad >= target_radius:
                # Select split points (first split_factor points in the parcel)
                # Note: C++ uses p2_randsample which just returns first K indices
                num_split_points = min(split_factor, sz)
                split_point_indices = parcel.indices[:num_split_points]

                # Assign each point to nearest split point
                assignments = np.zeros(sz, dtype=np.int32)
                for i, idx in enumerate(parcel.indices):
                    # Compute distances to all split points
                    dists = np.linalg.norm(
                        X[split_point_indices] - X[idx], axis=1
                    )
                    assignments[i] = np.argmin(dists)

                # Create new parcels from assignments
                new_parcels = []
                for j in range(num_split_points):
                    mask = assignments == j
                    if np.any(mask):
                        new_indices = parcel.indices[mask]
                        new_parcel = Parcel(new_indices, X)
                        new_parcels.append(new_parcel)

                        # Update labels
                        if j == 0:
                            # First group stays in current parcel
                            labels[new_indices] = p_index + 1
                        else:
                            # Other groups get new parcel IDs
                            labels[new_indices] = len(parcels) + len(new_parcels) - 1

                # Replace current parcel with first new parcel
                if len(new_parcels) > 0:
                    parcels[p_index] = new_parcels[0]

                    # Add remaining new parcels
                    if len(new_parcels) > 1:
                        parcels.extend(new_parcels[1:])

                    # Check if split actually happened
                    if len(new_parcels[0].indices) < sz:
                        something_changed = True
                    else:
                        p_index += 1
                else:
                    p_index += 1
            else:
                p_index += 1

    # Final reassignment (optional, usually False for isosplit)
    if final_reassign and len(parcels) > 0:
        # Compute centroids
        centroids = np.array([p.centroid for p in parcels])

        # Reassign each point to nearest centroid (like KMeans)
        for i in range(N):
            dists = np.linalg.norm(centroids - X[i], axis=1)
            labels[i] = np.argmin(dists) + 1

    return labels
