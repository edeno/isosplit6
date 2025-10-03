"""
Isosplit6 clustering algorithm - NumPy implementation.

This module implements the full isosplit6 clustering algorithm using NumPy,
following the C++ reference implementation.
"""

from typing import Optional, Set, Tuple

import numpy as np
from sklearn.cluster import KMeans

from ._isosplit_core import (
    compute_centroids,
    compute_covmats,
    isocut6,
    matrix_inverse_stable,
)

__all__ = ["isosplit6"]


def get_pairs_to_compare(
    centroids: np.ndarray, comparisons_made: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find mutually closest pairs of clusters to compare.

    For each cluster, finds its nearest neighbor that hasn't been compared yet.
    Only returns pairs that are mutual nearest neighbors (k1 is nearest to k2
    AND k2 is nearest to k1).

    Parameters
    ----------
    centroids : np.ndarray
        Cluster centroids of shape (K, M) where K is number of clusters
        and M is number of features
    comparisons_made : np.ndarray
        Boolean matrix of shape (K, K) tracking which pairs have been compared

    Returns
    -------
    inds1 : np.ndarray
        First cluster indices of pairs to compare (1-indexed)
    inds2 : np.ndarray
        Second cluster indices of pairs to compare (1-indexed)

    Notes
    -----
    Returns only mutually closest pairs to avoid redundant comparisons.
    Indices are 1-indexed to match C++ implementation.

    References
    ----------
    Based on src/isosplit5.cpp:866-923
    """
    K = len(centroids)

    # Compute pairwise distances
    dists = np.full((K, K), -1.0)
    for k1 in range(K):
        for k2 in range(K):
            if comparisons_made[k1, k2] or k1 == k2:
                dists[k1, k2] = -1.0
            else:
                diff = centroids[k1] - centroids[k2]
                dists[k1, k2] = np.sqrt(np.sum(diff**2))

    # Find nearest neighbor for each cluster
    best_inds = np.full(K, -1, dtype=int)
    for k in range(K):
        valid_dists = dists[k, :] >= 0
        if np.any(valid_dists):
            valid_indices = np.where(valid_dists)[0]
            best_ind = valid_indices[np.argmin(dists[k, valid_indices])]
            best_inds[k] = best_ind

    # Find mutual nearest neighbors
    inds1 = []
    inds2 = []
    for j in range(K):
        if best_inds[j] > j:  # Only consider each pair once
            if best_inds[best_inds[j]] == j:  # Mutual nearest neighbors
                if dists[j, best_inds[j]] >= 0:
                    inds1.append(j + 1)  # 1-indexed
                    inds2.append(best_inds[j] + 1)  # 1-indexed

    return np.array(inds1, dtype=np.int64), np.array(inds2, dtype=np.int64)


def merge_test(
    X1: np.ndarray,
    X2: np.ndarray,
    centroid1: np.ndarray,
    centroid2: np.ndarray,
    covmat1: np.ndarray,
    covmat2: np.ndarray,
    isocut_threshold: float,
) -> Tuple[bool, np.ndarray]:
    """
    Test if two clusters should be merged using isocut6 algorithm.

    Projects both clusters onto the whitened direction between centroids,
    then runs 1D dip test to check for bimodality.

    Parameters
    ----------
    X1 : np.ndarray
        Points in cluster 1, shape (N1, M)
    X2 : np.ndarray
        Points in cluster 2, shape (N2, M)
    centroid1 : np.ndarray
        Centroid of cluster 1, shape (M,)
    centroid2 : np.ndarray
        Centroid of cluster 2, shape (M,)
    covmat1 : np.ndarray
        Covariance matrix of cluster 1, shape (M, M)
    covmat2 : np.ndarray
        Covariance matrix of cluster 2, shape (M, M)
    isocut_threshold : float
        Dip score threshold for merging

    Returns
    -------
    do_merge : bool
        True if clusters should be merged, False otherwise
    L12 : np.ndarray
        New labels (1 or 2) for all points, shape (N1 + N2,)

    Notes
    -----
    Algorithm:
    1. Compute direction vector between centroids
    2. Whiten by average covariance matrix inverse
    3. Project both clusters onto direction
    4. Run isocut6 on 1D projection
    5. If dipscore < threshold, merge; otherwise redistribute

    References
    ----------
    Based on src/isosplit6.cpp:233-323
    """
    N1, _M = X1.shape
    N2 = len(X2)

    # Initialize labels to all 1 (will be updated)
    L12 = np.ones(N1 + N2, dtype=np.int32)

    if N1 == 0 or N2 == 0:
        return True, L12

    # Compute direction vector between centroids
    V = centroid2 - centroid1

    # Whiten by average covariance
    avg_covmat = (covmat1 + covmat2) / 2
    inv_avg_covmat = matrix_inverse_stable(avg_covmat)

    # Apply whitening: V = inv_avg_covmat @ V
    V = inv_avg_covmat @ V

    # Normalize
    norm = np.sqrt(np.sum(V**2))
    if norm > 0:
        V = V / norm

    # Project data onto direction vector
    projection1 = X1 @ V
    projection2 = X2 @ V
    projection12 = np.concatenate([projection1, projection2])

    # Run isocut6 on 1D projection
    dipscore, cutpoint = isocut6(projection12, already_sorted=False)

    # Decide whether to merge
    if dipscore < isocut_threshold:
        do_merge = True
    else:
        do_merge = False

    # Assign labels based on cutpoint
    L12[projection12 < cutpoint] = 1
    L12[projection12 >= cutpoint] = 2

    return do_merge, L12


def compare_pairs(
    X: np.ndarray,
    labels: np.ndarray,
    k1s: np.ndarray,
    k2s: np.ndarray,
    centroids: np.ndarray,
    covmats: np.ndarray,
    isocut_threshold: float,
    min_cluster_size: int,
) -> Tuple[Set[int], int]:
    """
    Compare multiple cluster pairs and merge or redistribute points.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (N, M)
    labels : np.ndarray
        Current cluster labels of shape (N,), modified in-place
    k1s : np.ndarray
        First cluster indices to compare
    k2s : np.ndarray
        Second cluster indices to compare
    centroids : np.ndarray
        Cluster centroids
    covmats : np.ndarray
        Cluster covariance matrices
    isocut_threshold : float
        Dip score threshold for merging
    min_cluster_size : int
        Minimum points per cluster

    Returns
    -------
    clusters_changed : Set[int]
        Set of cluster indices that changed
    total_num_label_changes : int
        Total number of points reassigned

    References
    ----------
    Based on src/isosplit6.cpp:325-408
    """
    _N, _M = X.shape
    np.max(labels)

    clusters_changed = set()
    new_labels = labels.copy()
    total_num_label_changes = 0

    for k1, k2 in zip(k1s, k2s):
        # Get indices for each cluster
        inds1 = np.where(labels == k1)[0]
        inds2 = np.where(labels == k2)[0]

        if len(inds1) == 0 or len(inds2) == 0:
            continue

        # Check if either cluster is too small (auto-merge)
        if len(inds1) < min_cluster_size or len(inds2) < min_cluster_size:
            do_merge = True
            L12 = np.ones(len(inds1) + len(inds2), dtype=np.int32)
        else:
            # Extract data for both clusters
            X1 = X[inds1]
            X2 = X[inds2]

            # Run merge test
            do_merge, L12 = merge_test(
                X1,
                X2,
                centroids[k1 - 1],
                centroids[k2 - 1],
                covmats[k1 - 1],
                covmats[k2 - 1],
                isocut_threshold,
            )

        if do_merge:
            # Merge: assign all of k2 to k1
            new_labels[inds2] = k1
            total_num_label_changes += len(inds2)
            clusters_changed.add(k1)
            clusters_changed.add(k2)
        else:
            # Redistribute based on L12 labels
            something_redistributed = False

            # Points in cluster 1 that should move to cluster 2
            for i, idx in enumerate(inds1):
                if L12[i] == 2:
                    new_labels[idx] = k2
                    total_num_label_changes += 1
                    something_redistributed = True

            # Points in cluster 2 that should move to cluster 1
            for i, idx in enumerate(inds2):
                if L12[len(inds1) + i] == 1:
                    new_labels[idx] = k1
                    total_num_label_changes += 1
                    something_redistributed = True

            if something_redistributed:
                clusters_changed.add(k1)
                clusters_changed.add(k2)

    # Update labels in-place
    labels[:] = new_labels

    return clusters_changed, total_num_label_changes


def initialize_labels(
    X: np.ndarray, K_init: int = 200, min_cluster_size: int = 10, random_state: int = 0
) -> np.ndarray:
    """
    Initialize cluster labels using K-means.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (N, M)
    K_init : int, optional
        Initial number of clusters. Default is 200.
    min_cluster_size : int, optional
        Minimum cluster size. Default is 10.
    random_state : int, optional
        Random seed for reproducibility. Default is 0.

    Returns
    -------
    labels : np.ndarray
        Initial cluster labels of shape (N,), 1-indexed

    Notes
    -----
    This is a simplified version compared to the C++ parcelate2 function.
    May produce slightly different results from C++ implementation.
    """
    N = len(X)

    # Adjust K_init if needed
    K_init = min(K_init, N // min_cluster_size)
    K_init = max(1, K_init)

    if K_init == 1:
        return np.ones(N, dtype=np.int32)

    # Run K-means
    kmeans = KMeans(n_clusters=K_init, random_state=random_state, n_init=1)
    labels = kmeans.fit_predict(X)

    # Convert to 1-indexed
    return labels.astype(np.int32) + 1


def remap_labels(labels: np.ndarray) -> np.ndarray:
    """
    Remap cluster labels to consecutive integers 1, 2, 3, ...

    Parameters
    ----------
    labels : np.ndarray
        Current cluster labels

    Returns
    -------
    labels : np.ndarray
        Remapped labels

    Examples
    --------
    >>> labels = np.array([1, 3, 3, 7, 7, 7])
    >>> remap_labels(labels)
    array([1, 2, 2, 3, 3, 3])
    """
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels, start=1)}
    return np.array([label_map[label] for label in labels], dtype=np.int32)


def isosplit6(
    X: np.ndarray,
    *,
    initial_labels: Optional[np.ndarray] = None,
    isocut_threshold: float = 2.0,
    min_cluster_size: int = 10,
    K_init: int = 200,
    max_iterations_per_pass: int = 500,
) -> np.ndarray:
    """
    Isosplit6 clustering algorithm (NumPy implementation).

    Finds clusters by detecting regions of low density between them using
    isotonic regression and Hartigan's dip test. Does not require specifying
    the number of clusters.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (N, M) where N is number of observations
        and M is number of features
    initial_labels : np.ndarray, optional
        Initial cluster labels of shape (N,). If None, will initialize
        using K-means with K_init clusters. Default is None.
    isocut_threshold : float, optional
        Dip score threshold for merging clusters. Lower values result in
        fewer clusters. Default is 2.0.
    min_cluster_size : int, optional
        Minimum points per cluster. Clusters smaller than this are
        automatically merged. Default is 10.
    K_init : int, optional
        Initial number of clusters (only used if initial_labels is None).
        Default is 200.
    max_iterations_per_pass : int, optional
        Maximum iterations per pass to prevent infinite loops.
        Default is 500.

    Returns
    -------
    labels : np.ndarray
        Cluster labels of shape (N,). Labels are 1-indexed integers:
        1, 2, 3, ..., K where K is the number of clusters found.

    Notes
    -----
    Algorithm phases:
    1. Initialization: Create initial parcels using K-means
    2. Iterative merging: Compare cluster pairs, merge or redistribute
    3. Label remapping: Renumber to consecutive integers

    The algorithm continues until no more merges occur, then does one
    final pass for redistribution.

    Examples
    --------
    >>> import numpy as np
    >>> from isosplit6.isosplit6_numpy import isosplit6
    >>> # Two well-separated Gaussians
    >>> X1 = np.random.randn(100, 2)
    >>> X2 = np.random.randn(100, 2) + 10
    >>> X = np.vstack([X1, X2])
    >>> labels = isosplit6(X)
    >>> len(np.unique(labels))
    2

    References
    ----------
    Based on src/isosplit6.cpp:29-230
    """
    _N, _M = X.shape

    # Phase 1: Initialization
    if initial_labels is None:
        labels = initialize_labels(X, K_init, min_cluster_size)
    else:
        labels = initial_labels.copy().astype(np.int32)

    Kmax = np.max(labels)

    # Compute initial centroids and covariances for all clusters
    clusters_to_update = set(range(1, Kmax + 1))
    centroids = compute_centroids(X, labels, Kmax, clusters_to_update)
    covmats = compute_covmats(X, labels, centroids, Kmax, clusters_to_update)

    # Track which comparisons have been made
    comparisons_made = np.zeros((Kmax, Kmax), dtype=bool)

    # Phase 2: Iterative merging
    final_pass = False

    while True:  # Outer loop: passes
        something_merged = False
        clusters_changed_in_pass = set()

        iteration_number = 0

        while True:  # Inner loop: iterations
            iteration_number += 1

            if iteration_number > max_iterations_per_pass:
                print("Warning: max iterations per pass exceeded.")
                break

            # Get active labels (clusters that still exist)
            active_labels = np.unique(labels)

            if len(active_labels) == 0:
                break

            # Create active centroids and comparisons for pair finding
            active_centroids = centroids[active_labels - 1]
            active_comparisons = comparisons_made[
                np.ix_(active_labels - 1, active_labels - 1)
            ]

            # Find pairs to compare
            inds1, inds2 = get_pairs_to_compare(active_centroids, active_comparisons)

            # Remap to original cluster indices
            if len(inds1) > 0:
                inds1 = active_labels[inds1 - 1]
                inds2 = active_labels[inds2 - 1]

            # No more pairs to compare
            if len(inds1) == 0:
                break

            # Compare pairs
            clusters_changed, _ = compare_pairs(
                X,
                labels,
                inds1,
                inds2,
                centroids,
                covmats,
                isocut_threshold,
                min_cluster_size,
            )

            # Track changes
            clusters_changed_in_pass.update(clusters_changed)

            # Update comparisons_made
            for k1, k2 in zip(inds1, inds2):
                comparisons_made[k1 - 1, k2 - 1] = True
                comparisons_made[k2 - 1, k1 - 1] = True

            # Recompute centroids and covmats for changed clusters
            if clusters_changed:
                centroids = compute_centroids(X, labels, Kmax, clusters_changed)
                covmats = compute_covmats(X, labels, centroids, Kmax, clusters_changed)

            # Check if something merged
            new_active_labels = np.unique(labels)
            if len(new_active_labels) < len(active_labels):
                something_merged = True

        # Reset comparisons for clusters that changed in this pass
        for k in clusters_changed_in_pass:
            comparisons_made[k - 1, :] = False
            comparisons_made[:, k - 1] = False

        # Check convergence
        if something_merged:
            final_pass = False
        if final_pass:
            break
        if not something_merged:
            final_pass = True

    # Phase 3: Remap labels to consecutive integers
    labels = remap_labels(labels)

    return labels
