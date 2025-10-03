"""Utility functions for isosplit6 implementation."""

from typing import Optional, Set

import numpy as np


def compute_centroid(X: np.ndarray) -> np.ndarray:
    """
    Compute centroid (mean) of data points.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (N, M) where N is number of points
        and M is number of features

    Returns
    -------
    np.ndarray
        Centroid of shape (M,)

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> compute_centroid(X)
    array([3., 4.])
    """
    return np.mean(X, axis=0)


def compute_covariance(X: np.ndarray, center: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute covariance matrix of data points.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (N, M)
    center : np.ndarray, optional
        Center point of shape (M,). If None, uses mean of X.

    Returns
    -------
    np.ndarray
        Covariance matrix of shape (M, M)

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> cov = compute_covariance(X)
    >>> cov.shape
    (2, 2)
    """
    if center is None:
        center = compute_centroid(X)

    # Center the data
    X_centered = X - center

    # Compute covariance
    N = X.shape[0]
    cov = (X_centered.T @ X_centered) / N

    return cov


def compute_centroids(
    X: np.ndarray,
    labels: np.ndarray,
    K: int,
    clusters_to_update: Optional[Set[int]] = None
) -> np.ndarray:
    """
    Compute centroids for multiple clusters.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (N, M)
    labels : np.ndarray
        Cluster labels of shape (N,), values from 1 to K
    K : int
        Number of clusters
    clusters_to_update : set of int, optional
        Set of cluster indices (1-indexed) to update.
        If None, computes all centroids.

    Returns
    -------
    np.ndarray
        Centroids array of shape (K, M)

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> labels = np.array([1, 1, 2, 2])
    >>> centroids = compute_centroids(X, labels, 2)
    >>> centroids.shape
    (2, 2)
    """
    M = X.shape[1]
    centroids = np.zeros((K, M))

    if clusters_to_update is None:
        clusters_to_update = set(range(1, K + 1))

    for k in clusters_to_update:
        mask = labels == k
        if np.any(mask):
            centroids[k - 1] = compute_centroid(X[mask])

    return centroids


def compute_covmats(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    K: int,
    clusters_to_update: Optional[Set[int]] = None
) -> np.ndarray:
    """
    Compute covariance matrices for multiple clusters.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (N, M)
    labels : np.ndarray
        Cluster labels of shape (N,), values from 1 to K
    centroids : np.ndarray
        Centroids array of shape (K, M)
    K : int
        Number of clusters
    clusters_to_update : set of int, optional
        Set of cluster indices (1-indexed) to update.
        If None, computes all covariances.

    Returns
    -------
    np.ndarray
        Covariance matrices array of shape (K, M, M)

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> labels = np.array([1, 1, 2, 2])
    >>> centroids = compute_centroids(X, labels, 2)
    >>> covmats = compute_covmats(X, labels, centroids, 2)
    >>> covmats.shape
    (2, 2, 2)
    """
    M = X.shape[1]
    covmats = np.zeros((K, M, M))

    if clusters_to_update is None:
        clusters_to_update = set(range(1, K + 1))

    for k in clusters_to_update:
        mask = labels == k
        if np.any(mask):
            covmats[k - 1] = compute_covariance(X[mask], centroids[k - 1])

    return covmats


def matrix_inverse_stable(A: np.ndarray, regularization: float = 1e-10) -> np.ndarray:
    """
    Compute matrix inverse with regularization for numerical stability.

    Adds a small value to the diagonal before inversion to handle
    near-singular matrices.

    Parameters
    ----------
    A : np.ndarray
        Square matrix of shape (M, M)
    regularization : float, optional
        Amount to add to diagonal (default 1e-10)

    Returns
    -------
    np.ndarray
        Inverse matrix of shape (M, M)

    Examples
    --------
    >>> A = np.array([[1, 0], [0, 1]])
    >>> A_inv = matrix_inverse_stable(A)
    >>> np.allclose(A_inv, np.eye(2))
    True

    Notes
    -----
    For very ill-conditioned matrices, may want to use pseudoinverse instead.
    """
    M = A.shape[0]
    A_reg = A + regularization * np.eye(M)
    return np.linalg.inv(A_reg)


def mahalanobis_distance(
    X: np.ndarray,
    center: np.ndarray,
    cov_inv: np.ndarray
) -> np.ndarray:
    """
    Compute Mahalanobis distance from points to center.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (N, M)
    center : np.ndarray
        Center point of shape (M,)
    cov_inv : np.ndarray
        Inverse covariance matrix of shape (M, M)

    Returns
    -------
    np.ndarray
        Distances of shape (N,)

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> center = np.array([0, 0])
    >>> cov_inv = np.eye(2)
    >>> mahalanobis_distance(X, center, cov_inv)
    array([2.23606798, 5.        ])

    Notes
    -----
    Distance is sqrt((x - center)^T * cov_inv * (x - center))
    """
    # Center the data
    X_centered = X - center

    # Compute (x - center)^T * cov_inv * (x - center) for each point
    # This is equivalent to sum((X_centered @ cov_inv) * X_centered, axis=1)
    distances_sq = np.sum((X_centered @ cov_inv) * X_centered, axis=1)

    return np.sqrt(np.maximum(distances_sq, 0))  # Avoid negative due to numerical errors
