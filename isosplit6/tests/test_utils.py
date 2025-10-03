"""
Tests for helper utility functions.
"""

import numpy as np

from isosplit6._isosplit_core import (
    compute_centroid,
    compute_centroids,
    compute_covariance,
    compute_covmats,
    mahalanobis_distance,
    matrix_inverse_stable,
)


def test_compute_centroid():
    """Test centroid computation."""
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    centroid = compute_centroid(X)

    expected = np.array([3.0, 4.0])
    assert np.allclose(centroid, expected)


def test_compute_centroid_single_point():
    """Test centroid of a single point."""
    X = np.array([[2.5, 3.5]], dtype=np.float64)
    centroid = compute_centroid(X)

    assert np.allclose(centroid, X[0])


def test_compute_covariance():
    """Test covariance matrix computation."""
    # Simple case: identity covariance
    np.random.seed(42)
    X = np.random.randn(100, 2)

    cov = compute_covariance(X)

    # Should be close to identity for standard normal data
    assert cov.shape == (2, 2)
    assert cov[0, 1] == cov[1, 0]  # Symmetric


def test_compute_covariance_with_center():
    """Test covariance with explicit center."""
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    center = np.array([3.0, 4.0])

    cov = compute_covariance(X, center)

    # Manually compute expected covariance
    X_centered = X - center
    expected = (X_centered.T @ X_centered) / 3

    assert np.allclose(cov, expected)


def test_compute_centroids():
    """Test computing centroids for multiple clusters."""
    X = np.array([[1, 2], [2, 3], [10, 11], [11, 12]], dtype=np.float64)
    labels = np.array([1, 1, 2, 2])
    K = 2

    centroids = compute_centroids(X, labels, K)

    assert centroids.shape == (2, 2)
    assert np.allclose(centroids[0], [1.5, 2.5])
    assert np.allclose(centroids[1], [10.5, 11.5])


def test_compute_centroids_subset():
    """Test computing centroids for subset of clusters."""
    X = np.array([[1, 2], [2, 3], [10, 11], [11, 12]], dtype=np.float64)
    labels = np.array([1, 1, 2, 2])
    K = 2

    # Only update cluster 1
    centroids = compute_centroids(X, labels, K, clusters_to_update={1})

    assert centroids.shape == (2, 2)
    assert np.allclose(centroids[0], [1.5, 2.5])
    assert np.allclose(centroids[1], [0.0, 0.0])  # Not updated


def test_compute_covmats():
    """Test computing covariance matrices for multiple clusters."""
    X = np.array([[1, 2], [2, 3], [10, 11], [11, 12]], dtype=np.float64)
    labels = np.array([1, 1, 2, 2])
    K = 2

    centroids = compute_centroids(X, labels, K)
    covmats = compute_covmats(X, labels, centroids, K)

    assert covmats.shape == (2, 2, 2)

    # Check symmetry
    assert np.allclose(covmats[0], covmats[0].T)
    assert np.allclose(covmats[1], covmats[1].T)


def test_matrix_inverse_stable():
    """Test stable matrix inversion."""
    # Well-conditioned matrix
    A = np.array([[2, 0], [0, 3]], dtype=np.float64)
    A_inv = matrix_inverse_stable(A)

    expected = np.array([[0.5, 0], [0, 1/3]])
    assert np.allclose(A_inv, expected, atol=1e-9)

    # Verify A * A_inv = I
    assert np.allclose(A @ A_inv, np.eye(2), atol=1e-9)


def test_matrix_inverse_stable_near_singular():
    """Test stable inversion of near-singular matrix."""
    # Nearly singular matrix
    A = np.array([[1, 1], [1, 1.0001]], dtype=np.float64)

    # Should not raise error
    A_inv = matrix_inverse_stable(A, regularization=1e-6)

    assert A_inv.shape == (2, 2)
    # Won't be exact inverse due to regularization, but should be close
    product = A @ A_inv
    # Diagonal should be close to 1
    assert np.abs(product[0, 0] - 1.0) < 0.1
    assert np.abs(product[1, 1] - 1.0) < 0.1


def test_mahalanobis_distance():
    """Test Mahalanobis distance computation."""
    # Identity covariance -> Mahalanobis = Euclidean
    X = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float64)
    center = np.array([0, 0], dtype=np.float64)
    cov_inv = np.eye(2)

    distances = mahalanobis_distance(X, center, cov_inv)

    expected = np.array([1.0, 1.0, np.sqrt(2)])
    assert np.allclose(distances, expected)


def test_mahalanobis_distance_scaled():
    """Test Mahalanobis distance with scaled covariance."""
    X = np.array([[2, 0], [0, 2]], dtype=np.float64)
    center = np.array([0, 0], dtype=np.float64)

    # Diagonal covariance: more variance in x-direction
    cov = np.array([[4, 0], [0, 1]], dtype=np.float64)
    cov_inv = np.linalg.inv(cov)

    distances = mahalanobis_distance(X, center, cov_inv)

    # (2,0) with cov [[4,0],[0,1]]: distance = sqrt((2/2)^2 + 0) = 1
    # (0,2) with cov [[4,0],[0,1]]: distance = sqrt(0 + (2/1)^2) = 2
    expected = np.array([1.0, 2.0])
    assert np.allclose(distances, expected)


def test_mahalanobis_distance_nonzero_center():
    """Test Mahalanobis distance with non-zero center."""
    X = np.array([[1, 1], [2, 2]], dtype=np.float64)
    center = np.array([1, 1], dtype=np.float64)
    cov_inv = np.eye(2)

    distances = mahalanobis_distance(X, center, cov_inv)

    # First point is at center -> distance 0
    # Second point is [1,1] away -> distance sqrt(2)
    expected = np.array([0.0, np.sqrt(2)])
    assert np.allclose(distances, expected)
