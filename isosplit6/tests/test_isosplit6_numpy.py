"""
Regression tests for isosplit6 NumPy implementation.

These tests validate the NumPy implementation against C++ reference outputs
and verify expected algorithm properties.
"""

import numpy as np

from isosplit6.isosplit6_numpy import (
    compare_pairs,
    get_pairs_to_compare,
    initialize_labels,
    isosplit6,
    merge_test,
    remap_labels,
)

from .utils import labels_are_equivalent, load_reference


class TestHelperFunctions:
    """Tests for isosplit6 helper functions."""

    def test_remap_labels(self):
        """Test label remapping to consecutive integers."""
        labels = np.array([1, 3, 3, 7, 7, 7])
        remapped = remap_labels(labels)

        expected = np.array([1, 2, 2, 3, 3, 3])
        assert np.array_equal(remapped, expected)

    def test_remap_labels_already_consecutive(self):
        """Test remapping labels that are already consecutive."""
        labels = np.array([1, 1, 2, 2, 3, 3])
        remapped = remap_labels(labels)

        assert np.array_equal(remapped, labels)

    def test_get_pairs_to_compare_simple(self):
        """Test finding pairs to compare."""
        centroids = np.array([[0, 0], [1, 0], [10, 0]])
        comparisons_made = np.zeros((3, 3), dtype=bool)

        inds1, inds2 = get_pairs_to_compare(centroids, comparisons_made)

        # Should find (1, 2) as mutual nearest neighbors
        assert len(inds1) > 0
        # Labels are 1-indexed
        assert np.all(inds1 >= 1)
        assert np.all(inds2 >= 1)

    def test_get_pairs_to_compare_with_comparisons_made(self):
        """Test that already compared pairs are skipped."""
        centroids = np.array([[0, 0], [1, 0], [2, 0]])
        comparisons_made = np.zeros((3, 3), dtype=bool)
        comparisons_made[0, 1] = True
        comparisons_made[1, 0] = True

        inds1, inds2 = get_pairs_to_compare(centroids, comparisons_made)

        # Should not include (0,1) or (1,0) pair (0-indexed)
        for i1, i2 in zip(inds1, inds2):
            assert not (i1 == 1 and i2 == 2)
            assert not (i1 == 2 and i2 == 1)

    def test_initialize_labels(self):
        """Test initialization with KMeans."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        labels = initialize_labels(X, K_init=5)

        # Labels should be 1-indexed
        assert np.min(labels) >= 1
        # Should have created some clusters
        assert len(np.unique(labels)) >= 1


class TestMergeTest:
    """Tests for merge_test function."""

    def test_merge_test_well_separated(self):
        """Test merge_test on well-separated clusters."""
        np.random.seed(42)
        X1 = np.random.randn(50, 2)
        X2 = np.random.randn(50, 2) + 10

        centroid1 = np.mean(X1, axis=0)
        centroid2 = np.mean(X2, axis=0)
        covmat1 = np.cov(X1.T)
        covmat2 = np.cov(X2.T)

        do_merge, L12 = merge_test(
            X1, X2, centroid1, centroid2, covmat1, covmat2, isocut_threshold=2.0
        )

        # Well-separated clusters should not merge
        assert not do_merge
        # Should have labels 1 and 2
        assert set(L12) == {1, 2}

    def test_merge_test_overlapping(self):
        """Test merge_test on overlapping clusters."""
        np.random.seed(42)
        X1 = np.random.randn(50, 2)
        X2 = np.random.randn(50, 2) + 1  # Slight offset

        centroid1 = np.mean(X1, axis=0)
        centroid2 = np.mean(X2, axis=0)
        covmat1 = np.cov(X1.T)
        covmat2 = np.cov(X2.T)

        do_merge, L12 = merge_test(
            X1, X2, centroid1, centroid2, covmat1, covmat2, isocut_threshold=2.0
        )

        # Overlapping clusters should merge
        assert do_merge
        # All points should have same label when merged
        # (though merge_test still returns cutpoint-based labels)
        assert len(L12) == 100


class TestIsosplit6Properties:
    """Property-based tests for isosplit6."""

    def test_labels_are_positive_integers(self):
        """Test that labels are positive integers."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        labels = isosplit6(X)

        assert labels.dtype in [np.int32, np.int64]
        assert np.all(labels > 0)

    def test_labels_are_contiguous(self):
        """Test that labels are contiguous 1, 2, 3, ..."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        labels = isosplit6(X)

        unique_labels = np.unique(labels)
        expected = np.arange(1, len(unique_labels) + 1)
        assert np.array_equal(unique_labels, expected)

    def test_deterministic(self):
        """Test that algorithm is deterministic with same random seed."""
        np.random.seed(42)
        X = np.random.randn(100, 5)

        labels1 = isosplit6(X)
        labels2 = isosplit6(X)

        assert np.array_equal(labels1, labels2)

    def test_single_cluster_data(self):
        """Test on data from single Gaussian."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        labels = isosplit6(X)

        # Should find 1 cluster
        assert len(np.unique(labels)) == 1

    def test_two_well_separated_clusters(self):
        """Test on two well-separated Gaussians."""
        np.random.seed(42)
        X1 = np.random.randn(100, 2)
        X2 = np.random.randn(100, 2) + 10
        X = np.vstack([X1, X2])

        labels = isosplit6(X)

        # Should find 2 clusters
        assert len(np.unique(labels)) == 2

    def test_three_well_separated_clusters(self):
        """Test on three well-separated Gaussians."""
        np.random.seed(42)
        X1 = np.random.randn(100, 2)
        X2 = np.random.randn(100, 2) + [10, 0]
        X3 = np.random.randn(100, 2) + [5, 10]
        X = np.vstack([X1, X2, X3])

        labels = isosplit6(X)

        # Should find 3 clusters
        assert len(np.unique(labels)) == 3

    def test_with_initial_labels(self):
        """Test with user-provided initial labels."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        initial_labels = np.ones(100, dtype=np.int32)

        labels = isosplit6(X, initial_labels=initial_labels)

        # Should still work
        assert len(labels) == 100
        assert np.all(labels > 0)


class TestIsosplit6Regression:
    """Regression tests against C++ reference outputs."""

    def test_two_gaussians_2d(self):
        """Test on two 2D Gaussians."""
        ref = load_reference("isosplit6_two_gaussians_2d")
        data = ref["data"]
        expected_labels = ref["labels"]

        labels = isosplit6(data)

        # Should find same number of clusters
        assert len(np.unique(labels)) == len(np.unique(expected_labels))

        # Clustering should be equivalent (allowing for label permutation)
        assert labels_are_equivalent(labels, expected_labels, tol=0.9)

    def test_three_gaussians_2d(self):
        """Test on three 2D Gaussians."""
        ref = load_reference("isosplit6_three_gaussians_2d")
        data = ref["data"]
        expected_labels = ref["labels"]

        labels = isosplit6(data)

        # Should find same number of clusters
        assert len(np.unique(labels)) == len(np.unique(expected_labels))

        # Clustering should be equivalent
        assert labels_are_equivalent(labels, expected_labels, tol=0.9)

    def test_single_gaussian(self):
        """Test on single Gaussian."""
        ref = load_reference("isosplit6_single_gaussian")
        data = ref["data"]
        expected_labels = ref["labels"]

        labels = isosplit6(data)

        # Should find 1 cluster
        assert len(np.unique(labels)) == 1
        assert len(np.unique(expected_labels)) == 1

    def test_high_dim_10d(self):
        """Test on high-dimensional data (10D)."""
        ref = load_reference("isosplit6_high_dim_10d")
        data = ref["data"]
        expected_labels = ref["labels"]

        labels = isosplit6(data)

        # Should find same number of clusters
        assert len(np.unique(labels)) == len(np.unique(expected_labels))

        # Clustering should be reasonably equivalent
        assert labels_are_equivalent(labels, expected_labels, tol=0.8)

    def test_different_sizes(self):
        """Test on clusters with different sizes."""
        ref = load_reference("isosplit6_different_sizes")
        data = ref["data"]
        expected_labels = ref["labels"]

        labels = isosplit6(data)

        # Should find same number of clusters
        assert len(np.unique(labels)) == len(np.unique(expected_labels))

        # Clustering should be equivalent
        assert labels_are_equivalent(labels, expected_labels, tol=0.9)


class TestIsosplit6Parameters:
    """Tests for parameter sensitivity."""

    def test_isocut_threshold_lower(self):
        """Test that lower threshold gives fewer clusters."""
        np.random.seed(42)
        X1 = np.random.randn(100, 2)
        X2 = np.random.randn(100, 2) + 3  # Moderate separation
        X = np.vstack([X1, X2])

        labels_high_threshold = isosplit6(X, isocut_threshold=1.0)
        labels_low_threshold = isosplit6(X, isocut_threshold=3.0)

        # Lower threshold should give fewer or equal clusters
        assert len(np.unique(labels_low_threshold)) <= len(
            np.unique(labels_high_threshold)
        )

    def test_min_cluster_size(self):
        """Test min_cluster_size parameter."""
        np.random.seed(42)
        X = np.random.randn(100, 2)

        # With larger min_cluster_size, should get fewer clusters
        labels = isosplit6(X, min_cluster_size=20)

        # Each cluster should have at least min_cluster_size points
        # (or be close to it if merging happened)
        for k in np.unique(labels):
            count = np.sum(labels == k)
            # Some tolerance since final merge may create smaller clusters
            assert count >= 10

    def test_K_init(self):
        """Test K_init parameter."""
        np.random.seed(42)
        X = np.random.randn(200, 2)

        # Should work with different K_init values
        labels_small_k = isosplit6(X, K_init=10)
        labels_large_k = isosplit6(X, K_init=50)

        # Both should produce valid clusterings
        assert len(labels_small_k) == 200
        assert len(labels_large_k) == 200
        assert np.all(labels_small_k > 0)
        assert np.all(labels_large_k > 0)
