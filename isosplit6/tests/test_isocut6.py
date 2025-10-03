"""
Regression tests for isocut6 implementation.

These tests validate the NumPy implementation against C++ reference outputs.
"""

import numpy as np

from isosplit6._isosplit_core import (
    compute_ks4,
    compute_ks5,
    find_max_index,
    find_min_index,
    isocut6,
)

from .utils import load_reference


class TestHelperFunctions:
    """Tests for isocut6 helper functions."""

    def test_find_min_index(self):
        """Test finding minimum index."""
        arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        assert find_min_index(arr) == 1

    def test_find_max_index(self):
        """Test finding maximum index."""
        arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        assert find_max_index(arr) == 4

    def test_compute_ks4_identical(self):
        """Test KS statistic on identical distributions."""
        counts1 = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        counts2 = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        ks = compute_ks4(counts1, counts2)
        assert np.isclose(ks, 0.0, atol=1e-10)

    def test_compute_ks4_different(self):
        """Test KS statistic on different distributions."""
        counts1 = np.array([1.0, 2.0, 3.0])
        counts2 = np.array([3.0, 2.0, 1.0])
        ks = compute_ks4(counts1, counts2)
        assert ks > 0

    def test_compute_ks4_zero_counts(self):
        """Test KS statistic with zero total counts."""
        counts1 = np.array([0.0, 0.0, 0.0])
        counts2 = np.array([1.0, 2.0, 3.0])
        ks = compute_ks4(counts1, counts2)
        assert ks == 0.0

    def test_compute_ks5_returns_valid_range(self):
        """Test that compute_ks5 returns valid critical range."""
        counts1 = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        counts2 = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        peak_idx = 2

        dipscore, min_idx, max_idx = compute_ks5(counts1, counts2, peak_idx)

        assert dipscore >= 0
        assert 0 <= min_idx <= max_idx < len(counts1)


class TestIsocut6Regression:
    """Regression tests against C++ reference outputs."""

    def test_unimodal_gaussian(self):
        """Test on unimodal Gaussian data."""
        ref = load_reference("isocut6_unimodal_gaussian")
        samples = ref["data"]
        expected_dipscore = float(ref["dipscore"])
        expected_cutpoint = float(ref["cutpoint"])

        dipscore, cutpoint = isocut6(samples)

        # Should match C++ output closely
        assert np.isclose(dipscore, expected_dipscore, rtol=1e-6, atol=1e-6), (
            f"Dipscore mismatch: expected {expected_dipscore}, got {dipscore}"
        )
        assert np.isclose(cutpoint, expected_cutpoint, rtol=1e-6, atol=1e-6), (
            f"Cutpoint mismatch: expected {expected_cutpoint}, got {cutpoint}"
        )

    def test_bimodal_separated(self):
        """Test on well-separated bimodal data."""
        ref = load_reference("isocut6_bimodal_separated")
        samples = ref["data"]
        expected_dipscore = float(ref["dipscore"])
        expected_cutpoint = float(ref["cutpoint"])

        dipscore, cutpoint = isocut6(samples)

        assert np.isclose(dipscore, expected_dipscore, rtol=1e-6, atol=1e-6)
        assert np.isclose(cutpoint, expected_cutpoint, rtol=1e-6, atol=1e-6)

    def test_bimodal_touching(self):
        """Test on touching bimodal data."""
        ref = load_reference("isocut6_bimodal_touching")
        samples = ref["data"]
        expected_dipscore = float(ref["dipscore"])
        expected_cutpoint = float(ref["cutpoint"])

        dipscore, cutpoint = isocut6(samples)

        assert np.isclose(dipscore, expected_dipscore, rtol=1e-6, atol=1e-6)
        assert np.isclose(cutpoint, expected_cutpoint, rtol=1e-6, atol=1e-6)

    def test_uniform_distribution(self):
        """Test on uniform distribution."""
        ref = load_reference("isocut6_uniform")
        samples = ref["data"]
        expected_dipscore = float(ref["dipscore"])
        expected_cutpoint = float(ref["cutpoint"])

        dipscore, cutpoint = isocut6(samples)

        assert np.isclose(dipscore, expected_dipscore, rtol=1e-6, atol=1e-6)
        assert np.isclose(cutpoint, expected_cutpoint, rtol=1e-6, atol=1e-6)

    def test_trimodal(self):
        """Test on trimodal data."""
        ref = load_reference("isocut6_trimodal")
        samples = ref["data"]
        expected_dipscore = float(ref["dipscore"])
        expected_cutpoint = float(ref["cutpoint"])

        dipscore, cutpoint = isocut6(samples)

        assert np.isclose(dipscore, expected_dipscore, rtol=1e-6, atol=1e-6)
        assert np.isclose(cutpoint, expected_cutpoint, rtol=1e-6, atol=1e-6)

    def test_sorted_vs_unsorted(self):
        """Test that sorted and unsorted inputs give same result."""
        ref_unsorted = load_reference("isocut6_unsorted")
        ref_sorted = load_reference("isocut6_sorted")

        samples_unsorted = ref_unsorted["data"]
        samples_sorted = ref_sorted["data"]

        dipscore_unsorted, cutpoint_unsorted = isocut6(samples_unsorted)
        dipscore_sorted, cutpoint_sorted = isocut6(samples_sorted, already_sorted=True)

        # Should give identical results
        assert np.isclose(dipscore_unsorted, dipscore_sorted, rtol=1e-10, atol=1e-10)
        assert np.isclose(
            cutpoint_unsorted, cutpoint_sorted, rtol=1e-10, atol=1e-10
        )

        # Should also match C++ reference
        expected_dipscore = float(ref_sorted["dipscore"])
        expected_cutpoint = float(ref_sorted["cutpoint"])

        assert np.isclose(dipscore_sorted, expected_dipscore, rtol=1e-6, atol=1e-6)
        assert np.isclose(cutpoint_sorted, expected_cutpoint, rtol=1e-6, atol=1e-6)


class TestIsocut6Properties:
    """Property-based tests for isocut6."""

    def test_unimodal_has_low_dipscore(self):
        """Test that unimodal data has low dip score."""
        np.random.seed(42)
        samples = np.random.randn(200)
        dipscore, _cutpoint = isocut6(samples)

        # Unimodal data should have low dip score
        assert dipscore < 3.0

    def test_bimodal_has_high_dipscore(self):
        """Test that bimodal data has high dip score."""
        np.random.seed(43)
        samples1 = np.random.randn(100) - 5.0
        samples2 = np.random.randn(100) + 5.0
        samples = np.concatenate([samples1, samples2])
        np.random.shuffle(samples)

        dipscore, cutpoint = isocut6(samples)

        # Well-separated bimodal should have high dip score
        assert dipscore > 5.0
        # Cutpoint should be roughly between the modes
        assert -2.0 < cutpoint < 2.0

    def test_dipscore_non_negative(self):
        """Test that dipscore is always non-negative."""
        np.random.seed(44)
        samples = np.random.uniform(-10, 10, 100)
        dipscore, _cutpoint = isocut6(samples)

        assert dipscore >= 0

    def test_cutpoint_in_range(self):
        """Test that cutpoint is within data range."""
        np.random.seed(45)
        samples = np.random.randn(100)
        _dipscore, cutpoint = isocut6(samples)

        assert np.min(samples) <= cutpoint <= np.max(samples)

    def test_small_sample(self):
        """Test with small sample size."""
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dipscore, cutpoint = isocut6(samples)

        assert dipscore >= 0
        assert 1.0 <= cutpoint <= 5.0
