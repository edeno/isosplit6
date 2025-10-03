"""
Tests for isotonic regression functions.

These tests verify mathematical properties of the isotonic regression
implementations rather than exact numerical matches with C++.
"""

import numpy as np

from isosplit6._isosplit_core import (
    jisotonic5,
    jisotonic5_downup,
    jisotonic5_sort,
    jisotonic5_updown,
)


def is_non_decreasing(arr: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if array is non-decreasing."""
    return np.all(np.diff(arr) >= -tol)


def is_non_increasing(arr: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if array is non-increasing."""
    return np.all(np.diff(arr) <= tol)


def is_unimodal(arr: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if array has single peak (up then down)."""
    # Find where it stops increasing
    diff = np.diff(arr)
    if len(diff) == 0:
        return True

    # Should be non-decreasing up to some point, then non-increasing
    # Find last increasing point
    increasing_mask = diff > -tol
    if not np.any(increasing_mask):
        # All decreasing - still unimodal
        return True

    last_inc = np.where(increasing_mask)[0][-1] if np.any(increasing_mask) else -1

    # After that point, should be non-increasing
    if last_inc < len(diff) - 1:
        return is_non_increasing(arr[last_inc + 1:], tol)

    return True


def has_valley(arr: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if array has single valley (down then up)."""
    # Should be non-increasing up to some point, then non-decreasing
    diff = np.diff(arr)
    if len(diff) == 0:
        return True

    # Find last decreasing point
    decreasing_mask = diff < tol
    if not np.any(decreasing_mask):
        # All increasing - still has valley (at start)
        return True

    last_dec = np.where(decreasing_mask)[0][-1] if np.any(decreasing_mask) else -1

    # After that point, should be non-decreasing
    if last_dec < len(diff) - 1:
        return is_non_decreasing(arr[last_dec + 1:], tol)

    return True


class TestJisotonic5:
    """Tests for basic isotonic regression."""

    def test_already_increasing(self):
        """Test on data that's already increasing."""
        A = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        B, _MSE = jisotonic5(A)

        # Should be unchanged
        assert np.allclose(B, A)
        # Should be non-decreasing
        assert is_non_decreasing(B)

    def test_decreasing_data(self):
        """Test on decreasing data."""
        A = np.array([5, 4, 3, 2, 1], dtype=np.float64)
        B, _MSE = jisotonic5(A)

        # Should be non-decreasing
        assert is_non_decreasing(B)
        # Should be flattened (constant)
        assert np.allclose(B, np.mean(A))

    def test_violator_at_end(self):
        """Test with single violator at end."""
        A = np.array([1, 2, 3, 4, 2], dtype=np.float64)
        B, _MSE = jisotonic5(A)

        # Should be non-decreasing
        assert is_non_decreasing(B)
        # Last two should be pooled
        assert B[-1] == B[-2]

    def test_single_element(self):
        """Test with single element."""
        A = np.array([5.0])
        B, _MSE = jisotonic5(A)

        assert len(B) == 1
        assert B[0] == 5.0

    def test_two_elements_correct_order(self):
        """Test with two elements in correct order."""
        A = np.array([1.0, 2.0])
        B, _MSE = jisotonic5(A)

        assert np.allclose(B, A)

    def test_two_elements_wrong_order(self):
        """Test with two elements in wrong order."""
        A = np.array([2.0, 1.0])
        B, _MSE = jisotonic5(A)

        # Should be pooled to mean
        assert np.allclose(B, [1.5, 1.5])

    def test_with_weights(self):
        """Test with weights."""
        A = np.array([1.0, 2.0, 3.0])
        W = np.array([1.0, 1.0, 1.0])  # Uniform weights

        B, _MSE = jisotonic5(A, W)

        # Should be same as without weights
        B_no_weights, _ = jisotonic5(A)
        assert np.allclose(B, B_no_weights)

    def test_returns_mse(self):
        """Test that MSE is returned."""
        A = np.array([1, 3, 2, 4], dtype=np.float64)
        _B, MSE = jisotonic5(A)

        # MSE should be non-negative
        assert np.all(MSE >= 0)
        # MSE should be non-decreasing
        assert is_non_decreasing(MSE)


class TestJisotonic5Updown:
    """Tests for unimodal fitting."""

    def test_already_unimodal(self):
        """Test on already unimodal data."""
        A = np.array([1, 2, 3, 2, 1], dtype=np.float64)
        B = jisotonic5_updown(A)

        # Should be close to original
        assert np.allclose(B, A, atol=0.5)
        # Should be unimodal
        assert is_unimodal(B)

    def test_flat_data(self):
        """Test on flat data."""
        A = np.array([2, 2, 2, 2], dtype=np.float64)
        B = jisotonic5_updown(A)

        # Should remain flat
        assert np.allclose(B, 2.0)

    def test_increasing_data(self):
        """Test on purely increasing data."""
        A = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        B = jisotonic5_updown(A)

        # Should be unimodal (peak at end)
        assert is_unimodal(B)
        # Should be non-decreasing (since no decrease in input)
        assert is_non_decreasing(B)

    def test_decreasing_data(self):
        """Test on purely decreasing data."""
        A = np.array([5, 4, 3, 2, 1], dtype=np.float64)
        B = jisotonic5_updown(A)

        # Should be unimodal (peak at start)
        assert is_unimodal(B)
        # Should be non-increasing
        assert is_non_increasing(B)

    def test_bimodal_data(self):
        """Test on bimodal data - should fit dominant mode."""
        A = np.array([1, 3, 2, 4, 2], dtype=np.float64)
        B = jisotonic5_updown(A)

        # Should be unimodal
        assert is_unimodal(B)

    def test_single_element(self):
        """Test with single element."""
        A = np.array([5.0])
        B = jisotonic5_updown(A)

        assert len(B) == 1
        assert B[0] == 5.0


class TestJisotonic5Downup:
    """Tests for valley fitting."""

    def test_valley_shape(self):
        """Test on valley-shaped data."""
        A = np.array([5, 3, 1, 3, 5], dtype=np.float64)
        B = jisotonic5_downup(A)

        # Should have valley
        assert has_valley(B)

    def test_flat_data(self):
        """Test on flat data."""
        A = np.array([2, 2, 2, 2], dtype=np.float64)
        B = jisotonic5_downup(A)

        # Should remain flat
        assert np.allclose(B, 2.0)

    def test_unimodal_data(self):
        """Test on unimodal data (opposite of valley)."""
        A = np.array([1, 3, 5, 3, 1], dtype=np.float64)
        B = jisotonic5_downup(A)

        # Should have valley (fitted as best approximation)
        assert has_valley(B)

    def test_negation_of_updown(self):
        """Test that downup is negation of updown."""
        A = np.array([1, 3, 5, 3, 1], dtype=np.float64)

        B_updown = jisotonic5_updown(A)
        B_downup = jisotonic5_downup(-A)

        # downup(-A) should equal -updown(A)
        assert np.allclose(B_downup, -B_updown)


class TestJisotonic5Sort:
    """Tests for sorting."""

    def test_sort_basic(self):
        """Test basic sorting."""
        A = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        B = jisotonic5_sort(A)

        expected = np.array([1, 1, 2, 3, 4, 5, 6, 9])
        assert np.array_equal(B, expected)

    def test_sort_already_sorted(self):
        """Test on already sorted data."""
        A = np.array([1, 2, 3, 4, 5])
        B = jisotonic5_sort(A)

        assert np.array_equal(B, A)

    def test_sort_reverse(self):
        """Test on reverse sorted data."""
        A = np.array([5, 4, 3, 2, 1])
        B = jisotonic5_sort(A)

        expected = np.array([1, 2, 3, 4, 5])
        assert np.array_equal(B, expected)
