"""
Utility functions for testing isosplit6 implementations.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import adjusted_rand_score


def load_reference(name: str, reference_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load reference output from C++ implementation.

    Args:
        name: Name of the reference file (without .npz extension)
        reference_dir: Directory containing reference outputs

    Returns:
        Dictionary with loaded data

    Examples:
        >>> ref = load_reference('isocut6_unimodal_gaussian')
        >>> data = ref['data']
        >>> dipscore = ref['dipscore']
    """
    if reference_dir is None:
        reference_dir = Path(__file__).parent / "reference_outputs"

    filepath = reference_dir / f"{name}.npz"
    if not filepath.exists():
        raise FileNotFoundError(f"Reference file not found: {filepath}")

    return dict(np.load(filepath))


def labels_are_equivalent(labels1: np.ndarray, labels2: np.ndarray, tol: float = 0.99) -> bool:
    """
    Check if two labelings represent the same clustering (up to permutation).

    Uses adjusted Rand index to compare clusterings. Labels may be permuted
    (cluster 1 could be cluster 2), but the groupings should be the same.

    Args:
        labels1: First labeling
        labels2: Second labeling
        tol: Minimum adjusted Rand index to consider equivalent (default 0.99)

    Returns:
        True if clusterings are equivalent

    Examples:
        >>> labels1 = np.array([1, 1, 2, 2])
        >>> labels2 = np.array([2, 2, 1, 1])  # Same clustering, different labels
        >>> labels_are_equivalent(labels1, labels2)
        True
    """
    if len(labels1) != len(labels2):
        return False

    # Check if they're exactly the same first (fast path)
    if np.array_equal(labels1, labels2):
        return True

    # Use adjusted Rand index for permutation-invariant comparison
    ari = adjusted_rand_score(labels1, labels2)
    return ari >= tol


def clustering_purity(labels: np.ndarray, true_labels: np.ndarray) -> float:
    """
    Compute adjusted Rand index for clustering quality.

    Args:
        labels: Predicted cluster labels
        true_labels: Ground truth labels

    Returns:
        Adjusted Rand index (1.0 = perfect, 0.0 = random, <0 = worse than random)

    Examples:
        >>> labels = np.array([1, 1, 2, 2])
        >>> true_labels = np.array([1, 1, 2, 2])
        >>> clustering_purity(labels, true_labels)
        1.0
    """
    return adjusted_rand_score(true_labels, labels)


def assert_close(
    actual: float,
    expected: float,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    name: str = "value"
):
    """
    Assert that two floating point values are close.

    Args:
        actual: Actual value
        expected: Expected value
        rtol: Relative tolerance
        atol: Absolute tolerance
        name: Name of value for error message

    Raises:
        AssertionError: If values are not close

    Examples:
        >>> assert_close(1.0000000001, 1.0, rtol=1e-9)
    """
    if not np.isclose(actual, expected, rtol=rtol, atol=atol):
        raise AssertionError(
            f"{name}: expected {expected}, got {actual} "
            f"(diff={abs(actual - expected)}, rtol={rtol}, atol={atol})"
        )


def assert_arrays_close(
    actual: np.ndarray,
    expected: np.ndarray,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    name: str = "array"
):
    """
    Assert that two arrays are element-wise close.

    Args:
        actual: Actual array
        expected: Expected array
        rtol: Relative tolerance
        atol: Absolute tolerance
        name: Name of array for error message

    Raises:
        AssertionError: If arrays are not close

    Examples:
        >>> actual = np.array([1.0, 2.0, 3.0])
        >>> expected = np.array([1.0, 2.0, 3.0])
        >>> assert_arrays_close(actual, expected)
    """
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{name} shape mismatch: expected {expected.shape}, got {actual.shape}"
        )

    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        max_diff = np.max(np.abs(actual - expected))
        raise AssertionError(
            f"{name} not close (max diff={max_diff}, rtol={rtol}, atol={atol})"
        )


def check_labels_properties(labels: np.ndarray):
    """
    Check that labels have expected properties.

    Args:
        labels: Cluster labels to check

    Raises:
        AssertionError: If labels don't have expected properties

    Properties checked:
        - Labels are integers
        - Labels are positive
        - Labels are contiguous (1, 2, 3, ..., K)
    """
    # Check type
    assert labels.dtype in [np.int32, np.int64], f"Labels should be int, got {labels.dtype}"

    # Check positive
    assert np.all(labels > 0), "Labels should be positive"

    # Check contiguous
    unique_labels = np.unique(labels)
    expected_labels = np.arange(1, len(unique_labels) + 1)
    assert np.array_equal(unique_labels, expected_labels), \
        f"Labels should be contiguous 1, 2, 3, ..., got {unique_labels}"
