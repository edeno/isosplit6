"""
Isocut6 algorithm for 1D dip testing.

This module implements the isocut6 algorithm which tests for bimodality
in 1D distributions using isotonic regression and a Kolmogorov-Smirnov-like
statistic.
"""

from typing import Tuple

import numpy as np

from .isotonic import jisotonic5_downup, jisotonic5_sort, jisotonic5_updown

__all__ = [
    "compute_ks4",
    "compute_ks5",
    "find_max_index",
    "find_min_index",
    "isocut6",
]


def find_min_index(arr: np.ndarray) -> int:
    """
    Find the index of the minimum value in an array.

    Parameters
    ----------
    arr : np.ndarray
        Input array of shape (N,)

    Returns
    -------
    int
        Index of minimum value

    Examples
    --------
    >>> arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    >>> find_min_index(arr)
    1

    References
    ----------
    Based on src/isocut5.cpp:195-203
    """
    return int(np.argmin(arr))


def find_max_index(arr: np.ndarray) -> int:
    """
    Find the index of the maximum value in an array.

    Parameters
    ----------
    arr : np.ndarray
        Input array of shape (N,)

    Returns
    -------
    int
        Index of maximum value

    Examples
    --------
    >>> arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    >>> find_max_index(arr)
    4

    References
    ----------
    Based on src/isocut5.cpp:205-213
    """
    return int(np.argmax(arr))


def compute_ks4(counts1: np.ndarray, counts2: np.ndarray) -> float:
    """
    Compute Kolmogorov-Smirnov-like statistic between two count distributions.

    Computes the maximum difference between cumulative distributions,
    scaled by the square root of the average total count.

    Parameters
    ----------
    counts1 : np.ndarray
        First count distribution of shape (N,)
    counts2 : np.ndarray
        Second count distribution of shape (N,)

    Returns
    -------
    float
        KS-like statistic (non-negative)

    Notes
    -----
    The statistic is computed as:
    max_diff * sqrt((sum_counts1 + sum_counts2) / 2)

    where max_diff is the maximum absolute difference between the
    cumulative distributions (normalized by their sums).

    Examples
    --------
    >>> counts1 = np.array([1.0, 2.0, 3.0])
    >>> counts2 = np.array([3.0, 2.0, 1.0])
    >>> ks = compute_ks4(counts1, counts2)
    >>> ks > 0
    True

    References
    ----------
    Based on src/isocut5.cpp:215-235
    """
    sum_counts1 = np.sum(counts1)
    sum_counts2 = np.sum(counts2)

    if sum_counts1 == 0 or sum_counts2 == 0:
        return 0.0

    cumsum_counts1 = np.cumsum(counts1)
    cumsum_counts2 = np.cumsum(counts2)

    # Compute maximum difference between normalized cumulative distributions
    diff = np.abs(cumsum_counts1 / sum_counts1 - cumsum_counts2 / sum_counts2)
    max_diff = np.max(diff)

    # Scale by square root of average total count
    return max_diff * np.sqrt((sum_counts1 + sum_counts2) / 2)


def compute_ks5(
    counts1: np.ndarray, counts2: np.ndarray, peak_index: int
) -> Tuple[float, int, int]:
    """
    Compute KS statistic with critical range search.

    Searches for the critical range (left or right of peak) that maximizes
    the KS statistic. This identifies the region of maximum deviation from
    unimodality.

    Parameters
    ----------
    counts1 : np.ndarray
        First count distribution of shape (N,)
    counts2 : np.ndarray
        Second count distribution of shape (N,)
    peak_index : int
        Index of the peak (mode) in the distribution

    Returns
    -------
    dipscore : float
        Maximum KS statistic found
    critical_range_min : int
        Start index of critical range
    critical_range_max : int
        End index of critical range (inclusive)

    Notes
    -----
    The algorithm searches both left and right of the peak, testing ranges
    of decreasing size (halving each time) while the range has at least 4
    elements.

    Examples
    --------
    >>> counts1 = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    >>> counts2 = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    >>> peak_idx = 2
    >>> dipscore, min_idx, max_idx = compute_ks5(counts1, counts2, peak_idx)
    >>> 0 <= min_idx <= max_idx < len(counts1)
    True

    References
    ----------
    Based on src/isocut5.cpp:237-288
    """
    N = len(counts1)
    critical_range_min = 0
    critical_range_max = N - 1
    ks_best = -1.0

    # Search from the left (up to and including peak)
    counts1_left = counts1[: peak_index + 1].copy()
    counts2_left = counts2[: peak_index + 1].copy()
    length = peak_index + 1

    while length >= 4 or length == peak_index + 1:
        ks0 = compute_ks4(counts1_left[:length], counts2_left[:length])
        if ks0 > ks_best:
            critical_range_min = 0
            critical_range_max = length - 1
            ks_best = ks0
        length = length // 2

    # Search from the right (from peak to end)
    counts1_right = counts1[peak_index:][::-1].copy()
    counts2_right = counts2[peak_index:][::-1].copy()
    length = N - peak_index

    while length >= 4 or length == N - peak_index:
        ks0 = compute_ks4(counts1_right[:length], counts2_right[:length])
        if ks0 > ks_best:
            critical_range_min = N - length
            critical_range_max = N - 1
            ks_best = ks0
        length = length // 2

    return ks_best, critical_range_min, critical_range_max


def isocut6(samples: np.ndarray, already_sorted: bool = False) -> Tuple[float, float]:
    """
    Compute dip statistic and cutpoint for 1D samples.

    Uses isotonic regression to fit a unimodal distribution and computes
    a Kolmogorov-Smirnov-like statistic to test for bimodality. Returns
    the dip score and optimal cutpoint for splitting the distribution.

    Parameters
    ----------
    samples : np.ndarray
        Input samples of shape (N,)
    already_sorted : bool, optional
        If True, assumes samples are already sorted. Default is False.

    Returns
    -------
    dipscore : float
        Dip statistic measuring deviation from unimodality
    cutpoint : float
        Optimal point to split the distribution

    Notes
    -----
    The algorithm:
    1. Sorts samples if needed
    2. Computes log densities from spacings between samples
    3. Fits unimodal (up-down) shape to log densities
    4. Finds critical range of maximum deviation
    5. Fits valley (down-up) shape to residuals on critical range
    6. Returns dip score and cutpoint at valley minimum

    Higher dip scores indicate stronger evidence for bimodality.

    Examples
    --------
    >>> # Unimodal data
    >>> samples = np.random.randn(100)
    >>> dipscore, cutpoint = isocut6(samples)
    >>> dipscore < 2.0  # Low dip score for unimodal
    True

    >>> # Bimodal data
    >>> samples = np.concatenate([
    ...     np.random.randn(100) - 5,
    ...     np.random.randn(100) + 5
    ... ])
    >>> dipscore, cutpoint = isocut6(samples)
    >>> dipscore > 5.0  # High dip score for bimodal
    True

    References
    ----------
    Based on src/isocut6.cpp:23-89
    """
    N = len(samples)

    # Sort samples if needed
    if already_sorted:
        samples_sorted = samples.copy()
    else:
        samples_sorted = jisotonic5_sort(samples)

    X = samples_sorted

    # Compute spacings and log densities
    spacings = np.diff(X)
    multiplicities = np.ones(N - 1)

    # Compute log densities (avoid log(0) with small epsilon)
    log_densities = np.zeros(N - 1)
    for i in range(N - 1):
        if spacings[i] > 0:
            log_densities[i] = np.log(multiplicities[i] / spacings[i])
        else:
            log_densities[i] = np.log(1e-9)  # Small value for zero spacing

    # Fit unimodal shape to log densities
    log_densities_unimodal_fit = jisotonic5_updown(log_densities, multiplicities)

    # Compute densities * spacings for KS statistic
    densities_unimodal_fit_times_spacings = (
        np.exp(log_densities_unimodal_fit) * spacings
    )

    # Find peak index
    peak_index = find_max_index(log_densities_unimodal_fit)

    # Compute KS statistic and critical range
    dipscore, critical_range_min, critical_range_max = compute_ks5(
        multiplicities, densities_unimodal_fit_times_spacings, peak_index
    )

    # Compute residuals and fit valley on critical range
    log_densities_resid = log_densities - log_densities_unimodal_fit

    critical_range_length = critical_range_max - critical_range_min + 1
    log_densities_resid_on_critical_range = log_densities_resid[
        critical_range_min : critical_range_max + 1
    ]
    weights_for_downup = np.ones(critical_range_length)

    log_densities_resid_fit_on_critical_range = jisotonic5_downup(
        log_densities_resid_on_critical_range, weights_for_downup
    )

    # Find cutpoint at minimum of valley fit
    cutpoint_index = find_min_index(log_densities_resid_fit_on_critical_range)
    cutpoint = (
        X[critical_range_min + cutpoint_index]
        + X[critical_range_min + cutpoint_index + 1]
    ) / 2

    return dipscore, cutpoint
