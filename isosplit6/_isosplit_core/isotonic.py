"""
Isotonic regression functions for isosplit6.

Implements Pool Adjacent Violators algorithm for fitting monotonic and unimodal functions.

Based on the C++ implementation in src/jisotonic5.cpp
"""

from typing import Optional, Tuple

import numpy as np


def jisotonic5(A: np.ndarray, W: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Isotonic (non-decreasing) regression using Pool Adjacent Violators algorithm.

    Fits a non-decreasing function to the input data that minimizes
    mean squared error.

    Parameters
    ----------
    A : np.ndarray
        Input array of shape (N,) to fit
    W : np.ndarray, optional
        Weights array of shape (N,). If None, uses uniform weights.

    Returns
    -------
    B : np.ndarray
        Fitted values (non-decreasing) of shape (N,)
    MSE : np.ndarray
        Mean squared error at each position of shape (N,)

    Examples
    --------
    >>> A = np.array([1, 3, 2, 4])
    >>> B, MSE = jisotonic5(A)
    >>> # B will be non-decreasing

    References
    ----------
    Based on src/jisotonic5.cpp:22-85
    """
    N = len(A)
    if N < 1:
        return np.array([]), np.array([])

    # Initialize weights if not provided
    if W is None:
        W = np.ones(N, dtype=np.float64)
    else:
        W = W.astype(np.float64)

    A = A.astype(np.float64)

    # Working arrays for PAV algorithm
    unweightedcount = np.zeros(N, dtype=np.float64)
    count = np.zeros(N, dtype=np.float64)
    sum_vals = np.zeros(N, dtype=np.float64)
    sumsqr = np.zeros(N, dtype=np.float64)
    MSE = np.zeros(N, dtype=np.float64)

    last_index = -1

    # Process first element
    last_index += 1
    unweightedcount[last_index] = 1
    w0 = W[0]
    count[last_index] = w0
    sum_vals[last_index] = A[0] * w0
    sumsqr[last_index] = A[0] * A[0] * w0
    MSE[0] = 0

    # Process remaining elements
    for j in range(1, N):
        last_index += 1
        unweightedcount[last_index] = 1
        w0 = W[j]
        count[last_index] = w0
        sum_vals[last_index] = A[j] * w0
        sumsqr[last_index] = A[j] * A[j] * w0
        MSE[j] = MSE[j - 1]

        # Pool adjacent violators
        while True:
            if last_index <= 0:
                break

            # Check if we need to pool (current level is less than previous)
            if sum_vals[last_index - 1] / count[last_index - 1] < sum_vals[last_index] / count[last_index]:
                break
            else:
                # Pool the last two blocks
                prevMSE = sumsqr[last_index - 1] - sum_vals[last_index - 1] * sum_vals[last_index - 1] / count[last_index - 1]
                prevMSE += sumsqr[last_index] - sum_vals[last_index] * sum_vals[last_index] / count[last_index]

                unweightedcount[last_index - 1] += unweightedcount[last_index]
                count[last_index - 1] += count[last_index]
                sum_vals[last_index - 1] += sum_vals[last_index]
                sumsqr[last_index - 1] += sumsqr[last_index]

                newMSE = sumsqr[last_index - 1] - sum_vals[last_index - 1] * sum_vals[last_index - 1] / count[last_index - 1]
                MSE[j] += newMSE - prevMSE
                last_index -= 1

    # Construct output
    B = np.zeros(N, dtype=np.float64)
    ii = 0
    for k in range(last_index + 1):
        n_pts = int(unweightedcount[k])
        value = sum_vals[k] / count[k]
        B[ii:ii + n_pts] = value
        ii += n_pts

    return B, MSE


def jisotonic5_updown(A: np.ndarray, W: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Fit unimodal (up-then-down) function using isotonic regression.

    Tries all possible peak locations and selects the one with minimum
    mean squared error.

    Parameters
    ----------
    A : np.ndarray
        Input array of shape (N,)
    W : np.ndarray, optional
        Weights array of shape (N,)

    Returns
    -------
    np.ndarray
        Fitted values of shape (N,) with single peak

    Examples
    --------
    >>> A = np.array([1, 2, 3, 2, 1])  # Already unimodal
    >>> B = jisotonic5_updown(A)
    >>> # B will be close to A

    References
    ----------
    Based on src/jisotonic5.cpp:87-131
    """
    N = len(A)
    if N < 1:
        return np.array([])

    # Prepare reversed arrays
    A_reversed = A[::-1].copy()
    W_reversed = None if W is None else W[::-1].copy()

    # Fit isotonic (increasing) on full forward array
    _, MSE1 = jisotonic5(A, W)

    # Fit isotonic (increasing) on full reversed array
    _, MSE2 = jisotonic5(A_reversed, W_reversed)

    # Combine MSEs to find best peak location
    # MSE1[j] is MSE for increasing fit up to j
    # MSE2[N-1-j] is MSE for decreasing fit after j
    total_MSE = MSE1 + MSE2[::-1]

    # Find index with minimum total MSE
    best_ind = np.argmin(total_MSE)

    # Refit with best peak location
    # Fit increasing from start to peak
    B1_best, _ = jisotonic5(A[:best_ind + 1], None if W is None else W[:best_ind + 1])

    # Fit decreasing from peak to end (by reversing)
    if N - best_ind > 0:
        B2_best, _ = jisotonic5(A_reversed[:N - best_ind], None if W_reversed is None else W_reversed[:N - best_ind])
    else:
        B2_best = np.array([])

    # Construct output
    out = np.zeros(N, dtype=np.float64)
    # Copy increasing part (indices 0 to best_ind)
    out[:best_ind + 1] = B1_best
    # Copy decreasing part (indices best_ind+1 to N-1)
    # B2_best is in reversed order, so we reverse back and skip the first element (which is the peak)
    for j in range(N - best_ind - 1):
        out[N - 1 - j] = B2_best[j]

    return out


def jisotonic5_downup(A: np.ndarray, W: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Fit valley (down-then-up) function using isotonic regression.

    Implemented by negating the input, fitting unimodal, and negating back.

    Parameters
    ----------
    A : np.ndarray
        Input array of shape (N,)
    W : np.ndarray, optional
        Weights array of shape (N,)

    Returns
    -------
    np.ndarray
        Fitted values of shape (N,) with single valley

    Examples
    --------
    >>> A = np.array([3, 2, 1, 2, 3])  # Valley shape
    >>> B = jisotonic5_downup(A)
    >>> # B will be close to A

    References
    ----------
    Based on src/jisotonic5.cpp:133-144
    """
    if len(A) < 1:
        return np.array([])

    # Negate input
    A_neg = -A

    # Fit unimodal (will be updown on negated data)
    out_neg = jisotonic5_updown(A_neg, W)

    # Negate back
    return -out_neg


def jisotonic5_sort(samples: np.ndarray) -> np.ndarray:
    """
    Sort samples in ascending order.

    Simple wrapper around np.sort for consistency with C++ implementation.

    Parameters
    ----------
    samples : np.ndarray
        Array to sort

    Returns
    -------
    np.ndarray
        Sorted array

    Examples
    --------
    >>> samples = np.array([3, 1, 2])
    >>> jisotonic5_sort(samples)
    array([1, 2, 3])

    References
    ----------
    Based on src/jisotonic5.cpp:146-155
    """
    return np.sort(samples)
