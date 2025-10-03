"""
Pytest configuration and fixtures for isosplit6 tests.
"""

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def reference_dir():
    """Get path to reference outputs directory."""
    return Path(__file__).parent / "reference_outputs"


@pytest.fixture
def two_gaussians_2d():
    """Two well-separated 2D Gaussians."""
    np.random.seed(42)
    cluster1 = np.random.randn(100, 2) + np.array([0.0, 0.0])
    cluster2 = np.random.randn(100, 2) + np.array([10.0, 0.0])
    data = np.vstack([cluster1, cluster2])
    true_labels = np.array([1] * 100 + [2] * 100)
    return data, true_labels


@pytest.fixture
def three_gaussians_2d():
    """Three well-separated 2D Gaussians."""
    np.random.seed(43)
    cluster1 = np.random.randn(100, 2) + np.array([0.0, 0.0])
    cluster2 = np.random.randn(100, 2) + np.array([10.0, 0.0])
    cluster3 = np.random.randn(100, 2) + np.array([5.0, 10.0])
    data = np.vstack([cluster1, cluster2, cluster3])
    true_labels = np.array([1] * 100 + [2] * 100 + [3] * 100)
    return data, true_labels


@pytest.fixture
def single_gaussian_2d():
    """Single 2D Gaussian."""
    np.random.seed(44)
    data = np.random.randn(200, 2)
    true_labels = np.ones(200, dtype=int)
    return data, true_labels


@pytest.fixture
def overlapping_gaussians_2d():
    """Two overlapping 2D Gaussians."""
    np.random.seed(45)
    cluster1 = np.random.randn(100, 2) + np.array([0.0, 0.0])
    cluster2 = np.random.randn(100, 2) + np.array([2.0, 0.0])
    data = np.vstack([cluster1, cluster2])
    true_labels = np.array([1] * 100 + [2] * 100)
    return data, true_labels


@pytest.fixture
def unimodal_1d():
    """Unimodal 1D Gaussian for isocut6 tests."""
    np.random.seed(100)
    return np.random.randn(200)


@pytest.fixture
def bimodal_separated_1d():
    """Well-separated bimodal 1D distribution."""
    np.random.seed(101)
    samples1 = np.random.randn(100) - 5.0
    samples2 = np.random.randn(100) + 5.0
    samples = np.concatenate([samples1, samples2])
    np.random.shuffle(samples)
    return samples


@pytest.fixture
def bimodal_touching_1d():
    """Touching bimodal 1D distribution."""
    np.random.seed(102)
    samples1 = np.random.randn(100) - 1.5
    samples2 = np.random.randn(100) + 1.5
    samples = np.concatenate([samples1, samples2])
    np.random.shuffle(samples)
    return samples
