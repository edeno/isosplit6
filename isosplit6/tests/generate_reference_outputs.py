"""
Generate reference outputs from the C++ implementation.

This script creates test fixtures and runs the C++ isosplit6/isocut6
implementations on them, saving the outputs for regression testing.

Run this script whenever the C++ implementation changes to update
reference outputs.
"""

from pathlib import Path

import numpy as np

from isosplit6 import isocut6, isosplit6


def save_reference(name, data, output, output_dir):
    """Save test data and reference output."""
    filepath = output_dir / f"{name}.npz"
    if isinstance(output, tuple):
        # isocut6 returns (dipscore, cutpoint)
        np.savez(filepath, data=data, dipscore=output[0], cutpoint=output[1])
    else:
        # isosplit6 returns labels
        np.savez(filepath, data=data, labels=output)
    print(f"Saved: {filepath}")


def generate_isocut6_references(output_dir):
    """Generate reference outputs for isocut6 tests."""
    print("\n=== Generating isocut6 reference outputs ===")

    # Test 1: Unimodal Gaussian
    np.random.seed(42)
    samples = np.random.randn(200)
    dipscore, cutpoint = isocut6(samples)
    save_reference("isocut6_unimodal_gaussian", samples, (dipscore, cutpoint), output_dir)
    print(f"  Unimodal Gaussian: dipscore={dipscore:.6f}, cutpoint={cutpoint:.6f}")

    # Test 2: Well-separated bimodal
    np.random.seed(43)
    samples1 = np.random.randn(100) - 5.0
    samples2 = np.random.randn(100) + 5.0
    samples = np.concatenate([samples1, samples2])
    np.random.shuffle(samples)
    dipscore, cutpoint = isocut6(samples)
    save_reference("isocut6_bimodal_separated", samples, (dipscore, cutpoint), output_dir)
    print(f"  Bimodal separated: dipscore={dipscore:.6f}, cutpoint={cutpoint:.6f}")

    # Test 3: Touching bimodal
    np.random.seed(44)
    samples1 = np.random.randn(100) - 1.5
    samples2 = np.random.randn(100) + 1.5
    samples = np.concatenate([samples1, samples2])
    np.random.shuffle(samples)
    dipscore, cutpoint = isocut6(samples)
    save_reference("isocut6_bimodal_touching", samples, (dipscore, cutpoint), output_dir)
    print(f"  Bimodal touching: dipscore={dipscore:.6f}, cutpoint={cutpoint:.6f}")

    # Test 4: Uniform distribution
    np.random.seed(45)
    samples = np.random.uniform(-5, 5, 200)
    dipscore, cutpoint = isocut6(samples)
    save_reference("isocut6_uniform", samples, (dipscore, cutpoint), output_dir)
    print(f"  Uniform: dipscore={dipscore:.6f}, cutpoint={cutpoint:.6f}")

    # Test 5: Trimodal
    np.random.seed(46)
    samples1 = np.random.randn(100) - 6.0
    samples2 = np.random.randn(100)
    samples3 = np.random.randn(100) + 6.0
    samples = np.concatenate([samples1, samples2, samples3])
    np.random.shuffle(samples)
    dipscore, cutpoint = isocut6(samples)
    save_reference("isocut6_trimodal", samples, (dipscore, cutpoint), output_dir)
    print(f"  Trimodal: dipscore={dipscore:.6f}, cutpoint={cutpoint:.6f}")

    # Test 6: Sorted input (should give same result as unsorted)
    np.random.seed(47)
    samples_unsorted = np.random.randn(200)
    samples_sorted = np.sort(samples_unsorted)
    dipscore_unsorted, cutpoint_unsorted = isocut6(samples_unsorted)
    dipscore_sorted, cutpoint_sorted = isocut6(samples_sorted)
    save_reference("isocut6_unsorted", samples_unsorted, (dipscore_unsorted, cutpoint_unsorted), output_dir)
    save_reference("isocut6_sorted", samples_sorted, (dipscore_sorted, cutpoint_sorted), output_dir)
    print(f"  Unsorted: dipscore={dipscore_unsorted:.6f}, cutpoint={cutpoint_unsorted:.6f}")
    print(f"  Sorted:   dipscore={dipscore_sorted:.6f}, cutpoint={cutpoint_sorted:.6f}")
    print(f"  Match: {np.isclose(dipscore_unsorted, dipscore_sorted) and np.isclose(cutpoint_unsorted, cutpoint_sorted)}")


def generate_isosplit6_references(output_dir):
    """Generate reference outputs for isosplit6 tests."""
    print("\n=== Generating isosplit6 reference outputs ===")

    # Test 1: Two well-separated Gaussians (2D)
    np.random.seed(100)
    cluster1 = np.random.randn(100, 2) + np.array([0.0, 0.0])
    cluster2 = np.random.randn(100, 2) + np.array([10.0, 0.0])
    data = np.vstack([cluster1, cluster2])
    labels = isosplit6(data)
    save_reference("isosplit6_two_gaussians_2d", data, labels, output_dir)
    print(f"  Two Gaussians 2D: {len(np.unique(labels))} clusters found")

    # Test 2: Three well-separated Gaussians (2D)
    np.random.seed(101)
    cluster1 = np.random.randn(100, 2) + np.array([0.0, 0.0])
    cluster2 = np.random.randn(100, 2) + np.array([10.0, 0.0])
    cluster3 = np.random.randn(100, 2) + np.array([5.0, 10.0])
    data = np.vstack([cluster1, cluster2, cluster3])
    labels = isosplit6(data)
    save_reference("isosplit6_three_gaussians_2d", data, labels, output_dir)
    print(f"  Three Gaussians 2D: {len(np.unique(labels))} clusters found")

    # Test 3: Single Gaussian (should find 1 cluster)
    np.random.seed(102)
    data = np.random.randn(200, 2)
    labels = isosplit6(data)
    save_reference("isosplit6_single_gaussian", data, labels, output_dir)
    print(f"  Single Gaussian: {len(np.unique(labels))} clusters found")

    # Test 4: High-dimensional (10D)
    np.random.seed(103)
    cluster1 = np.random.randn(100, 10)
    cluster2 = np.random.randn(100, 10) + 5.0
    data = np.vstack([cluster1, cluster2])
    labels = isosplit6(data)
    save_reference("isosplit6_high_dim_10d", data, labels, output_dir)
    print(f"  High-dim 10D: {len(np.unique(labels))} clusters found")

    # Test 5: Different cluster sizes
    np.random.seed(104)
    cluster1 = np.random.randn(1000, 2) + np.array([0.0, 0.0])
    cluster2 = np.random.randn(50, 2) + np.array([10.0, 0.0])
    data = np.vstack([cluster1, cluster2])
    labels = isosplit6(data)
    save_reference("isosplit6_different_sizes", data, labels, output_dir)
    print(f"  Different sizes: {len(np.unique(labels))} clusters found")

    # Test 6: Overlapping Gaussians
    np.random.seed(105)
    cluster1 = np.random.randn(100, 2) + np.array([0.0, 0.0])
    cluster2 = np.random.randn(100, 2) + np.array([2.0, 0.0])
    data = np.vstack([cluster1, cluster2])
    labels = isosplit6(data)
    save_reference("isosplit6_overlapping", data, labels, output_dir)
    print(f"  Overlapping: {len(np.unique(labels))} clusters found")

    # Test 7: 1D data
    np.random.seed(106)
    cluster1 = np.random.randn(100, 1) - 3.0
    cluster2 = np.random.randn(100, 1) + 3.0
    data = np.vstack([cluster1, cluster2])
    labels = isosplit6(data)
    save_reference("isosplit6_1d_two_clusters", data, labels, output_dir)
    print(f"  1D two clusters: {len(np.unique(labels))} clusters found")

    # Test 8: Random data (for determinism test)
    np.random.seed(107)
    data = np.random.randn(300, 5)
    labels = isosplit6(data)
    save_reference("isosplit6_random_5d", data, labels, output_dir)
    print(f"  Random 5D: {len(np.unique(labels))} clusters found")


def main():
    """Generate all reference outputs."""
    # Create output directory
    output_dir = Path(__file__).parent / "reference_outputs"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Generating Reference Outputs from C++ Implementation")
    print("=" * 60)

    # Generate references
    generate_isocut6_references(output_dir)
    generate_isosplit6_references(output_dir)

    print("\n" + "=" * 60)
    print("Reference output generation complete!")
    print(f"Outputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
