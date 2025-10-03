"""
Diagnostic tests to investigate 3-cluster failures.

This module tests whether the NumPy implementation can find 3 clusters
with different initializations to confirm the root cause is initialization
rather than an algorithm bug.
"""

import numpy as np

from isosplit6.isosplit6_numpy import isosplit6

from .utils import load_reference


def test_three_clusters_different_seeds():
    """Test if different random seeds can find 3 clusters."""
    # Load the 3-cluster reference data
    ref = load_reference("isosplit6_three_gaussians_2d")
    data = ref["data"]

    print("\n" + "="*60)
    print("Testing 3-cluster data with different random seeds")
    print("="*60)

    results = {}
    for seed in range(20):
        labels = isosplit6(data, K_init=200, random_state=seed)
        n_clusters = len(np.unique(labels))
        results[seed] = n_clusters

        if n_clusters == 3:
            print(f"✓ Seed {seed}: Found {n_clusters} clusters - MATCHES C++!")
        else:
            print(f"  Seed {seed}: Found {n_clusters} clusters")

    print("\nSummary:")
    print(f"Seeds that found 3 clusters: {[s for s, n in results.items() if n == 3]}")
    print(f"Seeds that found 2 clusters: {[s for s, n in results.items() if n == 2]}")
    print(f"Seeds that found 1 cluster: {[s for s, n in results.items() if n == 1]}")

    # If we find 3 clusters with ANY seed, it confirms initialization issue
    if any(n == 3 for n in results.values()):
        print("\n✓ CONFIRMED: Found 3 clusters with different initialization")
        print("  Root cause: KMeans initialization differences from C++ parcelate2")


def test_three_clusters_different_k_init():
    """Test if different K_init values can find 3 clusters."""
    ref = load_reference("isosplit6_three_gaussians_2d")
    data = ref["data"]

    print("\n" + "="*60)
    print("Testing 3-cluster data with different K_init values")
    print("="*60)

    results = {}
    for k_init in [10, 20, 50, 100, 150, 200, 300]:
        labels = isosplit6(data, K_init=k_init, random_state=42)
        n_clusters = len(np.unique(labels))
        results[k_init] = n_clusters

        print(f"K_init={k_init:3d}: Found {n_clusters} clusters")

    if any(n == 3 for n in results.values()):
        print("\n✓ CONFIRMED: Found 3 clusters with different K_init")


def test_three_clusters_manual_initialization():
    """Test with manual initialization based on true cluster structure."""
    ref = load_reference("isosplit6_three_gaussians_2d")
    data = ref["data"]
    expected_labels = ref["labels"]

    print("\n" + "="*60)
    print("Testing 3-cluster data with perfect initial labels")
    print("="*60)

    # Use the true labels as initial labels
    # This should definitely produce 3 clusters if algorithm is correct
    labels = isosplit6(data, initial_labels=expected_labels)
    n_clusters = len(np.unique(labels))

    print(f"Starting with perfect 3-cluster initialization:")
    print(f"  Final result: {n_clusters} clusters")

    if n_clusters == 3:
        print("✓ CONFIRMED: Algorithm preserves 3 clusters when initialized correctly")
        print("  Root cause: Initialization differences")
    else:
        print("⚠ UNEXPECTED: Algorithm merged clusters even with perfect initialization")
        print("  This suggests a potential algorithm issue, not just initialization")

        # Check which clusters merged
        for k in range(1, 4):
            mask = expected_labels == k
            final_clusters_for_k = np.unique(labels[mask])
            print(f"  Original cluster {k} → final clusters: {final_clusters_for_k}")


def test_check_cluster_geometry():
    """Analyze the geometry of the 3-cluster problem."""
    ref = load_reference("isosplit6_three_gaussians_2d")
    data = ref["data"]
    expected_labels = ref["labels"]

    print("\n" + "="*60)
    print("Analyzing 3-cluster geometry")
    print("="*60)

    # Compute centroids and pairwise distances
    centroids = []
    for k in range(1, 4):
        mask = expected_labels == k
        centroid = np.mean(data[mask], axis=0)
        centroids.append(centroid)
        print(f"Cluster {k} centroid: [{centroid[0]:6.2f}, {centroid[1]:6.2f}]")

    print("\nPairwise centroid distances:")
    centroids = np.array(centroids)
    for i in range(3):
        for j in range(i+1, 3):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            print(f"  Cluster {i+1} ↔ Cluster {j+1}: {dist:.2f}")

    # Check which pair is closest (most likely to merge)
    dists = []
    for i in range(3):
        for j in range(i+1, 3):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            dists.append((dist, i+1, j+1))

    dists.sort()
    print(f"\nClosest pair: Cluster {dists[0][1]} ↔ Cluster {dists[0][2]} (dist={dists[0][0]:.2f})")
    print("  → This pair is most likely to be compared and merged first")


if __name__ == "__main__":
    test_check_cluster_geometry()
    test_three_clusters_manual_initialization()
    test_three_clusters_different_seeds()
    test_three_clusters_different_k_init()
