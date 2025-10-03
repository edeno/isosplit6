# Test Plan for Isosplit6

This document outlines a comprehensive test plan for verifying the correctness of isosplit6 implementations (C++, NumPy, JAX).

## Goals

1. **Verify algorithmic correctness** - Ensure the algorithm produces correct clusterings on known data
2. **Enable cross-implementation validation** - Compare C++, NumPy, and JAX implementations
3. **Catch regressions** - Detect when changes break existing functionality
4. **Document expected behavior** - Serve as executable specifications

## Test Categories

### 1. Unit Tests for Core Components

These tests verify individual functions work correctly in isolation.

#### 1.1 Isocut6 (1D Dip Test)

**Purpose:** Test the core statistical test that decides if 1D data is unimodal or bimodal.

**Test cases:**

- **Unimodal Gaussian**: Single Gaussian → low dip score, merge decision
- **Well-separated bimodal**: Two Gaussians far apart → high dip score, split decision
- **Touching bimodal**: Two Gaussians close together → moderate dip score
- **Uniform distribution**: Flat distribution → low dip score (unimodal)
- **Trimodal**: Three modes → should detect at least one split
- **Sorted vs unsorted**: Same result whether input is pre-sorted or not
- **Known cut points**: Verify cut point location for simple cases

**Reference outputs:** Run C++ implementation on fixed data, save results

**Assertions:**

```python
dipscore, cutpoint = isocut6(samples)
assert dipscore > 0  # Always positive
assert min(samples) <= cutpoint <= max(samples)  # Cut point in range
```

#### 1.2 Isotonic Regression Functions

**Purpose:** Test the monotonic fitting utilities.

**Test cases:**

- **jisotonic5 (non-decreasing fit)**:
  - Already monotonic data → no change
  - Decreasing data → flat line at mean
  - Noisy increasing data → smooth monotonic curve

- **jisotonic5_updown (unimodal fit)**:
  - Perfect unimodal → no change
  - Flat data → flat line
  - Bimodal data → fits to dominant mode

- **jisotonic5_downup (valley fit)**:
  - Perfect valley → no change
  - Unimodal data → flat line or slight dip

**Reference outputs:** Use known mathematical properties or C++ outputs

#### 1.3 Helper Functions

Test any utility functions:

- Centroid computation
- Covariance matrix computation
- Distance calculations
- Matrix inversion
- Mahalanobis distance

### 2. Integration Tests for Isosplit6

These tests verify the full algorithm works correctly on synthetic data with known structure.

#### 2.1 Two Well-Separated Gaussians

**Setup:**

```python
# 2D: Two Gaussians separated by 10 standard deviations
cluster1 = np.random.randn(100, 2) + [0, 0]
cluster2 = np.random.randn(100, 2) + [10, 0]
data = np.vstack([cluster1, cluster2])
```

**Expected:** 2 clusters found, high purity (>95%)

**Rationale:** This is the easiest case - algorithm must handle this

#### 2.2 Three Well-Separated Gaussians

**Setup:**

```python
# 3 clusters in 2D, arranged in triangle
cluster1 = np.random.randn(100, 2) + [0, 0]
cluster2 = np.random.randn(100, 2) + [10, 0]
cluster3 = np.random.randn(100, 2) + [5, 10]
data = np.vstack([cluster1, cluster2, cluster3])
```

**Expected:** 3 clusters found, high purity (>90%)

#### 2.3 Overlapping Gaussians

**Setup:**

```python
# Two Gaussians with centers only 2 standard deviations apart
cluster1 = np.random.randn(100, 2) + [0, 0]
cluster2 = np.random.randn(100, 2) + [2, 0]
data = np.vstack([cluster1, cluster2])
```

**Expected:** Might find 1 or 2 clusters depending on overlap

**Rationale:** Tests sensitivity to isocut_threshold

#### 2.4 Nested/Concentric Clusters

**Setup:**

```python
# Inner circle and outer ring
inner = np.random.randn(50, 2) * 0.5
outer = np.random.randn(100, 2) * 2.0 + 5.0
# Arrange outer in a circle
```

**Expected:** Should find 2 clusters if well-separated in density

#### 2.5 Different Cluster Sizes

**Setup:**

```python
# One large cluster (1000 points) and one small (20 points)
cluster1 = np.random.randn(1000, 2) + [0, 0]
cluster2 = np.random.randn(20, 2) + [10, 0]
```

**Expected:** Both clusters found (tests min_cluster_size handling)

#### 2.6 High Dimensional Data

**Setup:**

```python
# 2 clusters in 10D
cluster1 = np.random.randn(100, 10)
cluster2 = np.random.randn(100, 10) + 5.0
```

**Expected:** 2 clusters found

**Rationale:** Tests that projection approach works in high dimensions

#### 2.7 Single Cluster (Unimodal)

**Setup:**

```python
# Single Gaussian
data = np.random.randn(200, 2)
```

**Expected:** 1 cluster found

**Rationale:** Algorithm should not over-split

### 3. Regression Tests (Cross-Implementation Validation)

These tests compare implementations against each other.

#### 3.1 Fixed Random Seed Tests

**Purpose:** Ensure different implementations produce identical results

**Test cases:**

- Generate 5-10 random datasets with fixed seeds
- Run C++ implementation, save outputs
- Run NumPy/JAX implementations, compare outputs

**Assertions:**

```python
labels_cpp = load_cpp_reference_output('test_case_1.npy')
labels_numpy = isosplit6_numpy(data)
labels_jax = isosplit6_jax(data)

# Labels might be permuted (cluster 1 could be cluster 2), so check equivalence
assert labels_are_equivalent(labels_cpp, labels_numpy)
assert labels_are_equivalent(labels_cpp, labels_jax)
```

**Note:** Need to handle label permutation (cluster IDs are arbitrary)

#### 3.2 Isocut6 Regression Tests

**Purpose:** Verify isocut6 implementations match exactly

**Test cases:**

- 10-20 fixed 1D arrays
- Run C++ isocut6, save (dipscore, cutpoint)
- Compare NumPy/JAX implementations

**Assertions:**

```python
dipscore_cpp, cutpoint_cpp = load_cpp_reference('isocut_test_1.npy')
dipscore_numpy, cutpoint_numpy = isocut6_numpy(samples)

assert np.isclose(dipscore_cpp, dipscore_numpy, rtol=1e-10)
assert np.isclose(cutpoint_cpp, cutpoint_numpy, rtol=1e-10)
```

### 4. Property-Based Tests

These tests verify mathematical properties that should always hold.

#### 4.1 Label Consistency

```python
labels = isosplit6(data)
assert len(labels) == len(data)  # Every point gets a label
assert labels.dtype == np.int32  # Correct type
assert np.all(labels > 0)  # Labels are positive
```

#### 4.2 Label Contiguity

```python
labels = isosplit6(data)
unique_labels = np.unique(labels)
assert np.array_equal(unique_labels, np.arange(1, len(unique_labels) + 1))
# Labels are 1, 2, 3, ... with no gaps
```

#### 4.3 Cluster Size Constraints

```python
labels = isosplit6(data, min_cluster_size=20)
for label in np.unique(labels):
    cluster_size = np.sum(labels == label)
    assert cluster_size >= 20  # Respects minimum
```

#### 4.4 Determinism

```python
labels1 = isosplit6(data)
labels2 = isosplit6(data)
assert np.array_equal(labels1, labels2)  # Same input → same output
```

#### 4.5 Permutation Invariance

```python
# Shuffling rows should give equivalent clustering
perm = np.random.permutation(len(data))
labels1 = isosplit6(data)
labels2 = isosplit6(data[perm])
assert labels_are_equivalent(labels1, labels2[np.argsort(perm)])
```

### 5. Edge Cases

#### 5.1 Minimal Data

```python
# Empty array
with pytest.raises(ValueError):
    isosplit6(np.array([]).reshape(0, 2))

# Single point
labels = isosplit6(np.array([[1.0, 2.0]]))
assert len(labels) == 1
assert labels[0] == 1

# Two points
labels = isosplit6(np.array([[0.0, 0.0], [1.0, 1.0]]))
assert len(labels) == 2
# Might be 1 or 2 clusters depending on algorithm
```

#### 5.2 Degenerate Data

```python
# All identical points
data = np.ones((100, 2))
labels = isosplit6(data)
assert len(np.unique(labels)) == 1  # Should find 1 cluster

# All points on a line
data = np.column_stack([np.linspace(0, 10, 100), np.zeros(100)])
labels = isosplit6(data)
# Should handle gracefully (covariance matrix near-singular)
```

#### 5.3 Single Dimension

```python
# 1D data (N x 1 array)
data = np.random.randn(100, 1)
labels = isosplit6(data)
assert len(labels) == 100
```

### 6. Parameter Sensitivity Tests

#### 6.1 Isocut Threshold

```python
data = generate_overlapping_gaussians(separation=2.0)

# Low threshold → more merging
labels_low = isosplit6(data, isocut_threshold=1.0)
n_clusters_low = len(np.unique(labels_low))

# High threshold → less merging
labels_high = isosplit6(data, isocut_threshold=5.0)
n_clusters_high = len(np.unique(labels_high))

assert n_clusters_high >= n_clusters_low
```

#### 6.2 Minimum Cluster Size

```python
data = generate_clusters_with_outliers()

labels_small = isosplit6(data, min_cluster_size=5)
labels_large = isosplit6(data, min_cluster_size=50)

n_clusters_small = len(np.unique(labels_small))
n_clusters_large = len(np.unique(labels_large))

assert n_clusters_large <= n_clusters_small  # Larger min → fewer clusters
```

#### 6.3 Initial Clusters (K_init)

```python
# More initial clusters → potentially finer distinctions
labels_k200 = isosplit6(data, K_init=200)
labels_k50 = isosplit6(data, K_init=50)

# Results should be similar but might differ slightly
# At minimum, should not crash
```

### 7. Initial Labels Tests

Test the ability to refine existing clusterings.

```python
# Start with ground truth labels
data, true_labels = generate_labeled_data()
initial_labels = add_noise_to_labels(true_labels)  # Perturb slightly

labels = isosplit6(data, initial_labels=initial_labels)

# Should improve over initial labels
assert clustering_purity(labels, true_labels) >= clustering_purity(initial_labels, true_labels)
```

### 8. Performance/Stress Tests

#### 8.1 Large Data

```python
# Test on larger datasets
data = np.random.randn(10000, 10)
labels = isosplit6(data)
# Should complete in reasonable time (<30 seconds)
```

#### 8.2 Many Clusters

```python
# 10 well-separated clusters
data = generate_many_clusters(n_clusters=10, separation=5.0)
labels = isosplit6(data)
assert len(np.unique(labels)) >= 8  # Should find most/all
```

## Test Utilities Needed

### Helper Functions

```python
def labels_are_equivalent(labels1, labels2):
    """Check if two labelings represent the same clustering (up to permutation)"""
    # Use adjusted Rand index or similar
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(labels1, labels2) == 1.0

def clustering_purity(labels, true_labels):
    """Compute purity score for clustering quality"""
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(labels, true_labels)

def generate_gaussian_clusters(n_clusters, n_points_per_cluster, n_dims, separation):
    """Generate synthetic data with known cluster structure"""
    # Gaussians arranged on a grid or randomly
    pass

def generate_overlapping_gaussians(separation):
    """Generate two Gaussians with controlled overlap"""
    pass

def save_reference_output(data, labels, filename):
    """Save C++ output for regression testing"""
    np.savez(filename, data=data, labels=labels)

def load_reference_output(filename):
    """Load C++ reference output"""
    return np.load(filename)
```

### Test Data Fixtures

Create reusable test datasets:

```python
@pytest.fixture
def two_gaussians_2d():
    """Two well-separated 2D Gaussians"""
    np.random.seed(42)
    cluster1 = np.random.randn(100, 2) + [0, 0]
    cluster2 = np.random.randn(100, 2) + [10, 0]
    data = np.vstack([cluster1, cluster2])
    true_labels = np.array([1]*100 + [2]*100)
    return data, true_labels

@pytest.fixture
def three_gaussians_2d():
    """Three well-separated 2D Gaussians"""
    # Similar structure
    pass

# Store these as .npz files for consistency across implementations
```

## Test Organization

### File Structure

```
isosplit6/tests/
├── __init__.py
├── test_simple_run.py              # Existing smoke test
├── test_isocut6.py                 # Unit tests for isocut6
├── test_isotonic.py                # Unit tests for isotonic regression
├── test_isosplit6_synthetic.py     # Integration tests on synthetic data
├── test_isosplit6_edge_cases.py    # Edge case tests
├── test_isosplit6_properties.py    # Property-based tests
├── test_isosplit6_parameters.py    # Parameter sensitivity tests
├── test_cross_implementation.py    # Regression tests vs C++
├── conftest.py                     # Pytest fixtures and helpers
├── utils.py                        # Test utility functions
└── reference_outputs/              # Directory for C++ reference outputs
    ├── isocut_test_001.npz
    ├── isocut_test_002.npz
    ├── isosplit_test_001.npz
    └── ...
```

### Test Naming Convention

```python
def test_<function>_<scenario>_<expected_behavior>():
    """
    Brief description of what this test verifies.
    """
    pass

# Examples:
def test_isocut6_unimodal_gaussian_low_dipscore():
    """Verify that isocut6 returns low dip score for unimodal Gaussian."""
    pass

def test_isosplit6_two_separated_gaussians_finds_two_clusters():
    """Verify that isosplit6 correctly identifies two well-separated clusters."""
    pass
```

## Metrics for Success

### Minimum Viable Test Suite

For a NumPy/JAX reimplementation to be considered correct, it must pass:

1. ✅ All unit tests for isocut6 (exact match with C++)
2. ✅ All unit tests for isotonic regression (exact match with C++)
3. ✅ At least 5 regression tests with C++ (exact label equivalence)
4. ✅ All property-based tests
5. ✅ All edge case tests (or explicitly document differences)
6. ✅ At least 10 synthetic data tests with >90% clustering accuracy

### Nice-to-Have

- Performance benchmarks (should be comparable to C++)
- Visualization of test cases (scatter plots with cluster assignments)
- Test coverage >80%
- Randomized property-based tests (using Hypothesis library)

## Implementation Priority

1. **Phase 1: Core validation**
   - Set up test infrastructure
   - Implement helper functions
   - Create test data fixtures
   - Write unit tests for isocut6

2. **Phase 2: Integration testing**
   - Generate reference outputs from C++
   - Implement synthetic data tests
   - Implement property-based tests

3. **Phase 3: Edge cases and parameters**
   - Edge case tests
   - Parameter sensitivity tests
   - Initial labels tests

4. **Phase 4: Cross-implementation**
   - Regression tests against C++
   - Performance benchmarks
   - Documentation of any differences

## Notes for NumPy/JAX Implementations

### Numerical Precision

- Use `float64` throughout to match C++ implementation
- Be aware of numerical differences in:
  - Matrix inversion (different algorithms)
  - Sorting (stable vs unstable)
  - Random initialization (if applicable)

### Differences to Document

If NumPy/JAX implementations differ from C++:

1. **Document why** (e.g., numerical stability, algorithmic improvement)
2. **Quantify difference** (how often? by how much?)
3. **Verify improvement** (is the difference beneficial?)

### Testing Strategy

1. Start with unit tests (isocut6, isotonic regression)
2. Get these matching C++ exactly (within floating point tolerance)
3. Move to integration tests
4. Accept minor differences in full algorithm if justified

## References

- Current test: `isosplit6/tests/test_simple_run.py`
- Algorithm description: `ISOSPLIT.md`
- C++ implementation: `src/isosplit6.cpp`, `src/isocut6.cpp`
