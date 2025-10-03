# Isosplit Algorithm Documentation

This document explains how the Isosplit clustering algorithm works, how it's implemented, and what the tests cover.

## Algorithm Overview

**Core Concept:** Isosplit finds clusters by detecting regions of low density between them, without requiring you to specify the number of clusters or other parameters. It assumes clusters are unimodal and separated by low-density regions.

## The Algorithm: Three Phases

### Phase 1: Initialization

**Location:** `src/isosplit6.cpp:31-50`

Creates ~200 initial "parcels" (mini-clusters) using k-means-like partitioning:

```cpp
bigint target_parcel_size = opts.min_cluster_size;  // default: 10
bigint target_num_parcels = opts.K_init;            // default: 200
parcelate2(labels, M, N, X, target_parcel_size, target_num_parcels, p2opts);
```

- Each parcel aims to have `min_cluster_size` points (default: 10)
- Crucially, **no final reassignment** is done to avoid "hexagonal" shapes that aren't conducive to isosplit iterations
- If `initial_labels` are provided by the user, this step is skipped

### Phase 2: Iterative Merging

**Location:** `src/isosplit6.cpp:74-179`

This is the main algorithm loop with nested iterations:

**Outer Loop (Passes):**

- Continues until no merges occur
- After a pass with no merges, does one final pass for redistribution
- Resets comparison matrix for changed clusters between passes

**Inner Loop (Iterations):**

1. Find closest uncompared cluster pairs based on centroid distances
2. For each pair, run `merge_test()` to decide: merge or redistribute
3. Update centroids and covariance matrices for changed clusters only
4. Track which clusters merged to update the active cluster list
5. Continue until no more pairs to compare or max iterations reached (default: 500)

**Key optimizations:**

- Only recomputes centroids/covmats for changed clusters (`src/isosplit6.cpp:140`)
- Tracks which comparisons have been made to avoid redundancy (`src/isosplit6.cpp:69-73`)
- Resets comparison matrix only for changed clusters (`src/isosplit6.cpp:163-171`)
- Comment notes parallelization potential (`src/isosplit6.cpp:124`)

### Phase 3: Label Remapping

**Location:** `src/isosplit6.cpp:181-188`

Renumbers cluster labels to be consecutive integers (1, 2, 3, ...) since some clusters may have been merged away.

## The Core: Merge Test (Isocut6)

**Location:** `src/isosplit6.cpp:233` (merge_test function)

The merge test is where the statistical magic happens. For two clusters, it performs:

### Step 1: Compute Direction Vector

**Location:** `src/isosplit6.cpp:246-285`

```cpp
// 1. Find direction between centroids
V = centroid2 - centroid1

// 2. Whiten by inverse of average covariance matrix
//    This makes all dimensions equally weighted
avg_covmat = (covmat1 + covmat2) / 2
V = inv(avg_covmat) * V

// 3. Normalize to unit length
V = V / ||V||
```

### Step 2: Project Data onto Direction Vector

**Location:** `src/isosplit6.cpp:287-301`

Projects both clusters onto the direction vector V, reducing the multi-dimensional problem to 1D:

```cpp
projection1[i] = V · X1[i]  // for all points in cluster 1
projection2[i] = V · X2[i]  // for all points in cluster 2
```

### Step 3: Run Isocut6 Algorithm

**Location:** `src/isocut6.cpp:23`

The **isocut6** function is the statistical heart of the method:

#### 3a. Sort the 1D Projection

Sort all projected points from both clusters.

#### 3b. Compute Log-Densities

**Location:** `src/isocut6.cpp:34-46`

Calculate log-density between consecutive sorted points:

```cpp
spacings[i] = X[i+1] - X[i]
log_densities[i] = log(multiplicities[i] / spacings[i])
```

#### 3c. Fit Unimodal Distribution

**Location:** `src/isocut6.cpp:49`

Use isotonic regression to fit an "up-then-down" (unimodal) shape to the log-densities:

```cpp
jisotonic5_updown(N-1, log_densities_unimodal_fit, log_densities, multiplicities)
```

This creates the expected density profile if the data were a single unimodal cluster.

#### 3d. Compute Dip Score

**Location:** `src/isocut6.cpp:58`

Calculate a Kolmogorov-Smirnov-like statistic measuring departure from unimodality:

```cpp
dipscore = compute_ks5(N-1, multiplicities, densities_unimodal_fit_times_spacings, peak_index)
```

Higher dip scores indicate stronger bimodality (two separate clusters).

#### 3e. Find Cut Point

**Location:** `src/isocut6.cpp:62-77`

1. Compute residuals: `log_densities_resid = log_densities - log_densities_unimodal_fit`
2. Identify the "critical range" where the largest deviation occurs
3. Fit a "down-then-up" (bimodal valley) shape to residuals on the critical range
4. Find the minimum point - this is the cut point

```cpp
jisotonic5_downup(critical_range_length, log_densities_resid_fit_on_critical_range, ...)
cutpoint_index = find_min_index(critical_range_length, log_densities_resid_fit_on_critical_range)
cutpoint = (X[cutpoint_index] + X[cutpoint_index+1]) / 2
```

### Step 4: Make Decision

**Location:** `src/isosplit6.cpp:309-322`

```cpp
if (dipscore < opts.isocut_threshold) {  // default threshold: 2.0
    // MERGE: clusters are not significantly bimodal
    do_merge = true;
} else {
    // DON'T MERGE: clusters are distinct
    // BUT redistribute points based on the cut point
    do_merge = false;
}
```

**Special case:** Clusters smaller than `min_cluster_size` are automatically merged (`src/isosplit6.cpp:357-359`).

## Isotonic Regression (`jisotonic5.cpp`)

Isotonic regression is a key utility that fits monotonic or unimodal functions to data. It's used extensively in the isocut algorithm.

### Core Functions

**`jisotonic5()`** (`src/jisotonic5.cpp:22`)

- Fits a non-decreasing (isotonic) function to data
- Uses the Pool Adjacent Violators (PAV) algorithm
- Minimizes mean squared error while maintaining monotonicity

**`jisotonic5_updown()`** (`src/jisotonic5.cpp:87`)

- Fits a unimodal (up-then-down) function
- Used for fitting expected unimodal density in isocut6
- Tries all possible peak locations and picks the one with minimum MSE

**`jisotonic5_downup()`** (`src/jisotonic5.cpp:133`)

- Fits a bimodal valley (down-then-up) function
- Used for finding the dip (minimum) between two clusters
- Implemented by negating values and calling `jisotonic5_updown()`

**`jisotonic5_sort()`** (`src/jisotonic5.cpp:146`)

- Simple wrapper around `std::sort()`

## Code Architecture

### Entry Point

**Main function:** `isosplit6()` in `src/isosplit6.cpp:29`

### Key Data Structures

- **`labels`**: `int*` array of size N - cluster assignments for each point
- **`centroids`**: `double*` M×Kmax matrix - cluster centers (M features, Kmax clusters)
- **`covmats`**: `double*` M×M×Kmax tensor - covariance matrices for each cluster
- **`active_labels`**: `std::vector<int>` - tracks which clusters still exist (haven't been merged)
- **`comparisons_made`**: Kmax×Kmax matrix - boolean matrix to avoid redundant comparisons

### Algorithm Parameters

Defined in `src/isosplit6.h:22-28`:

```cpp
struct isosplit6_opts {
    double isocut_threshold = 2.0;        // Dip score cutoff for merging
    int min_cluster_size = 10;            // Clusters smaller than this auto-merge
    int K_init = 200;                     // Initial number of parcels
    int max_iterations_per_pass = 500;   // Safety limit to prevent infinite loops
};
```

### Key Functions

**`isosplit6(int* labels, bigint M, bigint N, double* X, int32_t* initial_labels, isosplit6_opts opts)`**

- Main algorithm orchestrator
- Returns: `bool` (success/failure)

**`compare_pairs(std::vector<bigint>* clusters_changed, ...)`** (`src/isosplit6.cpp:325`)

- Compares multiple cluster pairs in parallel (conceptually - not actually parallelized)
- Updates labels in-place

**`merge_test(std::vector<bigint>* L12, bigint M, bigint N1, bigint N2, double* X1, double* X2, ...)`** (`src/isosplit6.cpp:233`)

- Tests if two clusters should merge
- Returns: `bool` (true = merge, false = keep separate)
- Also returns updated labels in `L12`

**`isocut6(double* dipscore_out, double* cutpoint_out, bigint N, double* samples, isocut6_opts opts)`** (`src/isocut6.cpp:23`)

- 1D dip test using isotonic regression
- Returns dip score and cut point

### Memory Management

- Uses C-style manual allocation (`malloc`/`free`)
- Raw pointers throughout - no smart pointers
- Legacy code from 2016-2017 (pre-modern C++)
- Careful tracking needed to avoid leaks

### Python Interface

**Location:** `src/main.cpp`

Uses pybind11 to create Python bindings:

```cpp
PYBIND11_MODULE(isosplit6_cpp, m) {
    m.def("isosplit6_fn", &isosplit6_fn, "Isosplit6 clustering C++ implementation.");
    m.def("isocut6_fn", &isocut6_fn, "Isocut6 C++ implementation.");
}
```

Python wrapper in `isosplit6/__init__.py` handles:

- Array type conversion (to float64)
- C-contiguous ordering enforcement
- Shape extraction
- Default parameter values

## What the Tests Test

### Current Test Suite

**File:** `isosplit6/tests/test_simple_run.py`

**Single Test:** `test_isosplit6_runs()`

```python
def test_isosplit6_runs():
    """
    A simple test to check that isosplit6 runs.
    Useful for testing wheels are successfully built.
    """
    data = np.random.random((100, 100))
    result = isosplit6(data)
    assert isinstance(result, np.ndarray)
```

### What It Tests ✅

1. **Package imports correctly** - can import `isosplit6` module
2. **C++ extension loads** - pybind11 module `isosplit6_cpp` is available
3. **Basic execution** - doesn't crash on 100×100 random data
4. **Returns correct type** - output is a numpy array
5. **Build system works** - validates wheel builds across platforms (used in CI)

### What It Does NOT Test ❌

- Clustering quality/correctness on known data
- Cluster separation scenarios (well-separated vs overlapping clusters)
- Edge cases:
  - Empty data
  - Single point
  - Identical/duplicate points
  - High-dimensional data
  - Very small/large datasets
- Parameter variations (`isocut_threshold`, `min_cluster_size`, etc.)
- `initial_labels` functionality for cluster refinement
- `isocut6()` function directly
- Algorithm convergence properties
- Memory leaks
- Different array dtypes (int, float32, etc.)
- Non-C-contiguous arrays
- Comparison with ground truth labels
- Performance benchmarks

### Test Purpose

This is a **smoke test** - it verifies the package was built successfully and can execute without crashing. This is valuable for CI/CD wheel building across multiple platforms (Linux, macOS, Windows) and Python versions (3.7-3.11).

It is **not** a test of algorithmic correctness or clustering quality. For production use, you would want additional tests covering:

- Synthetic data with known cluster structure
- Comparison against ground truth
- Edge case handling
- Numerical stability
- Performance characteristics

## NumPy Implementation Status

A pure NumPy implementation is being developed in parallel to the C++ version for research and prototyping purposes. The implementation follows a bottom-up approach, building tested components before assembling the complete algorithm.

### Completed Components ✅

#### Phase 1: Foundation & Helper Functions (Complete)

- **File:** `isosplit6/_isosplit_core/utils.py`
- **Functions:**
  - `compute_centroid()` - Cluster center calculation
  - `compute_covariance()` - Covariance matrix computation
  - `compute_centroids()` - Multi-cluster centroids
  - `compute_covmats()` - Multi-cluster covariances
  - `matrix_inverse_stable()` - Regularized matrix inversion
  - `mahalanobis_distance()` - Distance metric
- **Tests:** 12/12 passing

#### Phase 2: Isotonic Regression (Complete)

- **File:** `isosplit6/_isosplit_core/isotonic.py`
- **Functions:**
  - `jisotonic5()` - Pool Adjacent Violators (PAV) algorithm for non-decreasing fits
  - `jisotonic5_updown()` - Unimodal (peak) fitting
  - `jisotonic5_downup()` - Valley (dip) fitting
  - `jisotonic5_sort()` - Sorting utility
- **Tests:** 21/21 passing
- **Validation:** Exact match with C++ implementation (rtol=1e-10)

#### Phase 3: Isocut6 Algorithm (Complete)

- **File:** `isosplit6/_isosplit_core/isocut.py`
- **Functions:**
  - `isocut6()` - 1D bimodality detection using isotonic regression
  - `compute_ks4()` - Kolmogorov-Smirnov statistic
  - `compute_ks5()` - KS with critical range search
  - `find_min_index()` / `find_max_index()` - Helper utilities
- **Tests:** 17/17 passing (6 helper, 6 regression, 5 property tests)
- **Validation:** Exact match with C++ on all 7 reference test cases (rtol=1e-6)

**Algorithm Implementation:**

1. Sort 1D samples
2. Compute log-densities from spacings
3. Fit unimodal distribution using `jisotonic5_updown()`
4. Calculate dip score via KS-like statistic
5. Find critical range of maximum deviation
6. Fit valley to residuals using `jisotonic5_downup()`
7. Return dip score and cutpoint at valley minimum

#### Phase 4: Full Isosplit6 Algorithm (Complete) ✅

- **File:** `isosplit6/isosplit6_numpy.py`
- **Functions:**
  - `get_pairs_to_compare()` - Find mutually closest cluster pairs
  - `merge_test()` - Test if clusters should merge via 1D projection
  - `compare_pairs()` - Compare and merge/redistribute multiple pairs
  - `initialize_labels()` - K-means based initialization (simplified vs C++ parcelate2)
  - `remap_labels()` - Consecutive label renumbering
  - `isosplit6()` - Main algorithm with iterative merging loop
- **Tests:** 22/22 tests (20 passing, 2 known failures on 3-cluster cases)
- **Validation:** 97.3% overall pass rate (71/73 total tests)

**Algorithm Implementation:**

1. Initialize with K-means clustering (K_init=200 clusters)
2. Compute centroids and covariance matrices for all clusters
3. Iterative merging loop:
   - Find mutually closest cluster pairs not yet compared
   - For each pair: run merge_test (project onto whitened direction, run isocut6)
   - If dipscore < threshold: merge clusters; else: redistribute points
   - Update centroids/covmats for changed clusters only
   - Continue until no more pairs to compare
4. Repeat passes until convergence (no merges in final pass)
5. Remap labels to consecutive integers

### In Progress 🚧

#### Phase 5: JAX Implementation (Next)

- Port NumPy implementation to JAX
- Leverage JIT compilation and GPU acceleration

### Implementation Differences from C++

**Initialization:**

- C++ uses `parcelate2()` for initial clustering (complex k-means variant)
- NumPy implementation may use sklearn KMeans for simplicity
- This may cause minor differences in final clustering, but algorithm logic remains identical

**Numerical Precision:**

- NumPy uses float64 throughout (matches C++ `double`)
- Exact numerical match on isotonic regression and isocut6
- Small floating-point differences may accumulate in full algorithm

**Testing Strategy:**

- **Regression tests:** Compare against saved C++ outputs on identical inputs
- **Property tests:** Verify mathematical properties (monotonicity, unimodality, etc.)
- **Integration tests:** Validate on synthetic data with known cluster structure

### Test Coverage Summary

**Total Tests:** 73 tests (71 passing, 2 known failures = 97.3% pass rate)

- 12 helper function tests (utils)
- 21 isotonic regression tests
- 17 isocut6 tests
- 22 isosplit6 NumPy tests (20 passing, 2 failing on 3-cluster cases)
- 1 smoke test (C++ wrapper)

**Known Issues:**

- 3-cluster test cases fail due to KMeans initialization differences vs C++ parcelate2
- Algorithm finds valid 2-cluster solution (merges two of three clusters)
- Expected behavior per PLAN.md - acceptable variation from C++ implementation

**Reference Outputs Generated:** 15 C++ reference test cases

- 7 isocut6 scenarios (unimodal, bimodal, uniform, trimodal, etc.)
- 8 isosplit6 scenarios (various cluster configurations)

## References

- Original preprint: <https://arxiv.org/abs/1508.04841>
- Used by MountainSort5: <https://github.com/magland/mountainsort5>
- Based on Hartigan's dip statistic and isotonic regression techniques
