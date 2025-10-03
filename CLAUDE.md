# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Isosplit6 is a non-parametric clustering algorithm that uses Hartigan's dip statistic and isotonic regression. The package is a Python wrapper around a C++ implementation using pybind11. It's used by the MountainSort spike sorting algorithm.

## Architecture

### Hybrid Python/C++ Structure

The project consists of two main components:

1. **Python Interface** (`isosplit6/` directory):
   - `__init__.py`: Thin Python wrapper that exposes `isosplit6()` and `isocut6()` functions
   - Handles numpy array conversion and C-ordering requirements
   - All inputs are converted to float64 and C-contiguous arrays before passing to C++

2. **C++ Implementation** (`src/` directory):
   - `main.cpp`: pybind11 module definition that creates the `isosplit6_cpp` Python extension
   - `isosplit6.cpp/h`: Main clustering algorithm (version 6)
   - `isocut6.cpp/h`: Cut-point detection using dip statistic (version 6)
   - `isosplit5.cpp/h`: Legacy version 5 clustering algorithm
   - `isocut5.cpp/h`: Legacy version 5 cut-point detection
   - `jisotonic5.cpp/h`: Isotonic regression utilities
   - `ndarray.h`: Helper for numpy array interfacing

3. **NumPy/JAX Implementations** (in development):
   - `isosplit6_numpy.py`: Pure NumPy reimplementation
   - `isosplit6_jax.py`: JAX reimplementation with JIT support
   - `_isosplit_core/`: Shared utilities (isotonic regression, isocut, helpers)

### ⚠️ CRITICAL: C++ Reference Implementation

**DO NOT MODIFY THE C++ CODE IN `src/` DIRECTORY**

The C++ implementation in `src/` serves as the reference implementation and ground truth for all testing. It is used to:
- Generate reference outputs for validating NumPy/JAX implementations
- Provide the production-ready, performance-optimized version
- Serve as the authoritative specification of algorithm behavior

Any changes to the C++ code would:
- Invalidate all reference outputs in `isosplit6/tests/reference_outputs/`
- Break regression tests that verify NumPy/JAX match C++
- Potentially introduce bugs in the production implementation

If you need to fix bugs or make improvements:
1. First implement and test in NumPy/JAX versions
2. Document the issue and proposed fix
3. Consult with maintainers before modifying C++ code
4. If C++ must be changed, regenerate ALL reference outputs afterward

### Key Design Patterns

- The C++ code uses raw pointers and manual memory management
- pybind11 handles the Python/C++ boundary via `py::array_t` types
- `NDArray` template wrapper (`ndarray.h`) provides shape-aware access to numpy arrays
- Data layout: N observations × M features (row-major, C-contiguous)

### Algorithm Flow

1. **Initialization**: If no initial labels provided, uses `parcelate2()` to create ~200 initial clusters
2. **Iterative refinement**: Compares cluster pairs, merging based on isocut6 dip score threshold (default: 2.0)
3. **Convergence**: Continues until no merges occur or max iterations reached (default: 500 per pass)

## Development Commands

### Building and Installing

```bash
# Standard pip installation
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Using uv (faster alternative)
uv venv
source .venv/bin/activate  # On macOS/Linux (.venv\Scripts\activate on Windows)
uv pip install -e .
uv pip install -e ".[dev]"  # With dev dependencies

# Build wheels (used in CI)
python -m build
```

### M2 Silicon Mac Setup

On M2 Silicon Macs, you must set compiler variables before building. The default clang++ compiler will fail without C++14 standard flags.

```bash
# Set compiler variables (required for macOS M2 - do this first!)
export CC=gcc
export CXX=g++
export CXXFLAGS="-std=c++14 -I$(xcrun --show-sdk-path)/usr/include/c++/v1"
export LDFLAGS="-L$(xcrun --show-sdk-path)/usr/lib"

# Then install with pip
pip install -e ".[dev]"

# Or with uv
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Install mountainsort5 (if needed)
pip install mountainsort5
```

**Note:** The compiler variables must be exported in the same shell session before running the install command.

### Testing

```bash
# Run all tests
pytest

# Run specific test
pytest isosplit6/tests/test_simple_run.py

# Run tests with verbose output
pytest -v
```

### Linting

```bash
# Run ruff linter
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

### CI/CD

- Builds use `cibuildwheel` to create wheels for multiple Python versions (3.7-3.11) and platforms
- Deployment triggered on git tags
- C++11 standard required (`CFLAGS="-std=c++11"`)

## Important Configuration

- **Build system**: Uses setuptools with pybind11 integration
- **Python versions**: Supports 3.7-3.11
- **Platforms**: Linux, macOS (x86_64 + arm64), Windows
- **C++ standard**: C++11 (C++14 required for M2 Mac builds)
- **Dev dependencies**: Defined in `setup.cfg` under `[options.extras_require]` - includes numpy, pytest, ruff
- All C++ source files in `src/` are compiled into the `isosplit6_cpp` extension module

## Python Coding Standards

### Type Annotations

**ALL Python functions MUST have complete type annotations.**

Use type hints for:
- Function parameters
- Return values
- Instance variables (when not obvious)

Example:
```python
from typing import Optional, Tuple
import numpy as np

def compute_centroid(X: np.ndarray) -> np.ndarray:
    """Compute centroid of data points."""
    return np.mean(X, axis=0)

def jisotonic5(
    A: np.ndarray,
    W: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Isotonic regression using Pool Adjacent Violators."""
    # ... implementation
    return B, MSE
```

For numpy arrays, use `np.ndarray` (not `np.array` or `numpy.ndarray`).

### Docstrings

**ALL public functions and classes MUST have NumPy-style docstrings.**

NumPy docstring format:
```python
def function_name(param1: type1, param2: type2) -> return_type:
    """
    Short one-line summary.

    Longer description if needed. Can span multiple paragraphs.
    Explain what the function does, not how it does it.

    Parameters
    ----------
    param1 : type1
        Description of param1
    param2 : type2
        Description of param2

    Returns
    -------
    return_type
        Description of return value

    Examples
    --------
    >>> result = function_name(value1, value2)
    >>> print(result)
    expected_output

    Notes
    -----
    Additional information, implementation details, algorithm references.

    References
    ----------
    .. [1] Citation if applicable
    """
```

**Required sections:**
- Summary (one line)
- Parameters (if any)
- Returns (if not None)

**Optional but recommended sections:**
- Examples (especially for public API)
- Notes (for complex algorithms)
- References (for academic implementations)

**NOT Google-style or reStructuredText:**
```python
# ❌ WRONG - Don't use Google style
def bad_example(x, y):
    """
    Short description.

    Args:
        x: Description
        y: Description

    Returns:
        Description
    """
```

See existing implementations in `isosplit6/_isosplit_core/isotonic.py` for examples.

### Code Style

- Follow PEP 8
- Use `ruff` for linting and formatting
- Maximum line length: 100 characters
- Use descriptive variable names
- Sort imports: standard library, third-party, local (enforced by ruff)

## API Usage

```python
from isosplit6 import isosplit6

# X: numpy array of shape (N observations, M features)
cluster_labels = isosplit6(X)

# With initial labels for refinement
cluster_labels = isosplit6(X, initial_labels=initial_labels)

# Lower-level: get dip score and cut point
from isosplit6 import isocut6
dipscore, cutpoint = isocut6(samples_1d)
```

## Common Gotchas

- Input arrays must be numpy arrays; will be converted to float64 if needed
- C-contiguous array ordering is essential (enforced in Python wrapper)
- Initial labels array must be int32 type
- The algorithm assumes clusters are unimodal and separated by low-density regions
