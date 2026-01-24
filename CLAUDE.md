# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Test Commands

```bash
# Install in development mode (builds C++ extension)
pip install -e .

# Run all unit tests
pytest tests/test_bioenc.py -v

# Run a single test
pytest tests/test_bioenc.py::TestTokenizeDna::test_basic_forward -v

# Run benchmark tests (shows C++ vs Python speedup)
pytest tests/test_benchmark.py -v -s

# Test with specific OpenMP thread count
OMP_NUM_THREADS=4 pytest tests/test_benchmark.py -v -s
```

## Architecture

This is a C++17 library with Python bindings via pybind11. It tokenizes biological sequences (DNA/amino acids) into integer k-mer indices for ML pipelines.

### Code Structure

- **`src/bioenc.hpp`** - Header-only C++ implementation containing:
  - Alphabet lookup tables (`DNA_ACGTN`, `DNA_IUPAC`, `AA_TABLE`, complement tables)
  - Core tokenization functions with OpenMP `#pragma omp parallel for` on batch loops
  - k-mer index calculation: `index = sum(code[i] * base^(k-1-i))`

- **`src/bindings.cpp`** - pybind11 bindings that:
  - Wrap C++ functions for Python
  - Release GIL with `py::gil_scoped_release` during batch operations for true parallelism
  - Define the `_bioenc` native module

- **`bioenc/__init__.py`** - Python package that re-exports functions from `_bioenc`

### Key Design Patterns

1. **Shared buffer batch processing**: Sequences are concatenated into a single uint8 buffer with separate offset/length arrays, enabling zero-copy processing
2. **GIL release + OpenMP**: Batch functions release Python's GIL then parallelize across sequences with OpenMP
3. **Alphabet encoding**: Characters map to integer codes via 256-element lookup tables, then k-mers become base-N integers

### Vocabulary Sizes

- DNA ACGTN: `5^k` (e.g., 15,625 for k=6)
- DNA IUPAC: `15^k`
- Amino acids: `28^k`
