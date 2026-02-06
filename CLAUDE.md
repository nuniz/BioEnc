# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Test Commands

```bash
# Install in development mode (builds C++ extension via scikit-build-core + CMake)
pip install -e .

# Install with ML framework support
pip install -e ".[torch]"       # PyTorch
pip install -e ".[tensorflow]"  # TensorFlow
pip install -e ".[ml]"          # Both

# Run all unit tests
pytest tests/test_bioenc.py -v

# Run a single test
pytest tests/test_bioenc.py::TestTokenizeDna::test_basic_forward -v

# Run other test suites
pytest tests/test_validation.py -v   # Input validation tests
pytest tests/test_ml_utils.py -v     # ML integration tests (requires torch/tensorflow)
pytest tests/test_benchmark.py -v -s # Benchmark tests (shows C++ vs Python speedup)

# Test with specific OpenMP thread count
OMP_NUM_THREADS=4 pytest tests/test_benchmark.py -v -s
```

## Architecture

C++17 library with Python bindings via pybind11. Tokenizes biological sequences (DNA/amino acids) into integer k-mer indices for ML pipelines. Built with scikit-build-core + CMake; OpenMP is optional (gracefully degrades without it).

### Code Structure

- **`src/bioenc.hpp`** - Header-only C++ implementation containing:
  - Alphabet lookup tables (`DNA_ACGTN`, `DNA_IUPAC`, `AA_TABLE`, complement tables) as `constexpr uint8_t[256]`
  - Core tokenization functions with OpenMP `#pragma omp parallel for schedule(guided)` on batch loops
  - `RollingDnaCanon` template: O(1)-per-base rolling encoder for stride=1 canonical/revcomp modes
  - `PowerCache<32>`: precomputed power lookup table to avoid repeated `pow_u64` calls
  - k-mer index calculation: `index = sum(code[i] * base^(k-1-i))`

- **`src/bindings.cpp`** - pybind11 bindings that:
  - Wrap C++ functions for Python, computing offsets from lengths via cumsum internally
  - Release GIL with `py::gil_scoped_release` during batch operations for true parallelism
  - Validate inputs (k, stride, buffer bounds) before entering C++ core
  - Define the `_bioenc` native module

- **`bioenc/__init__.py`** - Python package that re-exports from `_bioenc` plus Python-side utilities:
  - `vocab_size(k, alphabet)` — returns `base**k` for embedding table sizing
  - `get_vocab(k, alphabet, include_unk=False)` — returns `{kmer_string: index}` dict with `<PAD>` entry; pass `include_unk=True` to add an `<UNK>` alias (points to all-N for DNA, all-X for AA)

- **`bioenc/ml_utils.py`** - Optional ML framework converters (`to_torch`, `batch_to_torch`, `frames_to_torch`, and TensorFlow equivalents). Uses lazy imports; functions raise `ImportError` if framework not installed.

### Key Design Patterns

1. **Shared buffer batch processing**: Sequences are concatenated into a single uint8 buffer with a lengths array; offsets are computed internally via cumsum. This enables zero-copy processing.
2. **GIL release + OpenMP**: Batch functions release Python's GIL then parallelize across sequences with OpenMP. For `all_frames`, parallelism is across sequences (not the 6 frames).
3. **Alphabet encoding**: Characters map to integer codes via 256-element lookup tables, then k-mers become base-N integers.
4. **Rolling window optimization**: stride=1 with canonical/revcomp/both uses `RollingDnaCanon` for O(1) per-base updates instead of recomputing k characters per position.

### Encoding Convention

All alphabets follow ML-standard encoding: **PAD=0, UNK=1**, characters start at 2.

### Vocabulary Sizes

- DNA ACGTN: `6^k` (PAD + UNK + A,C,G,T; e.g., 46,656 for k=6)
- DNA IUPAC: `16^k` (PAD + 15 IUPAC codes)
- Amino acids: `29^k` (PAD + UNK + 20 standard + B,Z,J,U,O,*,-)
