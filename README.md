# bioenc

C++ library with Python bindings for tokenizing DNA and amino acid sequences into k-mer indices. Supports OpenMP parallelization and includes integration with PyTorch or TensorFlow.

## Installation

```bash
# Basic installation
pip install -e .

# With PyTorch
pip install -e ".[torch]"

# With TensorFlow
pip install -e ".[tensorflow]"

# With both
pip install -e ".[ml]"
```

**Requirements:** Python ≥3.9, NumPy ≥1.20, C++17 compiler, CMake ≥3.18, pybind11 ≥2.11

## Quick Start

### Single Sequence

```python
import numpy as np
import bioenc

# Tokenize DNA sequence (stride=3 default for codons)
seq = np.frombuffer(b"ATGCGTAAATGA", dtype=np.uint8)
tokens = bioenc.tokenize_dna(seq, k=3)
# Returns: [106, 137, 86, 206] - 4 non-overlapping codons

# For overlapping k-mers, set stride=1 explicitly
tokens = bioenc.tokenize_dna(seq, k=3, stride=1)
```

### Batch Processing

```python
# Concatenate sequences into shared buffer
sequences = [b"ATGCGTAAA", b"TTTGGGCCC", b"AAACCCGGGTTT"]
buffer = np.frombuffer(b"".join(sequences), dtype=np.uint8)
lengths = np.array([9, 9, 12], dtype=np.int64)

# Batch tokenize (offsets computed automatically)
tokens = bioenc.batch_tokenize_dna_shared(buffer, lengths, k=3)
# Returns: List of 3 arrays with lengths [3, 3, 4]

# With padding for rectangular output
tokens = bioenc.batch_tokenize_dna_shared(
    buffer, lengths, k=3,
    enable_padding=True,
    max_len=5,
    pad_value=-1
)
# Returns: (3, 5) array with padding
```

### Reading Frames

```python
seq = np.frombuffer(b"ATGCGTAAATGA", dtype=np.uint8)

# Extract specific reading frame
frame1 = bioenc.tokenize_dna(seq, k=3, reading_frame=1)

# Extract all 6 reading frames (3 forward + 3 reverse complement)
all_frames = bioenc.tokenize_dna_all_frames(seq, k=3)
# Returns: [fwd0, fwd1, fwd2, rev0, rev1, rev2]

# Batch all frames
all_frames = bioenc.batch_tokenize_dna_all_frames(buffer, lengths, k=3)
```

### PyTorch Integration

```python
import torch

# Convert to PyTorch (zero-copy when device='cpu')
tensor = bioenc.to_torch(tokens, device='cpu')

# Batch with automatic padding
tokens_list = bioenc.batch_tokenize_dna_shared(buffer, lengths, k=3)
padded, seq_lengths = bioenc.batch_to_torch(tokens_list, pad_value=-1, device='cuda')

# All frames to tensor [batch, 6, max_len]
all_frames = bioenc.batch_tokenize_dna_all_frames(buffer, lengths, k=3)
frames_tensor = bioenc.frames_to_torch(all_frames, device='cpu')
```

### TensorFlow Integration

```python
import tensorflow as tf

# Convert to TensorFlow
tensor = bioenc.to_tensorflow(tokens)

# Batch with padding
tokens_list = bioenc.batch_tokenize_dna_shared(buffer, lengths, k=3)
padded, seq_lengths = bioenc.batch_to_tensorflow(tokens_list, pad_value=-1)

# Use with tf.data.Dataset
def sequence_generator():
    for seq_bytes in sequences:
        seq = np.frombuffer(seq_bytes, dtype=np.uint8)
        yield bioenc.tokenize_dna(seq, k=3)

dataset = tf.data.Dataset.from_generator(
    sequence_generator,
    output_signature=tf.TensorSpec(shape=(None,), dtype=tf.int64)
)
```

## Benchmarks

Tested on Apple M3 Pro (11 cores):

| Operation | Time | Details |
|-----------|------|---------|
| Single sequence | 0.003 ms | 1000bp, k=6 |
| Batch forward | 0.30 ms | 100 × 1000bp, k=6 |
| Batch canonical | 0.39 ms | 100 × 1000bp, k=6 (with reverse complement) |
| All 6 frames | 0.06 ms | 10 × 1000bp, k=3 |

Performance scales with CPU cores via OpenMP parallelization.

## API Reference

### Single Sequence Functions

#### `tokenize_dna(seq, k, stride=3, reading_frame=None, alphabet="acgtn", strand="forward")`

Tokenize DNA sequence into k-mer indices.

**Parameters:**
- `seq` (ndarray[uint8]): DNA sequence as ASCII bytes
- `k` (int): K-mer size
- `stride` (int): Step between k-mers (default: 3 for codons)
- `reading_frame` (int | None): Extract frame 0, 1, or 2 (crops sequence before tokenization)
- `alphabet` (str): "acgtn" (base-6) or "iupac" (base-16)
- `strand` (str): "forward", "revcomp", or "canonical"

**Returns:** ndarray[int64] - K-mer token indices

#### `tokenize_aa(seq, k, stride=1)`

Tokenize amino acid sequence.

**Returns:** ndarray[int64] - K-mer token indices

#### `tokenize_dna_all_frames(seq, k=3, stride=3, alphabet="acgtn")`

Extract all 6 reading frames.

**Returns:** List of 6 arrays: [fwd0, fwd1, fwd2, rev0, rev1, rev2]

#### `reverse_complement_dna(seq, alphabet="acgtn")`

Compute reverse complement.

**Returns:** ndarray[uint8]

### Batch Functions

#### `batch_tokenize_dna_shared(buffer, lengths, k, stride=3, reading_frame=None, enable_padding=False, max_len=None, pad_value=0, alphabet="acgtn", strand="forward")`

Batch tokenize DNA sequences. Uses OpenMP parallelization.

**Parameters:**
- `buffer` (ndarray[uint8]): Concatenated sequences
- `lengths` (ndarray[int64]): Sequence lengths (offsets computed automatically)
- `enable_padding` (bool): Return rectangular array with padding
- `max_len` (int | None): Pad to this length (auto-computed if None)

**Returns:**
- List of arrays (variable length) if `enable_padding=False`
- 2D array (num_seqs, max_len) if `enable_padding=True`

#### `batch_tokenize_dna_both(buffer, lengths, k, stride=3, ...)`

Batch tokenize both forward and reverse complement (reverse complement tokens are returned in reverse-complement order).

**Returns:** Tuple of (forward, revcomp) - lists or 2D arrays depending on `enable_padding`

#### `batch_tokenize_aa_shared(buffer, lengths, k, stride=1, ...)`

Batch tokenize amino acid sequences (stride defaults to 1).

#### `batch_tokenize_dna_all_frames(buffer, lengths, k=3, stride=3, alphabet="acgtn", enable_padding=False, max_len=None, pad_value=0)`

Batch extract all 6 reading frames.

**Returns:**
- `List[List[ndarray]]` if `enable_padding=False` - outer: sequences, inner: 6 frames
- `List[ndarray]` if `enable_padding=True` - one array per sequence with shape (6, max_len)

#### `crop_and_tokenize_dna(buffer, lengths, crop_starts, crop_lengths, k, ...)`

Crop sequences to windows, then tokenize. Useful for sliding windows.

**Parameters:**
- `crop_starts` (ndarray[int64]): Start position for each crop (relative to sequence start)
- `crop_lengths` (ndarray[int64]): Length for each crop

#### `crop_and_tokenize_aa(buffer, lengths, crop_starts, crop_lengths, k, ...)`

Crop and tokenize amino acid sequences.

### ML Integration Functions

Require optional dependencies: `pip install bioenc[torch]` or `bioenc[tensorflow]`

#### PyTorch

- `to_torch(tokens, dtype=torch.long, device='cpu')` - Convert tokens to PyTorch (zero-copy on CPU)
- `batch_to_torch(tokens_list, pad_value=-1, dtype=torch.long, device='cpu')` - Batch with padding, returns (padded_tensor, lengths)
- `frames_to_torch(frames_list, pad_value=-1, dtype=torch.long, device='cpu')` - Convert all-frames to tensor [batch, 6, max_len]

#### TensorFlow

- `to_tensorflow(tokens, dtype=tf.int64)` - Convert tokens to TensorFlow
- `batch_to_tensorflow(tokens_list, pad_value=-1, dtype=tf.int64)` - Batch with padding, returns (padded_tensor, lengths)
- `frames_to_tensorflow(frames_list, pad_value=-1, dtype=tf.int64)` - Convert all-frames to tensor [batch, 6, max_len]

### Utilities

#### `vocab_size(k, alphabet="acgtn")`

Return the vocabulary size (`base**k`) for sizing embedding tables.

**Parameters:**
- `k` (int): K-mer size
- `alphabet` (str): "acgtn", "iupac", or "aa"

**Returns:** int

#### `get_vocab(k, alphabet="acgtn", include_unk=False)`

Return a mapping from k-mer strings to token indices. Always includes `<PAD>: 0`. Pass `include_unk=True` to add an `<UNK>` entry that aliases the all-unknown k-mer (`'N'*k` for DNA, `'X'*k` for AA).

**Parameters:**
- `k` (int): K-mer size
- `alphabet` (str): "acgtn", "iupac", or "aa"
- `include_unk` (bool): Add `<UNK>` alias (default: False)

**Returns:** dict[str, int]

#### `hash_tokens(tokens, num_buckets)`

Hash token indices to fixed vocabulary size. Uses MurmurHash3 finalizer.

**Parameters:**
- `tokens` (ndarray): Any shape
- `num_buckets` (int): Number of hash buckets

**Returns:** ndarray with same shape, values in [0, num_buckets). Negative values (padding) preserved.

## Alphabets

All alphabets reserve code 0 for PAD and code 1 for UNK, with character codes starting at 2.

### DNA ACGTN (base=6)

| Character | Code | | Character | Code |
|-----------|------|-|-----------|------|
| PAD       | 0    | | G, g      | 4    |
| UNK/N, other | 1 | | T, t, U, u | 5   |
| A, a      | 2    | |           |      |
| C, c      | 3    | |           |      |

K-mer index: `sum(code[i] × 6^(k-1-i))` for i in 0..k-1

Vocabulary size: 6^k (e.g., 46,656 for k=6)

### DNA IUPAC (base=16)

PAD=0, UNK/N=1, then A=2, C=3, G=4, T=5, R=6, Y=7, S=8, W=9, K=10, M=11, B=12, D=13, H=14, V=15

Vocabulary size: 16^k

### Amino Acids (base=29)

PAD=0, UNK/X=1, then 20 standard amino acids (A=2 through Y=21), plus B=22, Z=23, J=24, U=25, O=26, \*=27, -=28

Vocabulary size: 29^k

## Examples

See `examples/` directory:
- `01_basic_usage.py` - Single sequence tokenization
- `02_batch_processing.py` - Batch operations and performance
- `03_ml_pipeline.py` - ML integration patterns
- `04_variable_length_output.py` - Variable vs fixed-length output
- `05_reading_frames.py` - Reading frame extraction
- `06_pytorch_integration.py` - PyTorch Dataset/DataLoader
- `07_tensorflow_integration.py` - TensorFlow tf.data pipelines

## Testing

```bash
# Core tests
pytest tests/test_bioenc.py -v

# Input validation tests
pytest tests/test_validation.py -v

# ML integration tests (requires torch/tensorflow)
pytest tests/test_ml_utils.py -v

# Benchmarks
pytest tests/test_benchmark.py -v -s

# Run all examples
python examples/05_reading_frames.py
python examples/06_pytorch_integration.py  # Requires torch
python examples/07_tensorflow_integration.py  # Requires tensorflow
```

## License

MIT
