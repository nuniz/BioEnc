"""
Batch processing examples.

This example demonstrates efficient batch tokenization using shared buffers,
which is the recommended approach for processing multiple sequences in parallel.
"""

import numpy as np
import bioenc
import time

# Example 1: Basic batch processing
print("=" * 60)
print("Example 1: Batch processing with shared buffer")
print("=" * 60)

# Create multiple sequences
sequences = [
    b"ACGTACGTACGT",
    b"GGGGCCCCAAAA",
    b"ATATATATATAT",
    b"CGCGCGCGCGCG",
]

# Concatenate into shared buffer
buffer = np.frombuffer(b"".join(sequences), dtype=np.uint8)

# Create length array
lengths = np.array([len(s) for s in sequences], dtype=np.int64)

print(f"Number of sequences: {len(sequences)}")
print(f"Buffer size: {len(buffer)} bytes")
print(f"Lengths: {lengths}")
print()

# Batch tokenize
k = 4
max_len = 10  # Maximum tokens per sequence
tokens = bioenc.batch_tokenize_dna_shared(
    buffer, lengths,
    k=k, max_len=max_len, pad_value=-1,
    stride=1, alphabet="acgtn", strand="forward",
    enable_padding=True  # Return rectangular array
)

print(f"Output shape: {tokens.shape}")
print(f"Tokens:\n{tokens}")
print()

# Example 2: Canonical batch processing
print("=" * 60)
print("Example 2: Canonical batch processing")
print("=" * 60)

tokens_canonical = bioenc.batch_tokenize_dna_shared(
    buffer, lengths,
    k=k, max_len=max_len, pad_value=-1,
    stride=1, alphabet="acgtn", strand="canonical",
    enable_padding=True  # Return rectangular array
)

print(f"Canonical tokens:\n{tokens_canonical}")
print()

# Example 3: Both strands
print("=" * 60)
print("Example 3: Both strands (forward + revcomp)")
print("=" * 60)

tokens_fwd, tokens_rev = bioenc.batch_tokenize_dna_both(
    buffer, lengths,
    k=k, max_len=max_len, pad_value=-1,
    stride=1, alphabet="acgtn",
    enable_padding=True  # Return rectangular arrays
)

print(f"Forward tokens:\n{tokens_fwd}")
print(f"Revcomp tokens:\n{tokens_rev}")
print()

# Example 4: Performance comparison
print("=" * 60)
print("Example 4: Performance - Batch vs Sequential")
print("=" * 60)

# Create larger dataset
num_seqs = 1000
seq_len = 200
np.random.seed(42)

# Generate random sequences
bases = np.frombuffer(b"ACGT", dtype=np.uint8)
large_seqs = [bases[np.random.randint(0, 4, seq_len)] for _ in range(num_seqs)]

# Shared buffer approach
buffer_large = np.concatenate(large_seqs)
lengths_large = np.full(num_seqs, seq_len, dtype=np.int64)

# Batch processing (recommended)
start = time.time()
tokens_batch = bioenc.batch_tokenize_dna_shared(
    buffer_large, lengths_large,
    k=8, max_len=50, pad_value=-1,
    stride=1, alphabet="acgtn", strand="canonical",
    enable_padding=True  # Return rectangular array
)
batch_time = time.time() - start

# Sequential processing (for comparison)
start = time.time()
tokens_seq = []
for seq in large_seqs:
    t = bioenc.tokenize_dna(seq, k=8, stride=1, alphabet="acgtn", strand="canonical")
    # Pad or truncate to max_len=50
    if len(t) < 50:
        t = np.pad(t, (0, 50 - len(t)), constant_values=-1)
    else:
        t = t[:50]
    tokens_seq.append(t)
tokens_seq = np.array(tokens_seq)
seq_time = time.time() - start

print(f"Dataset: {num_seqs} sequences × {seq_len} bp")
print(f"Batch processing time:      {batch_time*1000:.2f} ms")
print(f"Sequential processing time: {seq_time*1000:.2f} ms")
print(f"Speedup: {seq_time/batch_time:.2f}x")
print("\nNote: Batch processing uses OpenMP parallelism and is much faster!")
print()

# Example 5: Hash reduction for large k
print("=" * 60)
print("Example 5: Hash reduction for large vocabulary")
print("=" * 60)

# For k=12, vocabulary size is 5^12 = 244,140,625
k_large = 12
vocab_size = 5 ** k_large
num_buckets = 10000

print(f"k={k_large}, vocabulary size: {vocab_size:,}")
print(f"Reducing to {num_buckets:,} buckets using hash_tokens()")

# Generate some tokens
small_buffer = np.frombuffer(b"ACGTACGT" * 20, dtype=np.uint8)
small_tokens = bioenc.tokenize_dna(small_buffer, k=k_large, stride=1)

# Hash to smaller vocabulary
hashed_tokens = bioenc.hash_tokens(small_tokens, num_buckets=num_buckets)

print(f"Original tokens: {small_tokens[:10]}")
print(f"Hashed tokens:   {hashed_tokens[:10]}")
print(f"All hashed tokens in [0, {num_buckets}): {np.all((hashed_tokens >= 0) & (hashed_tokens < num_buckets))}")
print()
