"""
Basic usage examples for bioenc k-mer tokenization.

This example demonstrates single sequence tokenization with different parameters.
"""

import numpy as np
import bioenc

# Example 1: Simple forward tokenization
print("=" * 60)
print("Example 1: Forward tokenization")
print("=" * 60)

sequence = np.frombuffer(b"ACGTACGTACGT", dtype=np.uint8)
tokens = bioenc.tokenize_dna(sequence, k=3, stride=1, alphabet="acgtn", strand="forward")

print(f"Sequence: ACGTACGTACGT")
print(f"K-mer size: 3")
print(f"Tokens: {tokens}")
print(f"Number of tokens: {len(tokens)}")
print()

# Example 2: Canonical tokenization (strand-invariant)
print("=" * 60)
print("Example 2: Canonical tokenization")
print("=" * 60)

tokens_canonical = bioenc.tokenize_dna(sequence, k=3, stride=1, alphabet="acgtn", strand="canonical")

print(f"Sequence: ACGTACGTACGT")
print(f"Canonical tokens: {tokens_canonical}")
print(f"Note: Canonical tokens are min(forward, revcomp), making them strand-invariant")
print()

# Example 3: Reverse complement
print("=" * 60)
print("Example 3: Reverse complement")
print("=" * 60)

revcomp = bioenc.reverse_complement_dna(sequence, alphabet="acgtn")

print(f"Original:  {sequence.tobytes().decode('ascii')}")
print(f"Rev-comp:  {revcomp.tobytes().decode('ascii')}")
print()

# Example 4: Different stride values
print("=" * 60)
print("Example 4: Stride parameter")
print("=" * 60)

long_seq = np.frombuffer(b"ACGTACGTACGTACGTACGT", dtype=np.uint8)

tokens_stride1 = bioenc.tokenize_dna(long_seq, k=4, stride=1)
tokens_stride2 = bioenc.tokenize_dna(long_seq, k=4, stride=2)
tokens_stride3 = bioenc.tokenize_dna(long_seq, k=4, stride=3)

print(f"Sequence length: {len(long_seq)}")
print(f"K-mer size: 4")
print(f"Stride=1: {len(tokens_stride1)} tokens")
print(f"Stride=2: {len(tokens_stride2)} tokens")
print(f"Stride=3: {len(tokens_stride3)} tokens")
print()

# Example 5: IUPAC alphabet (supports ambiguity codes)
print("=" * 60)
print("Example 5: IUPAC alphabet")
print("=" * 60)

# Sequence with ambiguity code 'N'
ambig_seq = np.frombuffer(b"ACGTNACGT", dtype=np.uint8)

tokens_acgtn = bioenc.tokenize_dna(ambig_seq, k=3, alphabet="acgtn")
tokens_iupac = bioenc.tokenize_dna(ambig_seq, k=3, alphabet="iupac")

print(f"Sequence: ACGTNACGT (contains N)")
print(f"ACGTN tokens (base-5): {tokens_acgtn}")
print(f"IUPAC tokens (base-15): {tokens_iupac}")
print()

# Example 6: Vocabulary size
print("=" * 60)
print("Example 6: Vocabulary size for different k and alphabets")
print("=" * 60)

for k in [3, 6, 8, 10]:
    vocab_acgtn = 5 ** k
    vocab_iupac = 15 ** k
    print(f"k={k:2d}: ACGTN vocab={vocab_acgtn:>12,}  IUPAC vocab={vocab_iupac:>15,}")

print("\nNote: For large k values, use hash_tokens() to reduce vocabulary size")
print()
