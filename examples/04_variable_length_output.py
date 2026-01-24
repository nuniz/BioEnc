"""
Demonstrate variable-length output and padding options.

This example shows:
1. Variable-length output (default behavior)
2. Rectangular output with auto-calculated padding
3. Rectangular output with explicit padding length
4. Using crop_and_tokenize for reading frame analysis
"""

import numpy as np
import bioenc

# ============================================================================
# Example 1: Variable-Length Output (Default)
# ============================================================================

print("=" * 70)
print("1. Variable-Length Output (Default)")
print("=" * 70)

# Create sequences of different lengths
sequences = [
    b"ACGTACGTACGT",      # 12 bp
    b"ACGTACGTACGTACGT",  # 16 bp
    b"ACGT",              # 4 bp
]

buffer = np.frombuffer(b"".join(sequences), dtype=np.uint8)
lengths = np.array([len(s) for s in sequences], dtype=np.int64)

# Default: Variable-length output (no padding)
tokens = bioenc.batch_tokenize_dna_shared(
    buffer, lengths, k=4, stride=1
)

print(f"\nType: {type(tokens)}")
print(f"Number of sequences: {len(tokens)}")
for i, t in enumerate(tokens):
    print(f"  Sequence {i}: {len(t)} tokens, shape={t.shape}")
    print(f"    Expected: {lengths[i] - 4 + 1} tokens")

# Verify each sequence has the right number of tokens
for i in range(len(sequences)):
    expected_tokens = max(0, lengths[i] - 4 + 1)
    assert len(tokens[i]) == expected_tokens, f"Seq {i}: expected {expected_tokens}, got {len(tokens[i])}"

print("\n✓ Variable-length output works correctly!")

# ============================================================================
# Example 2: Rectangular Output with Auto Max-Length
# ============================================================================

print("\n" + "=" * 70)
print("2. Rectangular Output with Auto Max-Length")
print("=" * 70)

tokens_padded = bioenc.batch_tokenize_dna_shared(
    buffer, lengths, k=4, stride=1,
    enable_padding=True  # Auto-calculate max_len
)

print(f"\nType: {type(tokens_padded)}")
print(f"Shape: {tokens_padded.shape}")
print(f"\nPadded array:")
print(tokens_padded)

# Check that max_len was auto-calculated correctly
expected_max = max(max(0, l - 4 + 1) for l in lengths)
assert tokens_padded.shape[1] == expected_max
print(f"\n✓ Auto-calculated max_len = {expected_max}")

# ============================================================================
# Example 3: Rectangular Output with Explicit Max-Length
# ============================================================================

print("\n" + "=" * 70)
print("3. Rectangular Output with Explicit Max-Length")
print("=" * 70)

tokens_padded_explicit = bioenc.batch_tokenize_dna_shared(
    buffer, lengths, k=4, stride=1,
    enable_padding=True,
    max_len=20,      # Force specific length
    pad_value=-99       # Custom padding value
)

print(f"\nType: {type(tokens_padded_explicit)}")
print(f"Shape: {tokens_padded_explicit.shape}")
print(f"\nFirst sequence (12 bp -> 9 tokens + 11 padding):")
print(tokens_padded_explicit[0])
print(f"  Valid tokens: {np.sum(tokens_padded_explicit[0] >= 0)}")
print(f"  Padding tokens: {np.sum(tokens_padded_explicit[0] == -99)}")

assert tokens_padded_explicit.shape == (3, 20)
print("\n✓ Explicit padding works correctly!")

# ============================================================================
# Example 4: Reading Frames using crop_and_tokenize
# ============================================================================

print("\n" + "=" * 70)
print("4. Reading Frames using crop_and_tokenize (Codon-like Analysis)")
print("=" * 70)

# Example: DNA sequence for codon analysis
codon_seq = b"ATGCGTAAATGATAG"  # 15 bp
print(f"\nOriginal sequence: {codon_seq.decode()}")
print(f"Length: {len(codon_seq)} bp")

# Extract all 3 reading frames

# Replicate buffer for 3 frames
buffer3 = np.tile(np.frombuffer(codon_seq, dtype=np.uint8), 3)
lengths3 = np.array([15, 15, 15], dtype=np.int64)

# Crop to start at different positions, length = 12 (multiple of 3)
crop_starts = np.array([0, 1, 2], dtype=np.int64)
crop_lengths = np.array([12, 12, 12], dtype=np.int64)

tokens_frames = bioenc.crop_and_tokenize_dna(
    buffer3, lengths3,
    crop_starts, crop_lengths,
    k=3, stride=3  # Non-overlapping triplets (codon-like)
)

print(f"\nNumber of reading frames: {len(tokens_frames)}")

for frame_idx, tokens_frame in enumerate(tokens_frames):
    print(f"\nReading frame {frame_idx}:")
    print(f"  Number of codons: {len(tokens_frame)}")

    # Show first few codons
    for i in range(min(4, len(tokens_frame))):
        pos = frame_idx + i * 3
        codon = codon_seq[pos:pos+3].decode()
        print(f"    Position {pos:2d}: {codon} → token {tokens_frame[i]:3d}")

print("\n✓ Reading frame extraction works correctly!")

# ============================================================================
# Example 5: Amino Acid Variable-Length Output
# ============================================================================

print("\n" + "=" * 70)
print("5. Amino Acid Variable-Length Output")
print("=" * 70)

aa_sequences = [
    b"MKTAYIAKQRQISFVK",  # 16 AA
    b"ACDEFGHIK",         # 9 AA
]

aa_buffer = np.frombuffer(b"".join(aa_sequences), dtype=np.uint8)
aa_lengths = np.array([len(s) for s in aa_sequences], dtype=np.int64)

# Variable-length output for amino acids
aa_tokens = bioenc.batch_tokenize_aa_shared(
    aa_buffer, aa_lengths, k=3, stride=1
)

print(f"\nType: {type(aa_tokens)}")
print(f"Number of sequences: {len(aa_tokens)}")
for i, t in enumerate(aa_tokens):
    expected = aa_lengths[i] - 3 + 1
    print(f"  Sequence {i}: {len(t)} tokens (expected {expected})")
    assert len(t) == expected

print("\n✓ Amino acid variable-length output works correctly!")

# ============================================================================
# Example 6: Both Strands Variable-Length
# ============================================================================

print("\n" + "=" * 70)
print("6. Both Strands Variable-Length Output")
print("=" * 70)

seqs_both = [b"ACGT", b"ACGTACGT", b"ACGTACGTACGT"]
buffer_both = np.frombuffer(b"".join(seqs_both), dtype=np.uint8)
lengths_both = np.array([len(s) for s in seqs_both], dtype=np.int64)

fwd, rev = bioenc.batch_tokenize_dna_both(
    buffer_both, lengths_both, k=2, stride=1
)

print(f"\nForward type: {type(fwd)}")
print(f"Reverse type: {type(rev)}")
print(f"\nSequence lengths:")
for i in range(len(seqs_both)):
    print(f"  Sequence {i}: fwd={len(fwd[i])}, rev={len(rev[i])}")
    assert len(fwd[i]) == len(rev[i])  # Should be same length
    assert len(fwd[i]) == lengths_both[i] - 2 + 1

print("\n✓ Both strands variable-length output works correctly!")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

print("""
bioenc provides flexible output options:

1. Variable-length (default): Returns List[np.ndarray] for memory efficiency
   - Use when sequences have different token counts
   - No padding needed
   - Easy to convert to PyTorch/TensorFlow DataLoader

2. Rectangular with padding: Returns 2D np.ndarray for batch operations
   - Use enable_padding=True
   - Optionally specify max_len for fixed size

3. crop_and_tokenize: Flexible windowing before tokenization
   - Reading frame extraction
   - Sliding windows
   - Data augmentation with random crops
""")
