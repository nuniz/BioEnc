"""
Reading frame extraction examples.

This example demonstrates the new reading frame features for codon-level analysis:
- Default stride=3 for non-overlapping codons
- reading_frame parameter for specific frame extraction
- tokenize_dna_all_frames() for 6-frame analysis
- batch_tokenize_dna_all_frames() for batch processing
"""

import numpy as np
import bioenc

# Example 1: Default stride=3 for codons
print("=" * 60)
print("Example 1: Default stride=3 for non-overlapping codons")
print("=" * 60)

sequence = np.frombuffer(b"ATGCGTAAATGATAG", dtype=np.uint8)
tokens = bioenc.tokenize_dna(sequence, k=3)  # stride=3 is now default

print(f"Sequence: ATGCGTAAATGATAG (15 bases)")
print(f"K-mer size: 3 (codons)")
print(f"Stride: 3 (default - non-overlapping)")
print(f"Tokens: {tokens}")
print(f"Number of tokens: {len(tokens)} (5 complete codons)")
print()

# Example 2: Extract specific reading frame
print("=" * 60)
print("Example 2: Extracting specific reading frames")
print("=" * 60)

sequence = np.frombuffer(b"ATGCGTAAATGATAG", dtype=np.uint8)

frame0 = bioenc.tokenize_dna(sequence, k=3, reading_frame=0)  # Start at position 0
frame1 = bioenc.tokenize_dna(sequence, k=3, reading_frame=1)  # Start at position 1
frame2 = bioenc.tokenize_dna(sequence, k=3, reading_frame=2)  # Start at position 2

print(f"Sequence: ATGCGTAAATGATAG (15 bases)")
print(f"\nFrame 0 (starts at pos 0): ATGCGTAAATGATAG")
print(f"  Codons: {sequence[0:3].tobytes().decode('ascii')}, "
      f"{sequence[3:6].tobytes().decode('ascii')}, "
      f"{sequence[6:9].tobytes().decode('ascii')}, ...")
print(f"  Tokens: {frame0} ({len(frame0)} codons)")

print(f"\nFrame 1 (starts at pos 1):  TGCGTAAATGATAG")
print(f"  Codons: {sequence[1:4].tobytes().decode('ascii')}, "
      f"{sequence[4:7].tobytes().decode('ascii')}, "
      f"{sequence[7:10].tobytes().decode('ascii')}, ...")
print(f"  Tokens: {frame1} ({len(frame1)} codons)")

print(f"\nFrame 2 (starts at pos 2):   GCGTAAATGATAG")
print(f"  Codons: {sequence[2:5].tobytes().decode('ascii')}, "
      f"{sequence[5:8].tobytes().decode('ascii')}, "
      f"{sequence[8:11].tobytes().decode('ascii')}, ...")
print(f"  Tokens: {frame2} ({len(frame2)} codons)")
print()

# Example 3: Extract all 6 reading frames (3 forward + 3 reverse complement)
print("=" * 60)
print("Example 3: All 6 reading frames (forward + reverse)")
print("=" * 60)

sequence = np.frombuffer(b"ATGCGTAAATGA", dtype=np.uint8)
all_frames = bioenc.tokenize_dna_all_frames(sequence, k=3)

print(f"Sequence: ATGCGTAAATGA (12 bases)")
print(f"\nForward frames:")
print(f"  Frame 0: {all_frames[0]} ({len(all_frames[0])} codons)")
print(f"  Frame 1: {all_frames[1]} ({len(all_frames[1])} codons)")
print(f"  Frame 2: {all_frames[2]} ({len(all_frames[2])} codons)")

print(f"\nReverse complement frames:")
print(f"  Frame 0: {all_frames[3]} ({len(all_frames[3])} codons)")
print(f"  Frame 1: {all_frames[4]} ({len(all_frames[4])} codons)")
print(f"  Frame 2: {all_frames[5]} ({len(all_frames[5])} codons)")

print(f"\nTotal frames returned: {len(all_frames)}")
print()

# Example 4: Batch processing all reading frames
print("=" * 60)
print("Example 4: Batch processing with all reading frames")
print("=" * 60)

sequences = [
    b"ATGCGTAAATGA",  # 12 bases
    b"TTTGGGCCC",     # 9 bases
    b"AAACCCGGGTTT",  # 12 bases
]

# Prepare batch format
buffer = np.frombuffer(b"".join(sequences), dtype=np.uint8)
lengths = np.array([len(s) for s in sequences], dtype=np.int64)

# Extract all 6 frames for each sequence
all_frames_batch = bioenc.batch_tokenize_dna_all_frames(
    buffer, lengths, k=3
)

print(f"Number of sequences: {len(all_frames_batch)}")
print(f"Frames per sequence: {len(all_frames_batch[0])}")

for i, seq in enumerate(sequences):
    print(f"\nSequence {i+1}: {seq.decode('ascii')} ({len(seq)} bases)")
    print(f"  Forward frames: {len(all_frames_batch[i][0])}, "
          f"{len(all_frames_batch[i][1])}, {len(all_frames_batch[i][2])} codons")
    print(f"  Reverse frames: {len(all_frames_batch[i][3])}, "
          f"{len(all_frames_batch[i][4])}, {len(all_frames_batch[i][5])} codons")
print()

# Example 5: All frames with padding (rectangular output)
print("=" * 60)
print("Example 5: All frames with padding for ML models")
print("=" * 60)

sequences = [
    b"ATGCGTAAATGA",  # 12 bases -> 4 codons in frame 0
    b"TTTGGGCCC",     # 9 bases -> 3 codons in frame 0
]

buffer = np.frombuffer(b"".join(sequences), dtype=np.uint8)
lengths = np.array([len(s) for s in sequences], dtype=np.int64)

# Get padded 2D arrays
all_frames_padded = bioenc.batch_tokenize_dna_all_frames(
    buffer, lengths, k=3,
    enable_padding=True,  # Return 2D arrays
    pad_value=-1
)

print(f"Number of sequences: {len(all_frames_padded)}")
print(f"\nSequence 1 output shape: {all_frames_padded[0].shape}")
print(f"  (6 frames x max_length)")
print(f"  Frame 0: {all_frames_padded[0][0]}")
print(f"  Frame 1: {all_frames_padded[0][1]}")

print(f"\nSequence 2 output shape: {all_frames_padded[1].shape}")
print(f"  Frame 0: {all_frames_padded[1][0]}")
print(f"  (Note: Shorter sequence is padded with -1)")
print()

# Example 6: Overlapping k-mers (stride=1) for comparison
print("=" * 60)
print("Example 6: Overlapping vs non-overlapping (stride comparison)")
print("=" * 60)

sequence = np.frombuffer(b"ATGCGTAAA", dtype=np.uint8)

# New default: stride=3 (non-overlapping codons)
tokens_stride3 = bioenc.tokenize_dna(sequence, k=3)  # stride=3 default

# Old behavior: stride=1 (overlapping k-mers)
tokens_stride1 = bioenc.tokenize_dna(sequence, k=3, stride=1)

print(f"Sequence: ATGCGTAAA (9 bases)")
print(f"\nStride=3 (non-overlapping codons):")
print(f"  Tokens: {tokens_stride3}")
print(f"  Count: {len(tokens_stride3)} tokens")
print(f"  Codons: ATG, CGT, AAA")

print(f"\nStride=1 (overlapping k-mers, explicit parameter):")
print(f"  Tokens: {tokens_stride1}")
print(f"  Count: {len(tokens_stride1)} tokens")
print(f"  K-mers: ATG, TGC, GCG, CGT, GTA, TAA, AAA")

print(f"\nNote: For biological codon analysis, use stride=3 (default)")
print(f"      For general k-mer analysis, explicitly set stride=1")
print()

print("=" * 60)
print("Summary Reading Frame Features")
print("=" * 60)
print("1. stride=3 is now the default (breaking change)")
print("2. reading_frame parameter extracts specific frames (0, 1, or 2)")
print("3. tokenize_dna_all_frames() extracts all 6 frames at once")
print("4. batch_tokenize_dna_all_frames() for efficient batch processing")
print("5. Padding support for ML-ready rectangular outputs")
print("6. Optimized with single reverse complement computation")
