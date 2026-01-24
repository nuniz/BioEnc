"""
bioenc: Fast k-mer tokenization for bioinformatics ML applications.

This library provides zero-copy tokenization of DNA and amino acid sequences
for use in AI/ML pipelines. Uses OpenMP for parallel batch processing.
"""

from ._bioenc import (
    tokenize_dna,
    tokenize_dna_all_frames,
    batch_tokenize_dna_all_frames,
    tokenize_aa,
    reverse_complement_dna,
    batch_tokenize_dna_shared,
    batch_tokenize_dna_both,
    batch_tokenize_aa_shared,
    crop_and_tokenize_dna,
    crop_and_tokenize_aa,
    hash_tokens,
)

# ML utilities (optional dependencies)
from .ml_utils import (
    to_torch,
    batch_to_torch,
    frames_to_torch,
    to_tensorflow,
    batch_to_tensorflow,
    frames_to_tensorflow,
)

__all__ = [
    # Core tokenization
    "tokenize_dna",
    "tokenize_dna_all_frames",
    "batch_tokenize_dna_all_frames",
    "tokenize_aa",
    "reverse_complement_dna",
    "batch_tokenize_dna_shared",
    "batch_tokenize_dna_both",
    "batch_tokenize_aa_shared",
    "crop_and_tokenize_dna",
    "crop_and_tokenize_aa",
    "hash_tokens",
    # ML utilities
    "to_torch",
    "batch_to_torch",
    "frames_to_torch",
    "to_tensorflow",
    "batch_to_tensorflow",
    "frames_to_tensorflow",
]

__version__ = "0.1.0"
