"""
bioenc: Fast k-mer tokenization for bioinformatics ML applications.

This library provides zero-copy tokenization of DNA and amino acid sequences
for use in AI/ML pipelines. Uses OpenMP for parallel batch processing.
"""

import itertools

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
    # Vocabulary utilities
    "vocab_size",
    "get_vocab",
    # ML utilities
    "to_torch",
    "batch_to_torch",
    "frames_to_torch",
    "to_tensorflow",
    "batch_to_tensorflow",
    "frames_to_tensorflow",
]

__version__ = "0.3.1"

# ---------------------------------------------------------------------------
# Vocabulary utilities
# ---------------------------------------------------------------------------

_ALPHABET_INFO = {
    'acgtn': (6, {1: 'N', 2: 'A', 3: 'C', 4: 'G', 5: 'T'}),
    'iupac': (16, {1: 'N', 2: 'A', 3: 'C', 4: 'G', 5: 'T', 6: 'R', 7: 'Y',
                    8: 'S', 9: 'W', 10: 'K', 11: 'M', 12: 'B', 13: 'D',
                    14: 'H', 15: 'V'}),
    'aa':    (29, {1: 'X', 2: 'A', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G',
                   8: 'H', 9: 'I', 10: 'K', 11: 'L', 12: 'M', 13: 'N',
                   14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'V',
                   20: 'W', 21: 'Y', 22: 'B', 23: 'Z', 24: 'J', 25: 'U',
                   26: 'O', 27: '*', 28: '-'}),
}

_MAX_VOCAB_ENTRIES = 5_000_000


def vocab_size(k, alphabet='acgtn'):
    """Return the vocabulary size (``base**k``) for an embedding table.

    Parameters
    ----------
    k : int
        k-mer length.
    alphabet : str
        One of ``'acgtn'``, ``'iupac'``, or ``'aa'``.

    Returns
    -------
    int
    """
    alphabet = alphabet.lower()
    if alphabet not in _ALPHABET_INFO:
        raise ValueError(
            f"Unknown alphabet {alphabet!r}. "
            f"Choose from {sorted(_ALPHABET_INFO)}"
        )
    base, _ = _ALPHABET_INFO[alphabet]
    return base ** k


def get_vocab(k, alphabet='acgtn', include_unk=False):
    """Return a mapping from k-mer strings to token indices.

    The returned dict contains ``'<PAD>': 0`` plus all ``(base-1)**k``
    character k-mers. When *include_unk* is True, an additional
    ``'<UNK>'`` entry is added that aliases the all-unknown k-mer
    (e.g. ``'NNN'`` for DNA k=3, ``'XXX'`` for AA k=3).

    Indices are computed with the same formula as the C++ tokenizer:
    ``index = sum(code[i] * base**(k-1-i))``.

    Parameters
    ----------
    k : int
        k-mer length.
    alphabet : str
        One of ``'acgtn'``, ``'iupac'``, or ``'aa'``.
    include_unk : bool
        If True, add ``'<UNK>'`` entry (default: False).

    Returns
    -------
    dict[str, int]
    """
    alphabet = alphabet.lower()
    if alphabet not in _ALPHABET_INFO:
        raise ValueError(
            f"Unknown alphabet {alphabet!r}. "
            f"Choose from {sorted(_ALPHABET_INFO)}"
        )
    base, code_to_char = _ALPHABET_INFO[alphabet]
    total = base ** k
    if total > _MAX_VOCAB_ENTRIES:
        raise ValueError(
            f"Vocabulary too large ({total:,} entries) for alphabet="
            f"{alphabet!r}, k={k}. Maximum is {_MAX_VOCAB_ENTRIES:,}."
        )

    # Characters ordered by code (1, 2, …, base-1)
    codes = sorted(code_to_char)
    chars = [code_to_char[c] for c in codes]

    vocab = {'<PAD>': 0}

    for combo in itertools.product(codes, repeat=k):
        index = 0
        for i, code in enumerate(combo):
            index += code * (base ** (k - 1 - i))
        kmer = ''.join(code_to_char[c] for c in combo)
        vocab[kmer] = index

    if include_unk:
        # <UNK> aliases the all-UNK k-mer (code 1 repeated k times)
        unk_index = sum(1 * (base ** (k - 1 - i)) for i in range(k))
        vocab['<UNK>'] = unk_index

    return vocab
