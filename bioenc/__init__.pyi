"""Type stubs for bioenc package."""

from typing import List, Literal, Optional, Tuple, Union
import numpy as np
import numpy.typing as npt

AlphabetType = Literal["acgtn", "iupac"]
StrandType = Literal["forward", "revcomp", "canonical", "both"]

def tokenize_dna(
    seq: npt.NDArray[np.uint8],
    k: int,
    stride: int = 3,
    reading_frame: Optional[int] = None,
    alphabet: AlphabetType = "acgtn",
    strand: StrandType = "forward",
) -> npt.NDArray[np.int64]:
    """
    Tokenize a single DNA sequence into k-mer indices.

    Parameters
    ----------
    seq : np.ndarray[np.uint8]
        Input sequence as uint8 array (ASCII bytes)
    k : int
        K-mer size (must be > 0 and <= 31)
    stride : int, optional
        Step size between k-mers (default: 3 for non-overlapping codons)
    reading_frame : int, optional
        If specified, crop sequence to start at this reading frame (0, 1, or 2).
        For example, reading_frame=1 starts at position 1 (skips first base).
        Default: None (start at position 0)
    alphabet : str, optional
        'acgtn' (base=5) or 'iupac' (base=15) (default: 'acgtn')
    strand : str, optional
        'forward', 'revcomp', or 'canonical' (default: 'forward')

    Returns
    -------
    np.ndarray[np.int64]
        Array of k-mer indices

    Raises
    ------
    RuntimeError
        If k <= 0, k > 31, stride <= 0, or reading_frame not in {0, 1, 2, None}
    """
    ...

def tokenize_dna_all_frames(
    seq: npt.NDArray[np.uint8],
    k: int = 3,
    stride: int = 3,
    alphabet: AlphabetType = "acgtn",
) -> List[npt.NDArray[np.int64]]:
    """
    Extract all 6 reading frames (3 forward + 3 reverse complement).

    Computes reverse complement once for efficiency. This function
    is optimized for single sequences (no OpenMP overhead).

    Parameters
    ----------
    seq : np.ndarray[np.uint8]
        Input sequence as uint8 array (ASCII bytes)
    k : int, optional
        K-mer size (default: 3 for codons)
    stride : int, optional
        Step size between k-mers (default: 3 for non-overlapping)
    alphabet : str, optional
        'acgtn' (base=5) or 'iupac' (base=15) (default: 'acgtn')

    Returns
    -------
    List[np.ndarray[np.int64]]
        List of 6 arrays: [fwd_frame0, fwd_frame1, fwd_frame2,
                           rev_frame0, rev_frame1, rev_frame2]
        Each array contains int64 k-mer indices for that frame.

    Raises
    ------
    RuntimeError
        If k <= 0, k > 31, or stride <= 0

    Examples
    --------
    >>> seq = np.frombuffer(b"ATGCGTAAATGA", dtype=np.uint8)
    >>> frames = bioenc.tokenize_dna_all_frames(seq, k=3)
    >>> len(frames)  # 6 frames
    6
    >>> frames[0]  # Forward frame 0: ATG, CGT, AAA, TGA
    array([...])
    >>> frames[3]  # Reverse complement frame 0
    array([...])
    """
    ...

def batch_tokenize_dna_all_frames(
    buffer: npt.NDArray[np.uint8],
    lengths: npt.NDArray[np.int64],
    k: int = 3,
    stride: int = 3,
    alphabet: AlphabetType = "acgtn",
    enable_padding: bool = False,
    max_len: Optional[int] = None,
    pad_value: int = -1,
) -> Union[List[List[npt.NDArray[np.int64]]], List[npt.NDArray[np.int64]]]:
    """
    Batch tokenize all 6 reading frames for multiple sequences.

    Uses OpenMP for parallel processing across sequences (not frames).
    Releases the Python GIL during computation to allow true parallelism.

    Each thread independently computes reverse complement (no shared state).
    Optimized with schedule(guided) for load balancing with variable lengths.

    Parameters
    ----------
    buffer : np.ndarray[np.uint8]
        Concatenated sequences as uint8 array
    lengths : np.ndarray[np.int64]
        Length of each sequence
    k : int, optional
        K-mer size (default: 3 for codons)
    stride : int, optional
        Step size between k-mers (default: 3 for non-overlapping)
    alphabet : str, optional
        'acgtn' or 'iupac' (default: 'acgtn')
    enable_padding : bool, optional
        If True, return padded 2D arrays for each sequence (default: False)
    max_len : int, optional
        If enable_padding=True and this is set, pad to this length.
        If None, pad to actual maximum token count across all frames.
    pad_value : int, optional
        Padding value for short sequences (default: -1)

    Returns
    -------
    List[List[np.ndarray[np.int64]]] or List[np.ndarray[np.int64]]
        If enable_padding=False:
            List of lists: result[seq_idx][frame_idx] = np.ndarray
            Outer list = sequences, inner list = 6 frames per sequence
        If enable_padding=True:
            List of 2D arrays: result[seq_idx] = np.ndarray[6, max_len]

    Raises
    ------
    RuntimeError
        If k <= 0, k > 31, stride <= 0, or buffer access is out of bounds

    Examples
    --------
    >>> sequences = [b"ATGCGTAAA", b"TTTGGGCCC"]
    >>> buffer = np.frombuffer(b"".join(sequences), dtype=np.uint8)
    >>> lengths = np.array([9, 9], dtype=np.int64)
    >>> all_frames = bioenc.batch_tokenize_dna_all_frames(
    ...     buffer, lengths, k=3
    ... )
    >>> len(all_frames)  # 2 sequences
    2
    >>> len(all_frames[0])  # 6 frames
    6
    """
    ...

def reverse_complement_dna(
    seq: npt.NDArray[np.uint8],
    alphabet: AlphabetType = "acgtn",
) -> npt.NDArray[np.uint8]:
    """
    Compute reverse complement of a DNA sequence.

    Parameters
    ----------
    seq : np.ndarray[np.uint8]
        Input sequence as uint8 array (ASCII bytes)
    alphabet : str, optional
        'acgtn' or 'iupac' (default: 'acgtn')

    Returns
    -------
    np.ndarray[np.uint8]
        Reverse complement as uint8 array
    """
    ...

def tokenize_aa(
    seq: npt.NDArray[np.uint8],
    k: int,
    stride: int = 1,
) -> npt.NDArray[np.int64]:
    """
    Tokenize a single amino acid sequence into k-mer indices.

    Vocabulary size: 28^k (20 standard AA + 6 ambiguous + stop + gap)

    Parameters
    ----------
    seq : np.ndarray[np.uint8]
        Input sequence as uint8 array (ASCII bytes)
    k : int
        K-mer size (must be > 0 and <= 31)
    stride : int, optional
        Step size between k-mers (default: 1)

    Returns
    -------
    np.ndarray[np.int64]
        Array of k-mer indices (variable length)

    Raises
    ------
    RuntimeError
        If k <= 0, k > 31, or stride <= 0

    Examples
    --------
    >>> seq = np.frombuffer(b"ACDEFGHIK", dtype=np.uint8)
    >>> tokens = bioenc.tokenize_aa(seq, k=3, stride=1)
    >>> tokens.shape
    (7,)  # 9 - 3 + 1 = 7 tokens
    """
    ...

def batch_tokenize_dna_shared(
    buffer: npt.NDArray[np.uint8],
    lengths: npt.NDArray[np.int64],
    k: int,
    stride: int = 3,
    reading_frame: Optional[int] = None,
    enable_padding: bool = False,
    max_len: Optional[int] = None,
    pad_value: int = -1,
    alphabet: AlphabetType = "acgtn",
    strand: StrandType = "forward",
) -> Union[List[npt.NDArray[np.int64]], npt.NDArray[np.int64]]:
    """
    Batch tokenize DNA sequences from a shared buffer.

    Uses OpenMP for parallel processing across sequences. Releases the
    Python GIL during computation to allow true parallelism.

    By default, returns variable-length arrays (one per sequence).
    Set enable_padding=True for fixed rectangular output.

    Parameters
    ----------
    buffer : np.ndarray[np.uint8]
        Concatenated sequences as uint8 array
    lengths : np.ndarray[np.int64]
        Length of each sequence
    k : int
        K-mer size (must be > 0 and <= 31)
    stride : int, optional
        Step size between k-mers (default: 3 for non-overlapping codons)
    reading_frame : int, optional
        If specified, crop all sequences to start at this reading frame (0, 1, or 2).
        Default: None (start at position 0)
    enable_padding : bool, optional
        If True, return rectangular array with padding (default: False)
    max_len : int, optional
        If enable_padding=True and this is set, pad to this length.
        If None, pad to actual maximum token count.
    pad_value : int, optional
        Padding value for short sequences (default: -1)
    alphabet : str, optional
        'acgtn' or 'iupac' (default: 'acgtn')
    strand : str, optional
        'forward', 'revcomp', or 'canonical' (default: 'forward')

    Returns
    -------
    List[np.ndarray[np.int64]] or np.ndarray[np.int64]
        If enable_padding=False: List of variable-length arrays
        If enable_padding=True: 2D array of shape (num_seqs, max_len)

    Raises
    ------
    RuntimeError
        If k <= 0, k > 31, stride <= 0, reading_frame not in {0, 1, 2, None}, or buffer access is out of bounds
    """
    ...

def batch_tokenize_dna_both(
    buffer: npt.NDArray[np.uint8],
    lengths: npt.NDArray[np.int64],
    k: int,
    stride: int = 3,
    enable_padding: bool = False,
    max_len: Optional[int] = None,
    pad_value: int = -1,
    alphabet: AlphabetType = "acgtn",
) -> Union[
    Tuple[List[npt.NDArray[np.int64]], List[npt.NDArray[np.int64]]],
    Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]
]:
    """
    Batch tokenize DNA sequences returning both forward and reverse complement.

    Uses OpenMP for parallel processing across sequences. Releases the
    Python GIL during computation to allow true parallelism.

    By default, returns variable-length arrays (one per sequence).
    Set enable_padding=True for fixed rectangular output.

    Parameters
    ----------
    buffer : np.ndarray[np.uint8]
        Concatenated sequences as uint8 array
    lengths : np.ndarray[np.int64]
        Length of each sequence
    k : int
        K-mer size (must be > 0 and <= 31)
    stride : int, optional
        Step size between k-mers (default: 3 for non-overlapping codons)
    enable_padding : bool, optional
        If True, return rectangular arrays with padding (default: False)
    max_len : int, optional
        If enable_padding=True and this is set, pad to this length.
        If None, pad to actual maximum token count.
    pad_value : int, optional
        Padding value for short sequences (default: -1)
    alphabet : str, optional
        'acgtn' or 'iupac' (default: 'acgtn')

    Returns
    -------
    Tuple[List[np.ndarray], List[np.ndarray]] or Tuple[np.ndarray, np.ndarray]
        If enable_padding=False: Tuple of (fwd_list, rev_list) with variable-length arrays
        If enable_padding=True: Tuple of 2D arrays, each shape (num_seqs, max_len)

    Raises
    ------
    RuntimeError
        If k <= 0, k > 31, stride <= 0, or buffer access is out of bounds
    """
    ...

def batch_tokenize_aa_shared(
    buffer: npt.NDArray[np.uint8],
    lengths: npt.NDArray[np.int64],
    k: int,
    stride: int = 1,
    enable_padding: bool = False,
    max_len: Optional[int] = None,
    pad_value: int = -1,
) -> Union[List[npt.NDArray[np.int64]], npt.NDArray[np.int64]]:
    """
    Batch tokenize amino acid sequences from a shared buffer.

    Uses OpenMP for parallel processing across sequences. Releases the
    Python GIL during computation to allow true parallelism.

    By default, returns variable-length arrays (one per sequence).
    Set enable_padding=True for fixed rectangular output.

    Parameters
    ----------
    buffer : np.ndarray[np.uint8]
        Concatenated sequences as uint8 array
    lengths : np.ndarray[np.int64]
        Length of each sequence
    k : int
        K-mer size (must be > 0 and <= 31)
    stride : int, optional
        Step size between k-mers (default: 1)
    enable_padding : bool, optional
        If True, return rectangular array with padding (default: False)
    max_len : int, optional
        If enable_padding=True and this is set, pad to this length.
        If None, pad to actual maximum token count.
    pad_value : int, optional
        Padding value for short sequences (default: -1)

    Returns
    -------
    List[np.ndarray[np.int64]] or np.ndarray[np.int64]
        If enable_padding=False: List of variable-length arrays
        If enable_padding=True: 2D array of shape (num_seqs, max_len)

    Raises
    ------
    RuntimeError
        If k <= 0, k > 31, stride <= 0, or buffer access is out of bounds
    """
    ...

def crop_and_tokenize_dna(
    buffer: npt.NDArray[np.uint8],
    lengths: npt.NDArray[np.int64],
    crop_starts: npt.NDArray[np.int64],
    crop_lengths: npt.NDArray[np.int64],
    k: int,
    stride: int = 3,
    reading_frame: Optional[int] = None,
    enable_padding: bool = False,
    max_len: Optional[int] = None,
    pad_value: int = -1,
    alphabet: AlphabetType = "acgtn",
    strand: StrandType = "forward",
) -> Union[List[npt.NDArray[np.int64]], npt.NDArray[np.int64]]:
    """
    Crop sequences to windows, then tokenize.

    Useful for:
    - Reading frames (crop_starts=[0,1,2], replicate buffer 3x)
    - Sliding windows
    - Data augmentation with random crops

    Parameters
    ----------
    buffer : np.ndarray[np.uint8]
        Concatenated sequences as uint8 array
    lengths : np.ndarray[np.int64]
        Length of each sequence
    crop_starts : np.ndarray[np.int64]
        Start position for crop in each sequence
    crop_lengths : np.ndarray[np.int64]
        Length of crop for each sequence
    k : int
        K-mer size
    stride : int, optional
        Step between k-mers (default: 3 for non-overlapping codons)
    reading_frame : int, optional
        If specified, apply additional reading frame offset (0, 1, or 2) after cropping.
        Default: None
    enable_padding : bool, optional
        Return rectangular array (default: False)
    max_len : int, optional
        Max length if padding enabled
    pad_value : int, optional
        Padding value (default: -1)
    alphabet : str, optional
        'acgtn' or 'iupac' (default: 'acgtn')
    strand : str, optional
        'forward', 'revcomp', or 'canonical' (default: 'forward')

    Returns
    -------
    List[np.ndarray[np.int64]] or np.ndarray[np.int64]
        Tokenized cropped sequences

    Raises
    ------
    RuntimeError
        If crop extends beyond sequence boundaries or reading_frame not in {0, 1, 2, None}
    """
    ...

def crop_and_tokenize_aa(
    buffer: npt.NDArray[np.uint8],
    lengths: npt.NDArray[np.int64],
    crop_starts: npt.NDArray[np.int64],
    crop_lengths: npt.NDArray[np.int64],
    k: int,
    stride: int = 1,
    enable_padding: bool = False,
    max_len: Optional[int] = None,
    pad_value: int = -1,
) -> Union[List[npt.NDArray[np.int64]], npt.NDArray[np.int64]]:
    """
    Crop amino acid sequences to windows, then tokenize.

    Useful for windowing and data augmentation.

    Parameters
    ----------
    buffer : np.ndarray[np.uint8]
        Concatenated sequences as uint8 array
    lengths : np.ndarray[np.int64]
        Length of each sequence
    crop_starts : np.ndarray[np.int64]
        Start position for crop in each sequence
    crop_lengths : np.ndarray[np.int64]
        Length of crop for each sequence
    k : int
        K-mer size
    stride : int, optional
        Step between k-mers (default: 1)
    enable_padding : bool, optional
        Return rectangular array (default: False)
    max_len : int, optional
        Max length if padding enabled
    pad_value : int, optional
        Padding value (default: -1)

    Returns
    -------
    List[np.ndarray[np.int64]] or np.ndarray[np.int64]
        Tokenized cropped sequences

    Raises
    ------
    RuntimeError
        If crop extends beyond sequence boundaries
    """
    ...

def hash_tokens(
    tokens: npt.NDArray[np.int64],
    num_buckets: int,
) -> npt.NDArray[np.int64]:
    """
    Hash token indices to a fixed number of buckets.

    Useful for large k values where vocabulary size exceeds embedding table limits.
    Preserves negative values (padding) unchanged.

    Uses OpenMP for parallel processing. Releases the Python GIL during
    computation to allow true parallelism.

    Parameters
    ----------
    tokens : np.ndarray[np.int64]
        Array of int64 token indices (any shape)
    num_buckets : int
        Number of hash buckets (must be > 0)

    Returns
    -------
    np.ndarray[np.int64]
        Hashed tokens with same shape as input

    Raises
    ------
    RuntimeError
        If num_buckets <= 0
    """
    ...
