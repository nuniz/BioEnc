"""
ML framework integration utilities for bioenc.

Provides zero-copy (when possible) conversion from NumPy arrays to PyTorch
tensors and TensorFlow tensors, with support for batching and padding.
"""

import numpy as np
from typing import List, Tuple, Union, Optional

# Check for PyTorch availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

# Check for TensorFlow availability
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None  # type: ignore


# ============================================================================
# PyTorch conversion utilities
# ============================================================================

def to_torch(
    tokens: np.ndarray,
    dtype = None,
    device: Union[str, object] = 'cpu'
):
    """
    Convert NumPy token array to PyTorch tensor.

    Uses zero-copy conversion (torch.from_numpy) when device='cpu' and dtype
    matches, otherwise copies data.

    Parameters
    ----------
    tokens : np.ndarray
        Token array from tokenize_* functions
    dtype : torch.dtype, optional
        PyTorch dtype (default: torch.long for embedding layers)
    device : str or torch.device, optional
        Target device: 'cpu', 'cuda', or torch.device (default: 'cpu')

    Returns
    -------
    torch.Tensor
        PyTorch tensor (shares memory with NumPy if device='cpu' and dtype matches)

    Raises
    ------
    ImportError
        If PyTorch is not installed

    Examples
    --------
    >>> import bioenc
    >>> import numpy as np
    >>> seq = np.frombuffer(b"ATGCGTAAA", dtype=np.uint8)
    >>> tokens = bioenc.tokenize_dna(seq, k=3)
    >>> tensor = bioenc.to_torch(tokens, device='cuda')
    >>> # Ready for embedding layer
    >>> # embeddings = embedding_layer(tensor)
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch not installed. Install with: pip install torch"
        )

    if dtype is None:
        dtype = torch.long

    # Zero-copy if device='cpu' and dtype is compatible
    if device == 'cpu' and dtype == torch.long and tokens.dtype == np.int64:
        return torch.from_numpy(tokens)
    else:
        return torch.tensor(tokens, dtype=dtype, device=device)


def batch_to_torch(
    tokens_list: List[np.ndarray],
    pad_value: int = -1,
    dtype = None,
    device: Union[str, object] = 'cpu'
):
    """
    Convert batch of variable-length token arrays to padded PyTorch tensor.

    Parameters
    ----------
    tokens_list : List[np.ndarray]
        List of token arrays (from batch_tokenize_*)
    pad_value : int, optional
        Padding value for short sequences (default: -1)
    dtype : torch.dtype, optional
        PyTorch dtype (default: torch.long)
    device : str or torch.device, optional
        Target device (default: 'cpu')

    Returns
    -------
    padded_tensor : torch.Tensor
        Shape [batch_size, max_len], padded with pad_value
    lengths_tensor : torch.Tensor
        Shape [batch_size], original lengths for masking

    Raises
    ------
    ImportError
        If PyTorch is not installed

    Examples
    --------
    >>> import bioenc
    >>> import numpy as np
    >>> # Create batch
    >>> sequences = [b"ATGCGTAAA", b"TTT"]
    >>> buffer = np.frombuffer(b"".join(sequences), dtype=np.uint8)
    >>> lengths = np.array([9, 3], dtype=np.int64)
    >>> offsets = np.array([0, 9], dtype=np.int64)
    >>> tokens_list = bioenc.batch_tokenize_dna_shared(
    ...     buffer, offsets, lengths, k=3
    ... )
    >>> padded, lengths = bioenc.batch_to_torch(tokens_list, device='cuda')
    >>> # Use with PackedSequence or attention mask
    >>> mask = (padded != -1)
    >>> # output = model(padded, mask=mask)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not installed. Install with: pip install torch")

    if dtype is None:
        dtype = torch.long

    if len(tokens_list) == 0:
        return torch.empty((0, 0), dtype=dtype, device=device), torch.empty(0, dtype=torch.long, device=device)

    # Find max length
    max_len = max(len(t) for t in tokens_list)
    batch_size = len(tokens_list)

    # Create padded array
    padded = np.full((batch_size, max_len), pad_value, dtype=np.int64)
    lengths = np.array([len(t) for t in tokens_list], dtype=np.int64)

    # Fill in tokens
    for i, tokens in enumerate(tokens_list):
        if len(tokens) > 0:
            padded[i, :len(tokens)] = tokens

    # Convert to torch
    padded_tensor = torch.tensor(padded, dtype=dtype, device=device)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=device)

    return padded_tensor, lengths_tensor


def frames_to_torch(
    frames_list: List[List[np.ndarray]],
    pad_value: int = -1,
    dtype = None,
    device: Union[str, object] = 'cpu'
):
    """
    Convert all-frames batch output to PyTorch tensor.

    Parameters
    ----------
    frames_list : List[List[np.ndarray]]
        Output from batch_tokenize_dna_all_frames()
        Outer list: sequences, inner list: 6 frames

    pad_value : int, optional
        Padding value (default: -1)
    dtype : torch.dtype, optional
        PyTorch dtype (default: torch.long)
    device : str or torch.device, optional
        Target device (default: 'cpu')

    Returns
    -------
    torch.Tensor
        Shape [batch_size, 6, max_len]
        Each sequence has 6 reading frames padded to max_len

    Raises
    ------
    ImportError
        If PyTorch is not installed

    Examples
    --------
    >>> import bioenc
    >>> import numpy as np
    >>> sequences = [b"ATGCGTAAA", b"TTTGGGCCC"]
    >>> buffer = np.frombuffer(b"".join(sequences), dtype=np.uint8)
    >>> lengths = np.array([9, 9], dtype=np.int64)
    >>> offsets = np.array([0, 9], dtype=np.int64)
    >>> all_frames = bioenc.batch_tokenize_dna_all_frames(
    ...     buffer, offsets, lengths, k=3
    ... )
    >>> tensor = bioenc.frames_to_torch(all_frames, device='cuda')
    >>> tensor.shape  # [2, 6, max_len]
    >>> # Ready for multi-frame model
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not installed. Install with: pip install torch")

    if dtype is None:
        dtype = torch.long

    if len(frames_list) == 0:
        return torch.empty((0, 6, 0), dtype=dtype, device=device)

    batch_size = len(frames_list)
    num_frames = 6

    # Find max length across all frames and sequences
    max_len = 0
    for seq_frames in frames_list:
        for frame_tokens in seq_frames:
            max_len = max(max_len, len(frame_tokens))

    # Create padded array
    padded = np.full((batch_size, num_frames, max_len), pad_value, dtype=np.int64)

    # Fill in tokens
    for i, seq_frames in enumerate(frames_list):
        for j, frame_tokens in enumerate(seq_frames):
            if len(frame_tokens) > 0:
                padded[i, j, :len(frame_tokens)] = frame_tokens

    return torch.tensor(padded, dtype=dtype, device=device)


# ============================================================================
# TensorFlow conversion utilities
# ============================================================================

def to_tensorflow(
    tokens: np.ndarray,
    dtype = None
):
    """
    Convert NumPy token array to TensorFlow tensor.

    Parameters
    ----------
    tokens : np.ndarray
        Token array from tokenize_* functions
    dtype : tf.DType, optional
        TensorFlow dtype (default: tf.int64)

    Returns
    -------
    tf.Tensor
        TensorFlow tensor

    Raises
    ------
    ImportError
        If TensorFlow is not installed

    Examples
    --------
    >>> import bioenc
    >>> import numpy as np
    >>> seq = np.frombuffer(b"ATGCGTAAA", dtype=np.uint8)
    >>> tokens = bioenc.tokenize_dna(seq, k=3)
    >>> tensor = bioenc.to_tensorflow(tokens)
    >>> # Ready for embedding layer
    >>> # embeddings = embedding_layer(tensor)
    """
    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow not installed. Install with: pip install tensorflow"
        )

    if dtype is None:
        dtype = tf.int64

    return tf.convert_to_tensor(tokens, dtype=dtype)


def batch_to_tensorflow(
    tokens_list: List[np.ndarray],
    pad_value: int = -1,
    dtype = None
):
    """
    Convert batch of variable-length token arrays to padded TensorFlow tensor.

    Parameters
    ----------
    tokens_list : List[np.ndarray]
        List of token arrays (from batch_tokenize_*)
    pad_value : int, optional
        Padding value for short sequences (default: -1)
    dtype : tf.DType, optional
        TensorFlow dtype (default: tf.int64)

    Returns
    -------
    padded_tensor : tf.Tensor
        Shape [batch_size, max_len], padded with pad_value
    lengths_tensor : tf.Tensor
        Shape [batch_size], original lengths for masking

    Raises
    ------
    ImportError
        If TensorFlow is not installed

    Examples
    --------
    >>> import bioenc
    >>> import numpy as np
    >>> sequences = [b"ATGCGTAAA", b"TTT"]
    >>> buffer = np.frombuffer(b"".join(sequences), dtype=np.uint8)
    >>> lengths = np.array([9, 3], dtype=np.int64)
    >>> offsets = np.array([0, 9], dtype=np.int64)
    >>> tokens_list = bioenc.batch_tokenize_dna_shared(
    ...     buffer, offsets, lengths, k=3
    ... )
    >>> padded, lengths = bioenc.batch_to_tensorflow(tokens_list)
    >>> # Use with masking
    >>> mask = padded != -1
    >>> # output = model(padded, mask=mask)
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not installed. Install with: pip install tensorflow")

    if dtype is None:
        dtype = tf.int64

    if len(tokens_list) == 0:
        return tf.constant([], dtype=dtype, shape=(0, 0)), tf.constant([], dtype=tf.int64, shape=(0,))

    # Find max length
    max_len = max(len(t) for t in tokens_list)
    batch_size = len(tokens_list)

    # Create padded array
    padded = np.full((batch_size, max_len), pad_value, dtype=np.int64)
    lengths = np.array([len(t) for t in tokens_list], dtype=np.int64)

    # Fill in tokens
    for i, tokens in enumerate(tokens_list):
        if len(tokens) > 0:
            padded[i, :len(tokens)] = tokens

    # Convert to TensorFlow
    padded_tensor = tf.convert_to_tensor(padded, dtype=dtype)
    lengths_tensor = tf.convert_to_tensor(lengths, dtype=tf.int64)

    return padded_tensor, lengths_tensor


def frames_to_tensorflow(
    frames_list: List[List[np.ndarray]],
    pad_value: int = -1,
    dtype = None
):
    """
    Convert all-frames batch output to TensorFlow tensor.

    Parameters
    ----------
    frames_list : List[List[np.ndarray]]
        Output from batch_tokenize_dna_all_frames()
        Outer list: sequences, inner list: 6 frames
    pad_value : int, optional
        Padding value (default: -1)
    dtype : tf.DType, optional
        TensorFlow dtype (default: tf.int64)

    Returns
    -------
    tf.Tensor
        Shape [batch_size, 6, max_len]
        Each sequence has 6 reading frames padded to max_len

    Raises
    ------
    ImportError
        If TensorFlow is not installed

    Examples
    --------
    >>> import bioenc
    >>> import numpy as np
    >>> sequences = [b"ATGCGTAAA", b"TTTGGGCCC"]
    >>> buffer = np.frombuffer(b"".join(sequences), dtype=np.uint8)
    >>> lengths = np.array([9, 9], dtype=np.int64)
    >>> offsets = np.array([0, 9], dtype=np.int64)
    >>> all_frames = bioenc.batch_tokenize_dna_all_frames(
    ...     buffer, offsets, lengths, k=3
    ... )
    >>> tensor = bioenc.frames_to_tensorflow(all_frames)
    >>> tensor.shape  # [2, 6, max_len]
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not installed. Install with: pip install tensorflow")

    if dtype is None:
        dtype = tf.int64

    if len(frames_list) == 0:
        return tf.constant([], dtype=dtype, shape=(0, 6, 0))

    batch_size = len(frames_list)
    num_frames = 6

    # Find max length across all frames and sequences
    max_len = 0
    for seq_frames in frames_list:
        for frame_tokens in seq_frames:
            max_len = max(max_len, len(frame_tokens))

    # Create padded array
    padded = np.full((batch_size, num_frames, max_len), pad_value, dtype=np.int64)

    # Fill in tokens
    for i, seq_frames in enumerate(frames_list):
        for j, frame_tokens in enumerate(seq_frames):
            if len(frame_tokens) > 0:
                padded[i, j, :len(frame_tokens)] = frame_tokens

    return tf.convert_to_tensor(padded, dtype=dtype)
