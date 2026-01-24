"""Type stubs for bioenc.ml_utils module."""

from typing import List, Tuple, Union, Optional, Any
import numpy as np
import numpy.typing as npt

def to_torch(
    tokens: npt.NDArray[np.int64],
    dtype: Optional[Any] = None,
    device: Union[str, Any] = 'cpu'
) -> Any:
    """
    Convert NumPy token array to PyTorch tensor.

    Parameters
    ----------
    tokens : np.ndarray[np.int64]
        Token array from tokenize_* functions
    dtype : torch.dtype, optional
        PyTorch dtype (default: torch.long)
    device : str or torch.device, optional
        'cpu', 'cuda', or torch.device (default: 'cpu')

    Returns
    -------
    torch.Tensor
        PyTorch tensor (shares memory with NumPy if device='cpu')

    Raises
    ------
    ImportError
        If PyTorch not installed
    """
    ...

def batch_to_torch(
    tokens_list: List[npt.NDArray[np.int64]],
    pad_value: int = -1,
    dtype: Optional[Any] = None,
    device: Union[str, Any] = 'cpu'
) -> Tuple[Any, Any]:
    """
    Convert batch of variable-length token arrays to padded PyTorch tensor.

    Parameters
    ----------
    tokens_list : List[np.ndarray[np.int64]]
        List of token arrays (from batch_tokenize_*)
    pad_value : int, optional
        Padding value (default: -1)
    dtype : torch.dtype, optional
        PyTorch dtype (default: torch.long)
    device : str or torch.device, optional
        Target device (default: 'cpu')

    Returns
    -------
    padded_tensor : torch.Tensor
        Shape [batch_size, max_len]
    lengths_tensor : torch.Tensor
        Shape [batch_size]

    Raises
    ------
    ImportError
        If PyTorch not installed
    """
    ...

def frames_to_torch(
    frames_list: List[List[npt.NDArray[np.int64]]],
    pad_value: int = -1,
    dtype: Optional[Any] = None,
    device: Union[str, Any] = 'cpu'
) -> Any:
    """
    Convert all-frames batch output to PyTorch tensor.

    Parameters
    ----------
    frames_list : List[List[np.ndarray[np.int64]]]
        Output from batch_tokenize_dna_all_frames()
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

    Raises
    ------
    ImportError
        If PyTorch not installed
    """
    ...

def to_tensorflow(
    tokens: npt.NDArray[np.int64],
    dtype: Optional[Any] = None
) -> Any:
    """
    Convert NumPy token array to TensorFlow tensor.

    Parameters
    ----------
    tokens : np.ndarray[np.int64]
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
        If TensorFlow not installed
    """
    ...

def batch_to_tensorflow(
    tokens_list: List[npt.NDArray[np.int64]],
    pad_value: int = -1,
    dtype: Optional[Any] = None
) -> Tuple[Any, Any]:
    """
    Convert batch of variable-length token arrays to padded TensorFlow tensor.

    Parameters
    ----------
    tokens_list : List[np.ndarray[np.int64]]
        List of token arrays (from batch_tokenize_*)
    pad_value : int, optional
        Padding value (default: -1)
    dtype : tf.DType, optional
        TensorFlow dtype (default: tf.int64)

    Returns
    -------
    padded_tensor : tf.Tensor
        Shape [batch_size, max_len]
    lengths_tensor : tf.Tensor
        Shape [batch_size]

    Raises
    ------
    ImportError
        If TensorFlow not installed
    """
    ...

def frames_to_tensorflow(
    frames_list: List[List[npt.NDArray[np.int64]]],
    pad_value: int = -1,
    dtype: Optional[Any] = None
) -> Any:
    """
    Convert all-frames batch output to TensorFlow tensor.

    Parameters
    ----------
    frames_list : List[List[np.ndarray[np.int64]]]
        Output from batch_tokenize_dna_all_frames()
    pad_value : int, optional
        Padding value (default: -1)
    dtype : tf.DType, optional
        TensorFlow dtype (default: tf.int64)

    Returns
    -------
    tf.Tensor
        Shape [batch_size, 6, max_len]

    Raises
    ------
    ImportError
        If TensorFlow not installed
    """
    ...
