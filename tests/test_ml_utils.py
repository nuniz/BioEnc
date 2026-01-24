"""Unit tests for bioenc ML utilities."""

import numpy as np
import pytest
import bioenc

# Check if PyTorch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Check if TensorFlow is available
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestPyTorchConversion:
    """Tests for PyTorch conversion utilities."""

    def test_to_torch_basic(self):
        """Test basic NumPy to PyTorch conversion."""
        tokens = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        tensor = bioenc.to_torch(tokens)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.long
        assert tensor.shape == (5,)
        assert torch.all(tensor == torch.tensor([1, 2, 3, 4, 5]))

    def test_to_torch_zero_copy(self):
        """Test zero-copy conversion when device='cpu'."""
        tokens = np.array([1, 2, 3], dtype=np.int64)
        tensor = bioenc.to_torch(tokens, device='cpu')

        # Verify zero-copy by checking data pointers
        assert tensor.data_ptr() == tokens.ctypes.data

    def test_to_torch_cuda(self):
        """Test conversion to CUDA device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        tokens = np.array([1, 2, 3], dtype=np.int64)
        tensor = bioenc.to_torch(tokens, device='cuda')

        assert tensor.device.type == 'cuda'

    def test_batch_to_torch_basic(self):
        """Test batch conversion with padding."""
        tokens_list = [
            np.array([1, 2, 3], dtype=np.int64),
            np.array([4, 5], dtype=np.int64),
            np.array([6, 7, 8, 9], dtype=np.int64),
        ]

        padded, lengths = bioenc.batch_to_torch(tokens_list)

        assert isinstance(padded, torch.Tensor)
        assert isinstance(lengths, torch.Tensor)
        assert padded.shape == (3, 4)  # Max length is 4
        assert lengths.shape == (3,)

        # Check lengths
        assert torch.all(lengths == torch.tensor([3, 2, 4]))

        # Check padding
        assert padded[1, 2] == -1  # Second sequence padded at position 2
        assert padded[1, 3] == -1

    def test_batch_to_torch_custom_pad_value(self):
        """Test batch conversion with custom pad value."""
        tokens_list = [
            np.array([1, 2], dtype=np.int64),
            np.array([3], dtype=np.int64),
        ]

        padded, lengths = bioenc.batch_to_torch(tokens_list, pad_value=-99)

        assert padded[1, 1] == -99

    def test_batch_to_torch_empty_list(self):
        """Test batch conversion with empty list."""
        padded, lengths = bioenc.batch_to_torch([])

        assert padded.shape == (0, 0)
        assert lengths.shape == (0,)

    def test_frames_to_torch_basic(self):
        """Test frames conversion to PyTorch."""
        # Simulate output from batch_tokenize_dna_all_frames
        frames_list = [
            [  # Sequence 1
                np.array([1, 2, 3], dtype=np.int64),  # Frame 0
                np.array([4, 5], dtype=np.int64),     # Frame 1
                np.array([6], dtype=np.int64),        # Frame 2
                np.array([7, 8], dtype=np.int64),     # Frame 3
                np.array([9, 10, 11], dtype=np.int64),# Frame 4
                np.array([12], dtype=np.int64),       # Frame 5
            ],
            [  # Sequence 2
                np.array([13, 14], dtype=np.int64),
                np.array([15], dtype=np.int64),
                np.array([16, 17, 18], dtype=np.int64),
                np.array([19], dtype=np.int64),
                np.array([20, 21], dtype=np.int64),
                np.array([22, 23, 24, 25], dtype=np.int64),
            ],
        ]

        tensor = bioenc.frames_to_torch(frames_list)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (2, 6, 4)  # 2 seqs, 6 frames, max_len=4
        assert tensor.dtype == torch.long

        # Check specific values
        assert tensor[0, 0, 0] == 1  # Seq 0, frame 0, token 0
        assert tensor[0, 4, 2] == 11  # Seq 0, frame 4, token 2

        # Check padding
        assert tensor[0, 2, 1] == -1  # Seq 0, frame 2 has only 1 token

    def test_frames_to_torch_custom_pad_value(self):
        """Test frames conversion with custom pad value."""
        frames_list = [
            [
                np.array([1], dtype=np.int64),
                np.array([2], dtype=np.int64),
                np.array([3], dtype=np.int64),
                np.array([4], dtype=np.int64),
                np.array([5], dtype=np.int64),
                np.array([6], dtype=np.int64),
            ],
        ]

        tensor = bioenc.frames_to_torch(frames_list, pad_value=-999)

        assert tensor[0, 0, 1] == -999  # Padded position

    def test_frames_to_torch_empty(self):
        """Test frames conversion with empty list."""
        tensor = bioenc.frames_to_torch([])

        assert tensor.shape == (0, 6, 0)


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")
class TestTensorFlowConversion:
    """Tests for TensorFlow conversion utilities."""

    def test_to_tensorflow_basic(self):
        """Test basic NumPy to TensorFlow conversion."""
        tokens = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        tensor = bioenc.to_tensorflow(tokens)

        assert isinstance(tensor, tf.Tensor)
        assert tensor.dtype == tf.int64
        assert tensor.shape == (5,)
        assert np.array_equal(tensor.numpy(), tokens)

    def test_batch_to_tensorflow_basic(self):
        """Test batch conversion with padding."""
        tokens_list = [
            np.array([1, 2, 3], dtype=np.int64),
            np.array([4, 5], dtype=np.int64),
            np.array([6, 7, 8, 9], dtype=np.int64),
        ]

        padded, lengths = bioenc.batch_to_tensorflow(tokens_list)

        assert isinstance(padded, tf.Tensor)
        assert isinstance(lengths, tf.Tensor)
        assert padded.shape == (3, 4)  # Max length is 4
        assert lengths.shape == (3,)

        # Check lengths
        assert np.array_equal(lengths.numpy(), [3, 2, 4])

        # Check padding
        assert padded.numpy()[1, 2] == -1
        assert padded.numpy()[1, 3] == -1

    def test_batch_to_tensorflow_custom_pad_value(self):
        """Test batch conversion with custom pad value."""
        tokens_list = [
            np.array([1, 2], dtype=np.int64),
            np.array([3], dtype=np.int64),
        ]

        padded, lengths = bioenc.batch_to_tensorflow(tokens_list, pad_value=-99)

        assert padded.numpy()[1, 1] == -99

    def test_batch_to_tensorflow_empty_list(self):
        """Test batch conversion with empty list."""
        padded, lengths = bioenc.batch_to_tensorflow([])

        assert padded.shape == (0, 0)
        assert lengths.shape == (0,)

    def test_frames_to_tensorflow_basic(self):
        """Test frames conversion to TensorFlow."""
        frames_list = [
            [
                np.array([1, 2, 3], dtype=np.int64),
                np.array([4, 5], dtype=np.int64),
                np.array([6], dtype=np.int64),
                np.array([7, 8], dtype=np.int64),
                np.array([9, 10, 11], dtype=np.int64),
                np.array([12], dtype=np.int64),
            ],
        ]

        tensor = bioenc.frames_to_tensorflow(frames_list)

        assert isinstance(tensor, tf.Tensor)
        assert tensor.shape == (1, 6, 3)  # 1 seq, 6 frames, max_len=3
        assert tensor.dtype == tf.int64

    def test_frames_to_tensorflow_empty(self):
        """Test frames conversion with empty list."""
        tensor = bioenc.frames_to_tensorflow([])

        assert tensor.shape == (0, 6, 0)


class TestMLUtilsWithoutFrameworks:
    """Tests that ML utilities raise proper errors when frameworks not installed."""

    def test_to_torch_import_error(self):
        """Test that to_torch raises ImportError when PyTorch not installed."""
        if TORCH_AVAILABLE:
            pytest.skip("PyTorch is installed")

        tokens = np.array([1, 2, 3], dtype=np.int64)
        with pytest.raises(ImportError, match="PyTorch not installed"):
            bioenc.to_torch(tokens)

    def test_to_tensorflow_import_error(self):
        """Test that to_tensorflow raises ImportError when TensorFlow not installed."""
        if TF_AVAILABLE:
            pytest.skip("TensorFlow is installed")

        tokens = np.array([1, 2, 3], dtype=np.int64)
        with pytest.raises(ImportError, match="TensorFlow not installed"):
            bioenc.to_tensorflow(tokens)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestIntegrationWithBioenc:
    """Integration tests combining bioenc tokenization with ML utilities."""

    def test_single_sequence_to_torch(self):
        """Test end-to-end: tokenize -> to_torch."""
        seq = np.frombuffer(b"ATGCGTAAA", dtype=np.uint8)
        tokens = bioenc.tokenize_dna(seq, k=3)
        tensor = bioenc.to_torch(tokens)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3,)  # 3 non-overlapping codons

    def test_batch_to_torch_integration(self):
        """Test end-to-end: batch tokenize -> batch_to_torch."""
        sequences = [b"ATGCGTAAA", b"TTT", b"AAACCCGGG"]
        buffer = np.frombuffer(b"".join(sequences), dtype=np.uint8)
        lengths = np.array([9, 3, 9], dtype=np.int64)

        tokens_list = bioenc.batch_tokenize_dna_shared(
            buffer, lengths, k=3
        )

        padded, seq_lengths = bioenc.batch_to_torch(tokens_list, device='cpu')

        assert padded.shape == (3, 3)  # 3 sequences, max 3 tokens
        assert torch.all(seq_lengths == torch.tensor([3, 1, 3]))

    def test_all_frames_to_torch_integration(self):
        """Test end-to-end: all_frames tokenize -> frames_to_torch."""
        sequences = [b"ATGCGTAAATGA", b"TTTGGGCCCAAA"]
        buffer = np.frombuffer(b"".join(sequences), dtype=np.uint8)
        lengths = np.array([12, 12], dtype=np.int64)

        all_frames = bioenc.batch_tokenize_dna_all_frames(
            buffer, lengths, k=3
        )

        tensor = bioenc.frames_to_torch(all_frames)

        assert tensor.shape == (2, 6, 4)  # 2 seqs, 6 frames, max 4 tokens


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
