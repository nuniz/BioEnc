"""
Test input validation for bioenc.

These tests verify that invalid inputs raise informative exceptions
rather than causing crashes or undefined behavior.
"""

import pytest
import numpy as np
import bioenc


class TestKValidation:
    """Test k parameter validation."""

    def test_k_zero_raises(self):
        """Test that k=0 raises an error."""
        seq = np.frombuffer(b"ACGTACGT", dtype=np.uint8)
        with pytest.raises(RuntimeError, match="k must be positive, got 0"):
            bioenc.tokenize_dna(seq, k=0)

    def test_k_negative_raises(self):
        """Test that negative k raises an error."""
        seq = np.frombuffer(b"ACGTACGT", dtype=np.uint8)
        with pytest.raises(RuntimeError, match="k must be positive, got -1"):
            bioenc.tokenize_dna(seq, k=-1)

    def test_k_too_large_raises(self):
        """Test that k > 31 raises an error (overflow risk)."""
        seq = np.frombuffer(b"A" * 100, dtype=np.uint8)
        with pytest.raises(RuntimeError, match="k > 31 may cause overflow, got 32"):
            bioenc.tokenize_dna(seq, k=32)

    def test_k_31_works(self):
        """Test that k=31 is allowed (boundary case)."""
        seq = np.frombuffer(b"A" * 40, dtype=np.uint8)
        tokens = bioenc.tokenize_dna(seq, k=31)
        assert len(tokens) > 0


class TestStrideValidation:
    """Test stride parameter validation."""

    def test_stride_zero_raises(self):
        """Test that stride=0 raises an error."""
        seq = np.frombuffer(b"ACGTACGT", dtype=np.uint8)
        with pytest.raises(RuntimeError, match="stride must be positive, got 0"):
            bioenc.tokenize_dna(seq, k=3, stride=0)

    def test_stride_negative_raises(self):
        """Test that negative stride raises an error."""
        seq = np.frombuffer(b"ACGTACGT", dtype=np.uint8)
        with pytest.raises(RuntimeError, match="stride must be positive, got -5"):
            bioenc.tokenize_dna(seq, k=3, stride=-5)


class TestBatchValidation:
    """Test batch function validation."""

    def test_negative_length_raises(self):
        """Test that negative lengths raise an error."""
        buffer = np.frombuffer(b"ACGTACGT" * 10, dtype=np.uint8)
        lengths = np.array([8, -5, 8], dtype=np.int64)  # Negative length

        with pytest.raises(RuntimeError, match="negative.*length"):
            bioenc.batch_tokenize_dna_shared(
                buffer, lengths, k=3, max_len=10
            )

    def test_buffer_overrun_raises(self):
        """Test that out-of-bounds buffer access raises an error."""
        buffer = np.frombuffer(b"ACGTACGT" * 2, dtype=np.uint8)  # 16 bytes
        lengths = np.array([8, 8, 10], dtype=np.int64)  # Total 26 > 16

        with pytest.raises(RuntimeError, match="sequence.*extends beyond buffer"):
            bioenc.batch_tokenize_dna_shared(
                buffer, lengths, k=3, max_len=10
            )

    def test_batch_k_validation(self):
        """Test that batch functions validate k."""
        buffer = np.frombuffer(b"ACGTACGT", dtype=np.uint8)
        lengths = np.array([8], dtype=np.int64)

        with pytest.raises(RuntimeError, match="k must be positive"):
            bioenc.batch_tokenize_dna_shared(
                buffer, lengths, k=0, max_len=10
            )

    def test_batch_stride_validation(self):
        """Test that batch functions validate stride."""
        buffer = np.frombuffer(b"ACGTACGT", dtype=np.uint8)
        lengths = np.array([8], dtype=np.int64)

        with pytest.raises(RuntimeError, match="stride must be positive"):
            bioenc.batch_tokenize_dna_shared(
                buffer, lengths, k=3, max_len=10, stride=-1
            )


class TestAAValidation:
    """Test amino acid function validation."""

    def test_aa_k_validation(self):
        """Test that AA functions validate k."""
        buffer = np.frombuffer(b"ACDEFGHIKLMNPQRSTVWY", dtype=np.uint8)
        lengths = np.array([20], dtype=np.int64)

        with pytest.raises(RuntimeError, match="k > 31 may cause overflow"):
            bioenc.batch_tokenize_aa_shared(
                buffer, lengths, k=35, max_len=10
            )

    def test_aa_stride_validation(self):
        """Test that AA functions validate stride."""
        buffer = np.frombuffer(b"ACDEFGHIKLMNPQRSTVWY", dtype=np.uint8)
        lengths = np.array([20], dtype=np.int64)

        with pytest.raises(RuntimeError, match="stride must be positive"):
            bioenc.batch_tokenize_aa_shared(
                buffer, lengths, k=3, max_len=10, stride=0
            )

    def test_aa_buffer_validation(self):
        """Test that AA functions validate buffer access."""
        buffer = np.frombuffer(b"ACDEFG", dtype=np.uint8)  # 6 bytes
        lengths = np.array([3, 5], dtype=np.int64)  # Total 8 > 6

        with pytest.raises(RuntimeError, match="sequence.*extends beyond buffer"):
            bioenc.batch_tokenize_aa_shared(
                buffer, lengths, k=2, max_len=5
            )


class TestBothStrandsValidation:
    """Test batch_tokenize_dna_both validation."""

    def test_both_k_validation(self):
        """Test that both strands function validates k."""
        buffer = np.frombuffer(b"ACGTACGT", dtype=np.uint8)
        lengths = np.array([8], dtype=np.int64)

        with pytest.raises(RuntimeError, match="k must be positive, got -1"):
            bioenc.batch_tokenize_dna_both(
                buffer, lengths, k=-1, max_len=10
            )

    def test_both_buffer_validation(self):
        """Test that both strands function validates buffer access."""
        buffer = np.frombuffer(b"ACGT", dtype=np.uint8)  # 4 bytes
        lengths = np.array([2, 5], dtype=np.int64)  # Total 7 > 4

        with pytest.raises(RuntimeError, match="sequence.*extends beyond buffer"):
            bioenc.batch_tokenize_dna_both(
                buffer, lengths, k=2, max_len=5
            )


class TestErrorMessages:
    """Test that error messages are informative."""

    def test_error_includes_context(self):
        """Test that error messages include function name."""
        seq = np.frombuffer(b"ACGT", dtype=np.uint8)
        with pytest.raises(RuntimeError, match="tokenize_dna:"):
            bioenc.tokenize_dna(seq, k=0)

    def test_error_includes_value(self):
        """Test that error messages include the invalid value."""
        seq = np.frombuffer(b"ACGT", dtype=np.uint8)
        with pytest.raises(RuntimeError, match="got -3"):
            bioenc.tokenize_dna(seq, k=-3, stride=1)

    def test_buffer_error_includes_details(self):
        """Test that buffer overrun errors include helpful details."""
        buffer = np.frombuffer(b"ACGT" * 5, dtype=np.uint8)  # 20 bytes
        lengths = np.array([15, 10], dtype=np.int64)  # Total 25 > 20

        with pytest.raises(RuntimeError) as exc_info:
            bioenc.batch_tokenize_dna_shared(
                buffer, lengths, k=3, max_len=5
            )

        error_msg = str(exc_info.value)
        assert "sequence" in error_msg
        assert "buffer" in error_msg.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
