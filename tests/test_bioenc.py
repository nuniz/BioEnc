"""Unit tests for bioenc library."""

import numpy as np
import pytest
import bioenc


class TestTokenizeDna:
    """Tests for single sequence DNA tokenization."""

    def test_basic_forward(self):
        """Test basic forward strand tokenization."""
        seq = np.frombuffer(b"ACGT", dtype=np.uint8)
        tokens = bioenc.tokenize_dna(seq, k=2, stride=1)  # Explicitly set stride=1 for overlapping
        # PAD=0, UNK/N=1, A=2, C=3, G=4, T=5, base=6
        # AC = 2*6 + 3 = 15
        # CG = 3*6 + 4 = 22
        # GT = 4*6 + 5 = 29
        expected = np.array([15, 22, 29], dtype=np.int64)
        np.testing.assert_array_equal(tokens, expected)

    def test_forward_k3(self):
        """Test k=3 tokenization with stride=1 (overlapping)."""
        seq = np.frombuffer(b"ACGT", dtype=np.uint8)
        tokens = bioenc.tokenize_dna(seq, k=3, stride=1)
        # ACG = 2*36 + 3*6 + 4 = 94
        # CGT = 3*36 + 4*6 + 5 = 137
        expected = np.array([94, 137], dtype=np.int64)
        np.testing.assert_array_equal(tokens, expected)

    def test_stride3_default(self):
        """Test that stride=3 is the default."""
        seq = np.frombuffer(b"ATGCGTAAA", dtype=np.uint8)
        tokens = bioenc.tokenize_dna(seq, k=3)
        # Should get 3 non-overlapping codons: ATG, CGT, AAA
        assert len(tokens) == 3

    def test_stride3_explicit(self):
        """Test explicit stride=3 for non-overlapping codons."""
        seq = np.frombuffer(b"ATGCGTAAA", dtype=np.uint8)
        tokens = bioenc.tokenize_dna(seq, k=3, stride=3)
        # ATG = 2*36 + 5*6 + 4 = 106
        # CGT = 3*36 + 4*6 + 5 = 137
        # AAA = 2*36 + 2*6 + 2 = 86
        expected = np.array([106, 137, 86], dtype=np.int64)
        np.testing.assert_array_equal(tokens, expected)

    def test_revcomp_strand(self):
        """Test reverse complement strand tokenization."""
        seq = np.frombuffer(b"ACGT", dtype=np.uint8)
        tokens = bioenc.tokenize_dna(seq, k=2, stride=1, strand="revcomp")
        # RC of AC is GT, RC of CG is CG, RC of GT is AC
        # GT = 4*6 + 5 = 29
        # CG = 3*6 + 4 = 22
        # AC = 2*6 + 3 = 15
        expected = np.array([29, 22, 15], dtype=np.int64)
        np.testing.assert_array_equal(tokens, expected)

    def test_canonical_strand(self):
        """Test canonical strand tokenization (min of fwd and revcomp)."""
        seq = np.frombuffer(b"ACGT", dtype=np.uint8)
        tokens = bioenc.tokenize_dna(seq, k=2, stride=1, strand="canonical")
        # AC(15) vs GT(29) -> 15
        # CG(22) vs CG(22) -> 22
        # GT(29) vs AC(15) -> 15
        expected = np.array([15, 22, 15], dtype=np.int64)
        np.testing.assert_array_equal(tokens, expected)

    def test_stride(self):
        """Test tokenization with stride > 1."""
        seq = np.frombuffer(b"ACGTACGT", dtype=np.uint8)
        tokens = bioenc.tokenize_dna(seq, k=2, stride=2)
        # Positions 0,2,4,6 -> AC, GT, AC, GT
        expected = np.array([15, 29, 15, 29], dtype=np.int64)
        np.testing.assert_array_equal(tokens, expected)

    def test_lowercase(self):
        """Test that lowercase sequences work."""
        seq = np.frombuffer(b"acgt", dtype=np.uint8)
        tokens = bioenc.tokenize_dna(seq, k=2, stride=1)
        expected = np.array([15, 22, 29], dtype=np.int64)
        np.testing.assert_array_equal(tokens, expected)

    def test_unknown_char_maps_to_n(self):
        """Test that unknown characters map to UNK."""
        seq = np.frombuffer(b"AXGT", dtype=np.uint8)  # X is unknown
        tokens = bioenc.tokenize_dna(seq, k=2, stride=1)
        # AX -> A=2, X->UNK=1, so 2*6+1 = 13
        # XG -> UNK=1, G=4, so 1*6+4 = 10
        # GT -> 4*6+5 = 29
        expected = np.array([13, 10, 29], dtype=np.int64)
        np.testing.assert_array_equal(tokens, expected)

    def test_empty_sequence(self):
        """Test tokenization of empty sequence."""
        seq = np.frombuffer(b"", dtype=np.uint8)
        tokens = bioenc.tokenize_dna(seq, k=3)
        assert len(tokens) == 0

    def test_short_sequence(self):
        """Test sequence shorter than k."""
        seq = np.frombuffer(b"AC", dtype=np.uint8)
        tokens = bioenc.tokenize_dna(seq, k=3)
        assert len(tokens) == 0


class TestReadingFrame:
    """Tests for reading frame parameter."""

    def test_reading_frame_0(self):
        """Test reading_frame=0 (same as default)."""
        seq = np.frombuffer(b"ATGCGTAAA", dtype=np.uint8)
        tokens_default = bioenc.tokenize_dna(seq, k=3)
        tokens_frame0 = bioenc.tokenize_dna(seq, k=3, reading_frame=0)
        np.testing.assert_array_equal(tokens_default, tokens_frame0)

    def test_reading_frame_1(self):
        """Test reading_frame=1 starts at position 1."""
        seq = np.frombuffer(b"ATGCGTAAA", dtype=np.uint8)
        tokens = bioenc.tokenize_dna(seq, k=3, reading_frame=1)
        # Starts at "TGCGTAAA": TGC, GTA, AA_
        # Should get 3 tokens (8 bases / 3 = 2 complete + 1 partial)
        assert len(tokens) == 2  # 8 bases: TGC, GTA only (AA is incomplete)

    def test_reading_frame_2(self):
        """Test reading_frame=2 starts at position 2."""
        seq = np.frombuffer(b"ATGCGTAAA", dtype=np.uint8)
        tokens = bioenc.tokenize_dna(seq, k=3, reading_frame=2)
        # Starts at "GCGTAAA": GCG, TAA, A__
        assert len(tokens) == 2  # 7 bases: GCG, TAA only

    def test_reading_frame_too_short(self):
        """Test reading_frame on sequence too short after offset."""
        seq = np.frombuffer(b"ATG", dtype=np.uint8)
        tokens = bioenc.tokenize_dna(seq, k=3, reading_frame=2)
        # Only 1 base left after offset, can't make k=3
        assert len(tokens) == 0

    def test_reading_frame_with_stride1(self):
        """Test reading_frame works with stride=1 (overlapping)."""
        seq = np.frombuffer(b"ATGCGTAAA", dtype=np.uint8)
        tokens = bioenc.tokenize_dna(seq, k=3, stride=1, reading_frame=1)
        # Starts at "TGCGTAAA" (8 bases): TGC, GCG, CGT, GTA, TAA, AAA
        assert len(tokens) == 6

    def test_reading_frame_validation(self):
        """Test reading_frame validates input (must be 0, 1, or 2)."""
        seq = np.frombuffer(b"ATGCGTAAA", dtype=np.uint8)
        with pytest.raises(RuntimeError, match="reading_frame must be 0, 1, or 2"):
            bioenc.tokenize_dna(seq, k=3, reading_frame=3)
        with pytest.raises(RuntimeError, match="reading_frame must be 0, 1, or 2"):
            bioenc.tokenize_dna(seq, k=3, reading_frame=-1)


class TestAllFrames:
    """Tests for tokenize_dna_all_frames."""

    def test_all_frames_returns_6(self):
        """Test that all_frames returns 6 arrays."""
        seq = np.frombuffer(b"ATGCGTAAATGA", dtype=np.uint8)
        frames = bioenc.tokenize_dna_all_frames(seq, k=3)
        assert isinstance(frames, list)
        assert len(frames) == 6

    def test_all_frames_structure(self):
        """Test structure of all_frames output."""
        seq = np.frombuffer(b"ATGCGTAAATGA", dtype=np.uint8)
        frames = bioenc.tokenize_dna_all_frames(seq, k=3)
        # All should be numpy arrays
        for frame in frames:
            assert isinstance(frame, np.ndarray)
            assert frame.dtype == np.int64

    def test_all_frames_lengths(self):
        """Test frame lengths are correct."""
        seq = np.frombuffer(b"ATGCGTAAATGA", dtype=np.uint8)  # 12 bases
        frames = bioenc.tokenize_dna_all_frames(seq, k=3, stride=3)
        # Frame 0 (12 bases): 4 codons
        # Frame 1 (11 bases): 3 codons
        # Frame 2 (10 bases): 3 codons
        assert len(frames[0]) == 4  # Forward frame 0
        assert len(frames[1]) == 3  # Forward frame 1
        assert len(frames[2]) == 3  # Forward frame 2
        assert len(frames[3]) == 4  # Reverse frame 0
        assert len(frames[4]) == 3  # Reverse frame 1
        assert len(frames[5]) == 3  # Reverse frame 2

    def test_all_frames_different_values(self):
        """Test that different frames produce different tokens."""
        seq = np.frombuffer(b"ATGCGTAAATGA", dtype=np.uint8)
        frames = bioenc.tokenize_dna_all_frames(seq, k=3, stride=3)
        # Frames should not all be identical
        assert not np.array_equal(frames[0], frames[1])
        assert not np.array_equal(frames[1], frames[2])

    def test_all_frames_matches_reading_frame(self):
        """Test all_frames[i] matches reading_frame=i for forward."""
        seq = np.frombuffer(b"ATGCGTAAATGA", dtype=np.uint8)
        frames = bioenc.tokenize_dna_all_frames(seq, k=3, stride=3)

        # Check forward frames match
        f0 = bioenc.tokenize_dna(seq, k=3, stride=3, reading_frame=0)
        f1 = bioenc.tokenize_dna(seq, k=3, stride=3, reading_frame=1)
        f2 = bioenc.tokenize_dna(seq, k=3, stride=3, reading_frame=2)

        np.testing.assert_array_equal(frames[0], f0)
        np.testing.assert_array_equal(frames[1], f1)
        np.testing.assert_array_equal(frames[2], f2)

    def test_all_frames_short_sequence(self):
        """Test all_frames with sequence shorter than k."""
        seq = np.frombuffer(b"AT", dtype=np.uint8)
        frames = bioenc.tokenize_dna_all_frames(seq, k=3)
        assert len(frames) == 6
        # All frames should be empty
        for frame in frames:
            assert len(frame) == 0

    def test_all_frames_with_stride1(self):
        """Test all_frames works with stride=1 (overlapping)."""
        seq = np.frombuffer(b"ATGCGT", dtype=np.uint8)  # 6 bases
        frames = bioenc.tokenize_dna_all_frames(seq, k=3, stride=1)
        # Frame 0 (6 bases): ATG, TGC, GCG, CGT = 4 tokens
        # Frame 1 (5 bases): TGC, GCG, CGT = 3 tokens
        # Frame 2 (4 bases): GCG, CGT = 2 tokens
        assert len(frames[0]) == 4
        assert len(frames[1]) == 3
        assert len(frames[2]) == 2


class TestBatchAllFrames:
    """Tests for batch_tokenize_dna_all_frames."""

    def _make_batch(self, sequences):
        """Helper to create batch format from list of sequences."""
        buffer = b"".join(sequences)
        offsets = []
        lengths = []
        offset = 0
        for seq in sequences:
            offsets.append(offset)
            lengths.append(len(seq))
            offset += len(seq)
        return (
            np.frombuffer(buffer, dtype=np.uint8),
            np.array(lengths, dtype=np.int64),
        )

    def test_batch_all_frames_structure(self):
        """Test structure of batch_all_frames output."""
        sequences = [b"ATGCGTAAA", b"TTTGGGCCC"]
        buffer, lengths = self._make_batch(sequences)

        all_frames = bioenc.batch_tokenize_dna_all_frames(
            buffer, lengths, k=3
        )

        # Should be list of lists
        assert isinstance(all_frames, list)
        assert len(all_frames) == 2  # 2 sequences
        assert len(all_frames[0]) == 6  # 6 frames per sequence
        assert len(all_frames[1]) == 6

    def test_batch_all_frames_values(self):
        """Test batch_all_frames produces correct tokens."""
        sequences = [b"ATGCGTAAA"]
        buffer, lengths = self._make_batch(sequences)

        batch_frames = bioenc.batch_tokenize_dna_all_frames(
            buffer, lengths, k=3
        )

        # Should match single-sequence all_frames
        single_frames = bioenc.tokenize_dna_all_frames(
            np.frombuffer(sequences[0], dtype=np.uint8), k=3
        )

        for i in range(6):
            np.testing.assert_array_equal(batch_frames[0][i], single_frames[i])

    def test_batch_all_frames_multiple_seqs(self):
        """Test batch_all_frames with multiple sequences."""
        sequences = [b"ATGCGTAAA", b"TTTGGGCCC", b"AAACCCGGG"]
        buffer, lengths = self._make_batch(sequences)

        all_frames = bioenc.batch_tokenize_dna_all_frames(
            buffer, lengths, k=3, stride=3
        )

        assert len(all_frames) == 3
        # Each sequence should have 3 codons in frame 0
        assert len(all_frames[0][0]) == 3
        assert len(all_frames[1][0]) == 3
        assert len(all_frames[2][0]) == 3

    def test_batch_all_frames_with_padding(self):
        """Test batch_all_frames with padding enabled."""
        sequences = [b"ATGCGTAAA", b"TTT"]
        buffer, lengths = self._make_batch(sequences)

        all_frames = bioenc.batch_tokenize_dna_all_frames(
            buffer, lengths, k=3, stride=3,
            enable_padding=True
        )

        # Should return list of 2D arrays
        assert isinstance(all_frames, list)
        assert len(all_frames) == 2
        assert isinstance(all_frames[0], np.ndarray)
        assert all_frames[0].ndim == 2
        assert all_frames[0].shape[0] == 6  # 6 frames
        # Max length should be 3 (from first sequence)
        assert all_frames[0].shape[1] == 3

    def test_batch_all_frames_empty_sequence(self):
        """Test batch_all_frames with empty sequence."""
        sequences = [b"AT"]  # Too short for k=3
        buffer, lengths = self._make_batch(sequences)

        all_frames = bioenc.batch_tokenize_dna_all_frames(
            buffer, lengths, k=3
        )

        assert len(all_frames) == 1
        assert len(all_frames[0]) == 6
        # All frames should be empty
        for frame in all_frames[0]:
            assert len(frame) == 0


class TestReverseComplement:
    """Tests for reverse complement computation."""

    def test_basic_revcomp(self):
        """Test basic reverse complement."""
        seq = np.frombuffer(b"ACGT", dtype=np.uint8)
        rc = bioenc.reverse_complement_dna(seq)
        expected = np.frombuffer(b"ACGT", dtype=np.uint8)  # ACGT is its own RC
        np.testing.assert_array_equal(rc, expected)

    def test_asymmetric_revcomp(self):
        """Test reverse complement of non-palindromic sequence."""
        seq = np.frombuffer(b"AAACCC", dtype=np.uint8)
        rc = bioenc.reverse_complement_dna(seq)
        expected = np.frombuffer(b"GGGTTT", dtype=np.uint8)
        np.testing.assert_array_equal(rc, expected)

    def test_single_base(self):
        """Test reverse complement of single bases."""
        for base, comp in [(b"A", b"T"), (b"C", b"G"), (b"G", b"C"), (b"T", b"A")]:
            seq = np.frombuffer(base, dtype=np.uint8)
            rc = bioenc.reverse_complement_dna(seq)
            expected = np.frombuffer(comp, dtype=np.uint8)
            np.testing.assert_array_equal(rc, expected)


class TestTokenizeAa:
    """Tests for single sequence amino acid tokenization."""

    def test_basic_aa(self):
        """Test basic amino acid tokenization."""
        seq = np.frombuffer(b"ACDEFGHIK", dtype=np.uint8)
        tokens = bioenc.tokenize_aa(seq, k=3, stride=1)
        # 9 - 3 + 1 = 7 tokens
        assert len(tokens) == 7
        assert tokens.dtype == np.int64

    def test_aa_stride(self):
        """Test AA tokenization with stride > 1."""
        seq = np.frombuffer(b"ACDEFGHIKL", dtype=np.uint8)
        tokens = bioenc.tokenize_aa(seq, k=3, stride=2)
        # (10 - 3) / 2 + 1 = 4 tokens
        assert len(tokens) == 4

    def test_aa_short_sequence(self):
        """Test AA sequence shorter than k."""
        seq = np.frombuffer(b"AC", dtype=np.uint8)
        tokens = bioenc.tokenize_aa(seq, k=3)
        assert len(tokens) == 0  # Too short

    def test_aa_exact_k(self):
        """Test AA sequence exactly length k."""
        seq = np.frombuffer(b"ACD", dtype=np.uint8)
        tokens = bioenc.tokenize_aa(seq, k=3)
        assert len(tokens) == 1

    def test_aa_vocabulary_size(self):
        """Test that AA uses base=29 correctly."""
        # A=2, C=3, base=29
        # AC = 2*29 + 3 = 61
        seq = np.frombuffer(b"AC", dtype=np.uint8)
        tokens = bioenc.tokenize_aa(seq, k=2)
        assert tokens[0] == 61

    def test_aa_gap_character(self):
        """Test gap character (-) maps to 28."""
        # Gap (-) = 28, base=29
        # For k=2: 28*29 + 28 = 840
        seq = np.frombuffer(b"--", dtype=np.uint8)
        tokens = bioenc.tokenize_aa(seq, k=2)
        assert tokens[0] == 28 * 29 + 28  # 840

    def test_aa_stop_codon(self):
        """Test stop codon (*) maps to 27."""
        # Stop (*) = 27, A = 2, base=29
        # *A = 27*29 + 2 = 785
        seq = np.frombuffer(b"*A", dtype=np.uint8)
        tokens = bioenc.tokenize_aa(seq, k=2)
        assert tokens[0] == 27 * 29 + 2  # 785

    def test_aa_lowercase(self):
        """Test that lowercase amino acids work."""
        seq_upper = np.frombuffer(b"ACD", dtype=np.uint8)
        seq_lower = np.frombuffer(b"acd", dtype=np.uint8)
        tokens_upper = bioenc.tokenize_aa(seq_upper, k=2)
        tokens_lower = bioenc.tokenize_aa(seq_lower, k=2)
        np.testing.assert_array_equal(tokens_upper, tokens_lower)

    def test_aa_unknown_maps_to_x(self):
        """Test that unknown characters map to UNK/X=1."""
        # Unknown characters should map to UNK/X=1
        seq = np.frombuffer(b"A1", dtype=np.uint8)  # '1' is unknown
        tokens = bioenc.tokenize_aa(seq, k=2)
        # A=2, 1->UNK=1
        # A1 = 2*29 + 1 = 59
        assert tokens[0] == 59

    def test_aa_empty_sequence(self):
        """Test tokenization of empty sequence."""
        seq = np.frombuffer(b"", dtype=np.uint8)
        tokens = bioenc.tokenize_aa(seq, k=3)
        assert len(tokens) == 0

    def test_aa_matches_batch_single(self):
        """Test that tokenize_aa matches batch function for single sequence."""
        seq = np.frombuffer(b"ACDEFGHIK", dtype=np.uint8)

        # Single-sequence function
        tokens1 = bioenc.tokenize_aa(seq, k=3, stride=1)

        # Batch function with 1 sequence (variable-length by default)
        buffer = seq
        offsets = np.array([0], dtype=np.int64)
        lengths = np.array([len(seq)], dtype=np.int64)
        tokens2_list = bioenc.batch_tokenize_aa_shared(
            buffer, lengths, k=3, stride=1
        )
        # Extract first (and only) sequence from list
        tokens2 = tokens2_list[0]

        np.testing.assert_array_equal(tokens1, tokens2)


class TestBatchTokenizeDnaShared:
    """Tests for batch DNA tokenization."""

    def _make_batch(self, sequences):
        """Helper to create batch format from list of sequences."""
        buffer = b"".join(sequences)
        offsets = []
        lengths = []
        offset = 0
        for seq in sequences:
            offsets.append(offset)
            lengths.append(len(seq))
            offset += len(seq)
        return (
            np.frombuffer(buffer, dtype=np.uint8),
            np.array(lengths, dtype=np.int64),
        )

    def test_basic_batch(self):
        """Test basic batch tokenization with padding."""
        buf, lens = self._make_batch([b"ACGT", b"AAAA"])
        tokens = bioenc.batch_tokenize_dna_shared(
            buf, lens, k=2, stride=1,
            enable_padding=True, max_len=4
        )
        assert tokens.shape == (2, 4)
        # First seq: AC=15, CG=22, GT=29, pad=0
        # Second seq: AA=14, AA=14, AA=14, pad=0
        np.testing.assert_array_equal(tokens[0], [15, 22, 29, 0])
        np.testing.assert_array_equal(tokens[1], [14, 14, 14, 0])

    def test_variable_length_default(self):
        """Test that variable-length is now the default."""
        buf, lens = self._make_batch([b"ACGTACGT"])
        tokens = bioenc.batch_tokenize_dna_shared(
            buf, lens, k=2, stride=1
        )
        # Should get list of arrays
        assert isinstance(tokens, list)
        assert len(tokens) == 1
        # Should get all 7 tokens: AC=15, CG=22, GT=29, TA=32, AC=15, CG=22, GT=29
        assert len(tokens[0]) == 7

    def test_crop_explicit_start(self):
        """Test explicit start positions using crop_and_tokenize."""
        buf, lens = self._make_batch([b"ACGTACGT"])
        crop_starts = np.array([2], dtype=np.int64)  # Start at position 2
        crop_lengths = np.array([6], dtype=np.int64)  # Take 6 bases
        tokens = bioenc.crop_and_tokenize_dna(
            buf, lens, crop_starts, crop_lengths,
            k=2, stride=1
        )
        # Starting at position 2 (GTACGT): GT=29, TA=32, AC=15, CG=22, GT=29
        # TA = 5*6 + 2 = 32
        assert isinstance(tokens, list)
        np.testing.assert_array_equal(tokens[0], [29, 32, 15, 22, 29])

    def test_crop_per_sequence(self):
        """Test different start positions per sequence using crop_and_tokenize."""
        buf, lens = self._make_batch([b"ACGTACGT", b"TTTTAAAA"])
        crop_starts = np.array([0, 2], dtype=np.int64)  # First seq at 0, second at 2
        crop_lengths = np.array([6, 6], dtype=np.int64)
        tokens = bioenc.crop_and_tokenize_dna(
            buf, lens, crop_starts, crop_lengths,
            k=2, stride=1
        )
        # First seq (ACGTAC): AC=15, CG=22, GT=29, TA=32, AC=15
        np.testing.assert_array_equal(tokens[0], [15, 22, 29, 32, 15])
        # Second seq starting at 2 (TTAAAA): TT=35, TA=32, AA=14, AA=14, AA=14
        # TA = 5*6 + 2 = 32
        np.testing.assert_array_equal(tokens[1], [35, 32, 14, 14, 14])

    def test_padding(self):
        """Test that short sequences are padded."""
        buf, lens = self._make_batch([b"AC"])  # Only 1 2-mer possible
        tokens = bioenc.batch_tokenize_dna_shared(
            buf, lens, k=2, stride=1,
            enable_padding=True, max_len=5, pad_value=-99
        )
        assert tokens[0, 0] == 15  # AC = 2*6 + 3 = 15
        assert all(tokens[0, 1:] == -99)  # Rest padded

    def test_canonical_strand(self):
        """Test canonical strand in batch mode."""
        buf, lens = self._make_batch([b"ACGT"])
        tokens = bioenc.batch_tokenize_dna_shared(
            buf, lens, k=2, stride=1,
            enable_padding=True, max_len=4, strand="canonical"
        )
        # AC(15) vs GT(29) -> 15
        # CG(22) vs CG(22) -> 22
        # GT(29) vs AC(15) -> 15
        np.testing.assert_array_equal(tokens[0, :3], [15, 22, 15])


class TestBatchTokenizeDnaBoth:
    """Tests for batch DNA tokenization with both strands."""

    def _make_batch(self, sequences):
        """Helper to create batch format from list of sequences."""
        buffer = b"".join(sequences)
        offsets = []
        lengths = []
        offset = 0
        for seq in sequences:
            offsets.append(offset)
            lengths.append(len(seq))
            offset += len(seq)
        return (
            np.frombuffer(buffer, dtype=np.uint8),
            np.array(lengths, dtype=np.int64),
        )

    def test_both_returns_tuple(self):
        """Test that both strand mode returns a tuple of two arrays (with padding)."""
        buf, lens = self._make_batch([b"ACGT"])
        result = bioenc.batch_tokenize_dna_both(
            buf, lens, k=2, stride=1,
            enable_padding=True, max_len=4
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        fwd, rev = result
        assert fwd.shape == rev.shape == (1, 4)

    def test_both_values(self):
        """Test forward and revcomp values in both mode."""
        buf, lens = self._make_batch([b"ACGT"])
        fwd, rev = bioenc.batch_tokenize_dna_both(
            buf, lens, k=2, stride=1,
            enable_padding=True, max_len=4
        )
        # Forward: AC=15, CG=22, GT=29
        np.testing.assert_array_equal(fwd[0, :3], [15, 22, 29])
        # Revcomp: GT=29, CG=22, AC=15
        np.testing.assert_array_equal(rev[0, :3], [29, 22, 15])


class TestBatchTokenizeAaShared:
    """Tests for amino acid tokenization."""

    def _make_batch(self, sequences):
        """Helper to create batch format from list of sequences."""
        buffer = b"".join(sequences)
        offsets = []
        lengths = []
        offset = 0
        for seq in sequences:
            offsets.append(offset)
            lengths.append(len(seq))
            offset += len(seq)
        return (
            np.frombuffer(buffer, dtype=np.uint8),
            np.array(lengths, dtype=np.int64),
        )

    def test_basic_aa(self):
        """Test basic amino acid tokenization."""
        buf, lens = self._make_batch([b"ACDE"])
        tokens = bioenc.batch_tokenize_aa_shared(
            buf, lens, k=2, stride=1,
            enable_padding=True, max_len=4
        )
        # A=2, C=3, D=4, E=5, base=29
        # AC = 2*29 + 3 = 61
        # CD = 3*29 + 4 = 91
        # DE = 4*29 + 5 = 121
        np.testing.assert_array_equal(tokens[0, :3], [61, 91, 121])

    def test_aa_unknown_maps_to_x(self):
        """Test that unknown AA characters map to UNK/X=1."""
        buf, lens = self._make_batch([b"A1A"])  # '1' is unknown
        tokens = bioenc.batch_tokenize_aa_shared(
            buf, lens, k=2, stride=1,
            enable_padding=True, max_len=4
        )
        # A=2, 1->UNK=1
        # A1 = 2*29 + 1 = 59
        # 1A = 1*29 + 2 = 31
        np.testing.assert_array_equal(tokens[0, :2], [59, 31])

    def test_aa_crop(self):
        """Test crop_and_tokenize for AA."""
        buf, lens = self._make_batch([b"ACDEFGHIK"])
        crop_starts = np.array([2], dtype=np.int64)  # Start at position 2
        crop_lengths = np.array([6], dtype=np.int64)  # Take 6 AA
        tokens = bioenc.crop_and_tokenize_aa(
            buf, lens, crop_starts, crop_lengths,
            k=2, stride=1
        )
        assert isinstance(tokens, list)
        assert len(tokens) == 1
        # Starting at position 2 (DEFGHI): DE, EF, FG, GH, HI
        # D=4, E=5 -> DE = 4*29 + 5 = 121
        assert tokens[0][0] == 121
        assert len(tokens[0]) == 5  # 6 - 2 + 1 = 5


class TestHashTokens:
    """Tests for token hashing."""

    def test_basic_hash(self):
        """Test basic token hashing."""
        tokens = np.array([1, 100, 1000], dtype=np.int64)
        hashed = bioenc.hash_tokens(tokens, num_buckets=50)
        assert hashed.shape == tokens.shape
        assert all(0 <= h < 50 for h in hashed)

    def test_hash_preserves_padding(self):
        """Test that zero and negative values (padding) are preserved."""
        tokens = np.array([0, 100, -1, -99], dtype=np.int64)
        hashed = bioenc.hash_tokens(tokens, num_buckets=50)
        assert hashed[0] == 0   # 0 (padding) preserved
        assert hashed[2] == -1
        assert hashed[3] == -99

    def test_hash_2d(self):
        """Test hashing 2D token arrays."""
        tokens = np.array([[0, 1, 2], [10, 20, -1]], dtype=np.int64)
        hashed = bioenc.hash_tokens(tokens, num_buckets=5)
        assert hashed.shape == tokens.shape
        assert hashed[0, 0] == 0  # Padding preserved
        assert hashed[1, 2] == -1  # Negative preserved

    def test_hash_deterministic(self):
        """Test that hashing is deterministic."""
        tokens = np.array([42, 123, 456], dtype=np.int64)
        h1 = bioenc.hash_tokens(tokens, num_buckets=100)
        h2 = bioenc.hash_tokens(tokens, num_buckets=100)
        np.testing.assert_array_equal(h1, h2)


class TestVariableLengthOutput:
    """Tests for variable-length output (default behavior)."""

    def _make_batch(self, sequences):
        """Helper to create batch format from list of sequences."""
        buffer = b"".join(sequences)
        offsets = []
        lengths = []
        offset = 0
        for seq in sequences:
            offsets.append(offset)
            lengths.append(len(seq))
            offset += len(seq)
        return (
            np.frombuffer(buffer, dtype=np.uint8),
            np.array(lengths, dtype=np.int64),
        )

    def test_variable_length_basic(self):
        """Test default variable-length output for DNA."""
        seqs = [b"ACGTACGT", b"ACGTACGTACGT", b"ACG"]
        buffer, lengths = self._make_batch(seqs)

        tokens = bioenc.batch_tokenize_dna_shared(
            buffer, lengths, k=3, stride=1
        )

        # Should return list of arrays
        assert isinstance(tokens, list)
        assert len(tokens) == 3
        assert isinstance(tokens[0], np.ndarray)

        # Check lengths
        assert len(tokens[0]) == 8 - 3 + 1  # 6 tokens
        assert len(tokens[1]) == 12 - 3 + 1  # 10 tokens
        assert len(tokens[2]) == 1  # 1 token (exactly k=3)

    def test_enable_padding_auto_max(self):
        """Test rectangular output with auto-calculated max_len."""
        seqs = [b"ACGTACGT", b"ACGTACGTACGT"]
        buffer, lengths = self._make_batch(seqs)

        tokens = bioenc.batch_tokenize_dna_shared(
            buffer, lengths, k=3, stride=1,
            enable_padding=True
        )

        # Should return 2D array
        assert isinstance(tokens, np.ndarray)
        assert tokens.ndim == 2
        assert tokens.shape == (2, 10)  # Max is 10 tokens from second seq

        # First sequence should be padded with 0
        assert np.all(tokens[0, 6:] == 0)  # Last 4 positions padded

    def test_pad_to_max_explicit(self):
        """Test rectangular output with explicit pad_to_max."""
        seqs = [b"ACGTACGT"]
        buffer, lengths = self._make_batch(seqs)

        tokens = bioenc.batch_tokenize_dna_shared(
            buffer, lengths, k=3, stride=1,
            enable_padding=True,
            max_len=20
        )

        assert tokens.shape == (1, 20)
        assert np.sum(tokens[0] > 0) == 6  # Only 6 valid tokens (all > 0)
        assert np.sum(tokens[0] == 0) == 14  # 14 padding tokens

    def test_variable_length_aa(self):
        """Test variable-length output for amino acids."""
        seqs = [b"ACDEF", b"ACDEFGHIKL"]
        buffer, lengths = self._make_batch(seqs)

        tokens = bioenc.batch_tokenize_aa_shared(
            buffer, lengths, k=3, stride=1
        )

        # Should return list of arrays
        assert isinstance(tokens, list)
        assert len(tokens) == 2
        assert len(tokens[0]) == 5 - 3 + 1  # 3 tokens
        assert len(tokens[1]) == 10 - 3 + 1  # 8 tokens

    def test_variable_length_both_strands(self):
        """Test variable-length output for both strands."""
        seqs = [b"ACGT", b"ACGTACGT"]
        buffer, lengths = self._make_batch(seqs)

        fwd, rev = bioenc.batch_tokenize_dna_both(
            buffer, lengths, k=2, stride=1
        )

        # Should return tuple of lists
        assert isinstance(fwd, list)
        assert isinstance(rev, list)
        assert len(fwd) == 2
        assert len(rev) == 2
        assert len(fwd[0]) == 3  # 4-2+1=3
        assert len(fwd[1]) == 7  # 8-2+1=7

    def test_padded_both_strands(self):
        """Test padded output for both strands."""
        seqs = [b"ACGT", b"ACGTACGT"]
        buffer, lengths = self._make_batch(seqs)

        fwd, rev = bioenc.batch_tokenize_dna_both(
            buffer, lengths, k=2, stride=1,
            enable_padding=True
        )

        # Should return tuple of 2D arrays
        assert isinstance(fwd, np.ndarray)
        assert isinstance(rev, np.ndarray)
        assert fwd.shape == (2, 7)  # Max is 7 from second seq
        assert rev.shape == (2, 7)


class TestCropAndTokenize:
    """Tests for crop_and_tokenize functions."""

    def _make_batch(self, sequences):
        """Helper to create batch format from list of sequences."""
        buffer = b"".join(sequences)
        offsets = []
        lengths = []
        offset = 0
        for seq in sequences:
            offsets.append(offset)
            lengths.append(len(seq))
            offset += len(seq)
        return (
            np.frombuffer(buffer, dtype=np.uint8),
            np.array(lengths, dtype=np.int64),
        )

    def test_crop_basic(self):
        """Test basic cropping and tokenization."""
        seq = b"ATGCGTAAATGATAG"  # 15 bp
        buffer, lengths = self._make_batch([seq])

        # Crop positions 3-12 (10 bp)
        crop_starts = np.array([3], dtype=np.int64)
        crop_lengths = np.array([10], dtype=np.int64)

        tokens = bioenc.crop_and_tokenize_dna(
            buffer, lengths,
            crop_starts, crop_lengths,
            k=3, stride=1
        )

        # Should return list with one array
        assert isinstance(tokens, list)
        assert len(tokens) == 1
        assert len(tokens[0]) == 10 - 3 + 1  # 8 tokens

    def test_crop_reading_frames(self):
        """Test extracting 3 reading frames using crop_and_tokenize."""
        seq = b"ATGCGTAAATGATAG"  # 15 bp
        # Replicate buffer for 3 frames
        buffer = np.tile(np.frombuffer(seq, dtype=np.uint8), 3)
        lengths = np.array([15, 15, 15], dtype=np.int64)

        # Crop to start at different positions, length = 12 (multiple of 3)
        crop_starts = np.array([0, 1, 2], dtype=np.int64)
        crop_lengths = np.array([12, 12, 12], dtype=np.int64)

        tokens_frames = bioenc.crop_and_tokenize_dna(
            buffer, lengths,
            crop_starts, crop_lengths,
            k=3, stride=3  # Non-overlapping codons
        )

        # Should return 3 sequences with different tokens
        assert len(tokens_frames) == 3
        assert len(tokens_frames[0]) == 4  # 12 / 3 = 4 codons
        assert len(tokens_frames[1]) == 4
        assert len(tokens_frames[2]) == 4

        # Frames should be different
        assert not np.array_equal(tokens_frames[0], tokens_frames[1])
        assert not np.array_equal(tokens_frames[1], tokens_frames[2])

    def test_crop_validation(self):
        """Test crop_and_tokenize validates boundaries."""
        seq = b"ATGCGTAAA"
        buffer, lengths = self._make_batch([seq])
        crop_starts = np.array([0], dtype=np.int64)
        crop_lengths = np.array([20], dtype=np.int64)  # Too long!

        with pytest.raises(RuntimeError, match="crop.*extends beyond sequence boundary"):
            bioenc.crop_and_tokenize_dna(
                buffer, lengths,
                crop_starts, crop_lengths,
                k=3, stride=1
            )

    def test_crop_aa(self):
        """Test cropping amino acid sequences."""
        seq = b"ACDEFGHIKL"  # 10 AA
        buffer, lengths = self._make_batch([seq])

        # Crop positions 2-8 (6 AA)
        crop_starts = np.array([2], dtype=np.int64)
        crop_lengths = np.array([6], dtype=np.int64)

        tokens = bioenc.crop_and_tokenize_aa(
            buffer, lengths,
            crop_starts, crop_lengths,
            k=2, stride=1
        )

        # Should return list with one array
        assert isinstance(tokens, list)
        assert len(tokens) == 1
        assert len(tokens[0]) == 6 - 2 + 1  # 5 tokens

    def test_crop_with_padding(self):
        """Test crop_and_tokenize with padding enabled."""
        seqs = [b"ATGCGTAAA", b"ATGCGTAAATGATAG"]
        buffer, lengths = self._make_batch(seqs)

        # Crop first 9 bp from each
        crop_starts = np.array([0, 0], dtype=np.int64)
        crop_lengths = np.array([9, 9], dtype=np.int64)

        tokens = bioenc.crop_and_tokenize_dna(
            buffer, lengths,
            crop_starts, crop_lengths,
            k=3, stride=1,
            enable_padding=True
        )

        # Should return 2D array since both cropped to same length
        assert isinstance(tokens, np.ndarray)
        assert tokens.shape[0] == 2
        # Both should produce 7 tokens from 9 bp
        assert tokens.shape[1] == 7


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_k_equals_seq_len(self):
        """Test when k equals sequence length."""
        seq = np.frombuffer(b"ACG", dtype=np.uint8)
        tokens = bioenc.tokenize_dna(seq, k=3)
        assert len(tokens) == 1

    def test_very_short_with_stride(self):
        """Test short sequence with large stride."""
        seq = np.frombuffer(b"ACGT", dtype=np.uint8)
        tokens = bioenc.tokenize_dna(seq, k=2, stride=3)
        # Only position 0 fits
        assert len(tokens) == 1

    def test_iupac_alphabet(self):
        """Test IUPAC alphabet handling."""
        seq = np.frombuffer(b"ACGTRYWSKMBDHVN", dtype=np.uint8)
        tokens = bioenc.tokenize_dna(seq, k=2, alphabet="iupac")
        # Should not crash, tokens should be in range (0, 16^2)
        assert all(0 < t < 256 for t in tokens)

    def test_batch_strand_both_error(self):
        """Test that batch_tokenize_dna_shared raises error for strand='both'."""
        buf = np.frombuffer(b"ACGT", dtype=np.uint8)
        off = np.array([0], dtype=np.int64)
        lens = np.array([4], dtype=np.int64)
        with pytest.raises(RuntimeError, match="batch_tokenize_dna_both"):
            bioenc.batch_tokenize_dna_shared(
                buf, lens, k=2, stride=1, strand="both"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
