"""Benchmark tests comparing bioenc C++ implementation vs pure Python."""

import time
import numpy as np
import pytest
import bioenc


# ============================================================================
# Pure Python reference implementations
# ============================================================================

DNA_TABLE_PY = {
    ord('A'): 0, ord('a'): 0,
    ord('C'): 1, ord('c'): 1,
    ord('G'): 2, ord('g'): 2,
    ord('T'): 3, ord('t'): 3,
    ord('U'): 3, ord('u'): 3,
}
DNA_COMP_PY = {0: 3, 1: 2, 2: 1, 3: 0, 4: 4}  # A<->T, C<->G, N<->N

AA_TABLE_PY = {
    ord('A'): 0, ord('a'): 0, ord('C'): 1, ord('c'): 1,
    ord('D'): 2, ord('d'): 2, ord('E'): 3, ord('e'): 3,
    ord('F'): 4, ord('f'): 4, ord('G'): 5, ord('g'): 5,
    ord('H'): 6, ord('h'): 6, ord('I'): 7, ord('i'): 7,
    ord('K'): 8, ord('k'): 8, ord('L'): 9, ord('l'): 9,
    ord('M'): 10, ord('m'): 10, ord('N'): 11, ord('n'): 11,
    ord('P'): 12, ord('p'): 12, ord('Q'): 13, ord('q'): 13,
    ord('R'): 14, ord('r'): 14, ord('S'): 15, ord('s'): 15,
    ord('T'): 16, ord('t'): 16, ord('V'): 17, ord('v'): 17,
    ord('W'): 18, ord('w'): 18, ord('Y'): 19, ord('y'): 19,
}


def tokenize_dna_python(seq: np.ndarray, k: int, stride: int = 1) -> np.ndarray:
    """Pure Python DNA tokenization."""
    if len(seq) < k:
        return np.array([], dtype=np.int64)

    base = 5
    tokens = []
    for i in range(0, len(seq) - k + 1, stride):
        value = 0
        for j in range(k):
            code = DNA_TABLE_PY.get(seq[i + j], 4)  # N=4 for unknown
            value = value * base + code
        tokens.append(value)
    return np.array(tokens, dtype=np.int64)


def tokenize_dna_canonical_python(seq: np.ndarray, k: int, stride: int = 1) -> np.ndarray:
    """Pure Python canonical DNA tokenization."""
    if len(seq) < k:
        return np.array([], dtype=np.int64)

    base = 5
    tokens = []
    for i in range(0, len(seq) - k + 1, stride):
        fwd = 0
        rev = 0
        for j in range(k):
            code = DNA_TABLE_PY.get(seq[i + j], 4)
            comp = DNA_COMP_PY[code]
            fwd = fwd * base + code
            rev = rev + comp * (base ** j)
        tokens.append(min(fwd, rev))
    return np.array(tokens, dtype=np.int64)


def reverse_complement_python(seq: np.ndarray) -> np.ndarray:
    """Pure Python reverse complement."""
    bases = "ACGTN"
    result = []
    for i in range(len(seq) - 1, -1, -1):
        code = DNA_TABLE_PY.get(seq[i], 4)
        comp = DNA_COMP_PY[code]
        result.append(ord(bases[comp]))
    return np.array(result, dtype=np.uint8)


def batch_tokenize_dna_python(
    buffer: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    k: int,
    max_len: int,
    pad_value: int = -1,
    stride: int = 1,
) -> np.ndarray:
    """Pure Python batch DNA tokenization."""
    num_seqs = len(offsets)
    result = np.full((num_seqs, max_len), pad_value, dtype=np.int64)
    base = 5

    for seq_idx in range(num_seqs):
        offset = offsets[seq_idx]
        length = lengths[seq_idx]
        seq = buffer[offset:offset + length]

        if length < k:
            continue

        out_idx = 0
        for i in range(0, length - k + 1, stride):
            if out_idx >= max_len:
                break
            value = 0
            for j in range(k):
                code = DNA_TABLE_PY.get(seq[i + j], 4)
                value = value * base + code
            result[seq_idx, out_idx] = value
            out_idx += 1

    return result


def batch_tokenize_aa_python(
    buffer: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    k: int,
    max_len: int,
    pad_value: int = -1,
    stride: int = 1,
) -> np.ndarray:
    """Pure Python batch AA tokenization."""
    num_seqs = len(offsets)
    result = np.full((num_seqs, max_len), pad_value, dtype=np.int64)
    base = 28

    for seq_idx in range(num_seqs):
        offset = offsets[seq_idx]
        length = lengths[seq_idx]
        seq = buffer[offset:offset + length]

        if length < k:
            continue

        out_idx = 0
        for i in range(0, length - k + 1, stride):
            if out_idx >= max_len:
                break
            value = 0
            for j in range(k):
                code = AA_TABLE_PY.get(seq[i + j], 25)  # X=25 for unknown
                value = value * base + code
            result[seq_idx, out_idx] = value
            out_idx += 1

    return result


def hash_tokens_python(tokens: np.ndarray, num_buckets: int) -> np.ndarray:
    """Pure Python token hashing (MurmurHash3-style)."""
    result = np.empty_like(tokens)
    flat_in = tokens.ravel()
    flat_out = result.ravel()

    for i in range(len(flat_in)):
        val = flat_in[i]
        if val < 0:
            flat_out[i] = val  # preserve padding
        else:
            # MurmurHash3 finalizer
            key = np.uint64(val)
            key ^= key >> 33
            key = np.uint64(key * 0xff51afd7ed558ccd)
            key ^= key >> 33
            key = np.uint64(key * 0xc4ceb9fe1a85ec53)
            key ^= key >> 33
            flat_out[i] = int(key % num_buckets)

    return result


# ============================================================================
# Benchmark utilities
# ============================================================================

def generate_random_dna(length: int, seed: int = 42) -> np.ndarray:
    """Generate random DNA sequence."""
    rng = np.random.default_rng(seed)
    bases = np.array([ord('A'), ord('C'), ord('G'), ord('T')], dtype=np.uint8)
    return rng.choice(bases, size=length)


def generate_random_aa(length: int, seed: int = 42) -> np.ndarray:
    """Generate random amino acid sequence."""
    rng = np.random.default_rng(seed)
    aas = np.array([ord(c) for c in "ACDEFGHIKLMNPQRSTVWY"], dtype=np.uint8)
    return rng.choice(aas, size=length)


def make_batch(sequences: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create batch format from list of sequences."""
    buffer = np.concatenate(sequences)
    offsets = np.zeros(len(sequences), dtype=np.int64)
    lengths = np.array([len(s) for s in sequences], dtype=np.int64)
    offset = 0
    for i, seq in enumerate(sequences):
        offsets[i] = offset
        offset += len(seq)
    return buffer, offsets, lengths


class BenchmarkResult:
    """Store and display benchmark results."""

    def __init__(self, name: str, python_time: float, cpp_time: float, size_info: str):
        self.name = name
        self.python_time = python_time
        self.cpp_time = cpp_time
        self.size_info = size_info
        self.speedup = python_time / cpp_time if cpp_time > 0 else float('inf')

    def __str__(self):
        return (
            f"{self.name} ({self.size_info}):\n"
            f"  Python: {self.python_time*1000:.3f} ms\n"
            f"  C++:    {self.cpp_time*1000:.3f} ms\n"
            f"  Speedup: {self.speedup:.1f}x"
        )


def benchmark(func_python, func_cpp, args_python, args_cpp, n_runs: int = 5):
    """Run benchmark comparing Python and C++ implementations."""
    # Warmup
    func_python(*args_python)
    func_cpp(*args_cpp)

    # Time Python
    start = time.perf_counter()
    for _ in range(n_runs):
        func_python(*args_python)
    python_time = (time.perf_counter() - start) / n_runs

    # Time C++
    start = time.perf_counter()
    for _ in range(n_runs):
        func_cpp(*args_cpp)
    cpp_time = (time.perf_counter() - start) / n_runs

    return python_time, cpp_time


# ============================================================================
# Benchmark tests
# ============================================================================

class TestBenchmarkSingleSequence:
    """Benchmarks for single sequence tokenization."""

    @pytest.mark.parametrize("seq_len", [100, 1000, 10000, 100000])
    def test_tokenize_dna_forward(self, seq_len):
        """Benchmark single DNA tokenization (forward strand)."""
        seq = generate_random_dna(seq_len)
        k = 6

        python_time, cpp_time = benchmark(
            tokenize_dna_python, bioenc.tokenize_dna,
            (seq, k), (seq, k),
            n_runs=10 if seq_len < 10000 else 3
        )

        result = BenchmarkResult(
            "tokenize_dna (forward)",
            python_time, cpp_time,
            f"len={seq_len}, k={k}"
        )
        print(f"\n{result}")

        # Verify correctness
        py_tokens = tokenize_dna_python(seq, k)
        cpp_tokens = bioenc.tokenize_dna(seq, k)
        np.testing.assert_array_equal(py_tokens, cpp_tokens)

        assert result.speedup > 1, "C++ should be faster than Python"

    @pytest.mark.parametrize("seq_len", [100, 1000, 10000])
    def test_tokenize_dna_canonical(self, seq_len):
        """Benchmark canonical strand tokenization."""
        seq = generate_random_dna(seq_len)
        k = 6

        python_time, cpp_time = benchmark(
            tokenize_dna_canonical_python,
            lambda s, k: bioenc.tokenize_dna(s, k, strand="canonical"),
            (seq, k), (seq, k),
            n_runs=10 if seq_len < 10000 else 3
        )

        result = BenchmarkResult(
            "tokenize_dna (canonical)",
            python_time, cpp_time,
            f"len={seq_len}, k={k}"
        )
        print(f"\n{result}")

        # Verify correctness
        py_tokens = tokenize_dna_canonical_python(seq, k)
        cpp_tokens = bioenc.tokenize_dna(seq, k, strand="canonical")
        np.testing.assert_array_equal(py_tokens, cpp_tokens)

        assert result.speedup > 1, "C++ should be faster than Python"

    @pytest.mark.parametrize("seq_len", [100, 1000, 10000, 100000])
    def test_reverse_complement(self, seq_len):
        """Benchmark reverse complement computation."""
        seq = generate_random_dna(seq_len)

        python_time, cpp_time = benchmark(
            reverse_complement_python, bioenc.reverse_complement_dna,
            (seq,), (seq,),
            n_runs=10 if seq_len < 10000 else 3
        )

        result = BenchmarkResult(
            "reverse_complement",
            python_time, cpp_time,
            f"len={seq_len}"
        )
        print(f"\n{result}")

        # Verify correctness
        py_rc = reverse_complement_python(seq)
        cpp_rc = bioenc.reverse_complement_dna(seq)
        np.testing.assert_array_equal(py_rc, cpp_rc)

        assert result.speedup > 1, "C++ should be faster than Python"


class TestBenchmarkBatch:
    """Benchmarks for batch tokenization."""

    @pytest.mark.parametrize("num_seqs,seq_len", [
        (10, 100),
        (100, 100),
        (100, 1000),
        (1000, 100),
        (1000, 1000),
    ])
    def test_batch_tokenize_dna(self, num_seqs, seq_len):
        """Benchmark batch DNA tokenization."""
        sequences = [generate_random_dna(seq_len, seed=i) for i in range(num_seqs)]
        buffer, offsets, lengths = make_batch(sequences)
        k = 6
        max_len = seq_len - k + 1

        python_time, cpp_time = benchmark(
            batch_tokenize_dna_python,
            bioenc.batch_tokenize_dna_shared,
            (buffer, offsets, lengths, k, max_len),
            (buffer, offsets, lengths, k, max_len),
            n_runs=5 if num_seqs * seq_len < 100000 else 2
        )

        result = BenchmarkResult(
            "batch_tokenize_dna",
            python_time, cpp_time,
            f"n={num_seqs}, len={seq_len}, k={k}"
        )
        print(f"\n{result}")

        # Verify correctness
        py_tokens = batch_tokenize_dna_python(buffer, offsets, lengths, k, max_len)
        cpp_tokens = bioenc.batch_tokenize_dna_shared(buffer, offsets, lengths, k, max_len)
        np.testing.assert_array_equal(py_tokens, cpp_tokens)

        assert result.speedup > 1, "C++ should be faster than Python"

    @pytest.mark.parametrize("num_seqs,seq_len", [
        (100, 100),
        (100, 500),
        (500, 100),
    ])
    def test_batch_tokenize_aa(self, num_seqs, seq_len):
        """Benchmark batch amino acid tokenization."""
        sequences = [generate_random_aa(seq_len, seed=i) for i in range(num_seqs)]
        buffer, offsets, lengths = make_batch(sequences)
        k = 3
        max_len = seq_len - k + 1

        python_time, cpp_time = benchmark(
            batch_tokenize_aa_python,
            bioenc.batch_tokenize_aa_shared,
            (buffer, offsets, lengths, k, max_len),
            (buffer, offsets, lengths, k, max_len),
            n_runs=5
        )

        result = BenchmarkResult(
            "batch_tokenize_aa",
            python_time, cpp_time,
            f"n={num_seqs}, len={seq_len}, k={k}"
        )
        print(f"\n{result}")

        # Verify correctness
        py_tokens = batch_tokenize_aa_python(buffer, offsets, lengths, k, max_len)
        cpp_tokens = bioenc.batch_tokenize_aa_shared(buffer, offsets, lengths, k, max_len)
        np.testing.assert_array_equal(py_tokens, cpp_tokens)

        assert result.speedup > 1, "C++ should be faster than Python"


class TestBenchmarkHash:
    """Benchmarks for token hashing."""

    @pytest.mark.parametrize("num_tokens", [1000, 10000, 100000, 1000000])
    def test_hash_tokens(self, num_tokens):
        """Benchmark token hashing."""
        rng = np.random.default_rng(42)
        tokens = rng.integers(0, 1000000, size=num_tokens, dtype=np.int64)
        num_buckets = 50000

        python_time, cpp_time = benchmark(
            hash_tokens_python, bioenc.hash_tokens,
            (tokens, num_buckets), (tokens, num_buckets),
            n_runs=5 if num_tokens < 100000 else 2
        )

        result = BenchmarkResult(
            "hash_tokens",
            python_time, cpp_time,
            f"n={num_tokens}, buckets={num_buckets}"
        )
        print(f"\n{result}")

        # Verify correctness (both should produce same hash values)
        py_hashed = hash_tokens_python(tokens, num_buckets)
        cpp_hashed = bioenc.hash_tokens(tokens, num_buckets)
        np.testing.assert_array_equal(py_hashed, cpp_hashed)

        assert result.speedup > 1, "C++ should be faster than Python"


class TestBenchmarkBothStrands:
    """Benchmarks for both-strand tokenization."""

    @pytest.mark.parametrize("num_seqs,seq_len", [
        (100, 100),
        (100, 1000),
        (500, 500),
    ])
    def test_batch_tokenize_both(self, num_seqs, seq_len):
        """Benchmark batch DNA tokenization with both strands."""
        sequences = [generate_random_dna(seq_len, seed=i) for i in range(num_seqs)]
        buffer, offsets, lengths = make_batch(sequences)
        k = 6
        max_len = seq_len - k + 1

        def python_both(buf, off, lens, k, max_len):
            """Python implementation returning both strands."""
            base = 5
            num_seqs = len(off)
            fwd_result = np.full((num_seqs, max_len), -1, dtype=np.int64)
            rev_result = np.full((num_seqs, max_len), -1, dtype=np.int64)

            for seq_idx in range(num_seqs):
                offset = off[seq_idx]
                length = lens[seq_idx]
                seq = buf[offset:offset + length]

                if length < k:
                    continue

                out_idx = 0
                for i in range(0, length - k + 1):
                    if out_idx >= max_len:
                        break
                    fwd = 0
                    rev = 0
                    for j in range(k):
                        code = DNA_TABLE_PY.get(seq[i + j], 4)
                        comp = DNA_COMP_PY[code]
                        fwd = fwd * base + code
                        rev = rev + comp * (base ** j)
                    fwd_result[seq_idx, out_idx] = fwd
                    rev_result[seq_idx, out_idx] = rev
                    out_idx += 1

            return fwd_result, rev_result

        python_time, cpp_time = benchmark(
            python_both,
            bioenc.batch_tokenize_dna_both,
            (buffer, offsets, lengths, k, max_len),
            (buffer, offsets, lengths, k, max_len),
            n_runs=3
        )

        result = BenchmarkResult(
            "batch_tokenize_dna_both",
            python_time, cpp_time,
            f"n={num_seqs}, len={seq_len}, k={k}"
        )
        print(f"\n{result}")

        # Verify correctness
        py_fwd, py_rev = python_both(buffer, offsets, lengths, k, max_len)
        cpp_fwd, cpp_rev = bioenc.batch_tokenize_dna_both(buffer, offsets, lengths, k, max_len)
        np.testing.assert_array_equal(py_fwd, cpp_fwd)
        np.testing.assert_array_equal(py_rev, cpp_rev)

        assert result.speedup > 1, "C++ should be faster than Python"


# ============================================================================
# Summary benchmark
# ============================================================================

def test_benchmark_summary():
    """Run a comprehensive benchmark summary."""
    print("\n" + "=" * 70)
    print("BIOENC BENCHMARK SUMMARY")
    print("=" * 70)

    results = []

    # Single sequence DNA (various sizes)
    for seq_len in [1000, 10000, 100000]:
        seq = generate_random_dna(seq_len)
        k = 6
        py_time, cpp_time = benchmark(
            tokenize_dna_python, bioenc.tokenize_dna,
            (seq, k), (seq, k), n_runs=5
        )
        results.append(BenchmarkResult(
            "Single DNA", py_time, cpp_time, f"len={seq_len:,}"
        ))

    # Batch DNA
    for num_seqs, seq_len in [(100, 1000), (1000, 1000)]:
        sequences = [generate_random_dna(seq_len, seed=i) for i in range(num_seqs)]
        buffer, offsets, lengths = make_batch(sequences)
        k, max_len = 6, seq_len - 5
        py_time, cpp_time = benchmark(
            batch_tokenize_dna_python, bioenc.batch_tokenize_dna_shared,
            (buffer, offsets, lengths, k, max_len),
            (buffer, offsets, lengths, k, max_len), n_runs=3
        )
        results.append(BenchmarkResult(
            "Batch DNA", py_time, cpp_time, f"n={num_seqs}, len={seq_len}"
        ))

    # Batch AA
    sequences = [generate_random_aa(500, seed=i) for i in range(500)]
    buffer, offsets, lengths = make_batch(sequences)
    k, max_len = 3, 498
    py_time, cpp_time = benchmark(
        batch_tokenize_aa_python, bioenc.batch_tokenize_aa_shared,
        (buffer, offsets, lengths, k, max_len),
        (buffer, offsets, lengths, k, max_len), n_runs=3
    )
    results.append(BenchmarkResult(
        "Batch AA", py_time, cpp_time, "n=500, len=500"
    ))

    # Hash tokens
    tokens = np.random.default_rng(42).integers(0, 1000000, size=500000, dtype=np.int64)
    py_time, cpp_time = benchmark(
        hash_tokens_python, bioenc.hash_tokens,
        (tokens, 50000), (tokens, 50000), n_runs=3
    )
    results.append(BenchmarkResult(
        "Hash tokens", py_time, cpp_time, "n=500,000"
    ))

    # Print summary table
    print(f"\n{'Operation':<25} {'Size':<25} {'Python (ms)':<15} {'C++ (ms)':<15} {'Speedup':<10}")
    print("-" * 90)
    for r in results:
        print(f"{r.name:<25} {r.size_info:<25} {r.python_time*1000:<15.3f} {r.cpp_time*1000:<15.3f} {r.speedup:<10.1f}x")

    print("-" * 90)
    avg_speedup = sum(r.speedup for r in results) / len(results)
    print(f"{'Average speedup:':<52} {'':<30} {avg_speedup:.1f}x")
    print("=" * 70)

    # All should be faster
    for r in results:
        assert r.speedup > 1, f"{r.name} should be faster in C++"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
