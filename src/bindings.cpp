#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <stdexcept>
#include <optional>
#include "bioenc.hpp"

namespace py = pybind11;

using namespace bioenc;

// ============================================================================
// Input validation helpers
// ============================================================================

namespace validation {
    inline void validate_k(int k, const char* context) {
        if (k <= 0) {
            throw std::runtime_error(std::string(context) +
                ": k must be positive, got " + std::to_string(k));
        }
        if (k > 31) {
            throw std::runtime_error(std::string(context) +
                ": k > 31 may cause overflow, got " + std::to_string(k));
        }
    }

    inline void validate_stride(int stride, const char* context) {
        if (stride <= 0) {
            throw std::runtime_error(std::string(context) +
                ": stride must be positive, got " + std::to_string(stride));
        }
    }

    inline void validate_buffer_access(
        size_t buffer_size,
        const int64_t* offsets,
        const int64_t* lengths,
        size_t num_seqs,
        const char* context
    ) {
        for (size_t i = 0; i < num_seqs; i++) {
            if (offsets[i] < 0 || lengths[i] < 0) {
                throw std::runtime_error(std::string(context) +
                    ": negative offset or length at index " + std::to_string(i));
            }
            size_t end = static_cast<size_t>(offsets[i]) + static_cast<size_t>(lengths[i]);
            if (end > buffer_size) {
                throw std::runtime_error(std::string(context) +
                    ": sequence " + std::to_string(i) + " extends beyond buffer (offset=" +
                    std::to_string(offsets[i]) + " length=" + std::to_string(lengths[i]) +
                    " buffer_size=" + std::to_string(buffer_size) + ")");
            }
        }
    }
}

// ============================================================================
// Helper function to compute offsets from lengths (cumsum)
// ============================================================================

inline py::array_t<int64_t> compute_offsets_from_lengths(const py::array_t<int64_t>& lengths) {
    size_t num_seqs = static_cast<size_t>(lengths.size());
    py::array_t<int64_t> offsets(num_seqs);

    const int64_t* len_ptr = lengths.data();
    int64_t* off_ptr = offsets.mutable_data();

    int64_t cumsum = 0;
    for (size_t i = 0; i < num_seqs; i++) {
        off_ptr[i] = cumsum;
        cumsum += len_ptr[i];
    }

    return offsets;
}

// ============================================================================
// Internal helper for batch_tokenize_dna with explicit offsets
// (Used by crop_and_tokenize which needs custom offsets)
// ============================================================================

py::object batch_tokenize_dna_shared_internal(
    py::buffer buffer_buf,
    py::array_t<int64_t> offsets,
    py::array_t<int64_t> lengths,
    int k,
    int stride,
    std::optional<int> reading_frame,
    bool enable_padding,
    std::optional<size_t> max_len,
    int64_t pad_value,
    const std::string& alphabet,
    const std::string& strand)
{
    // This is the original implementation that accepts offsets explicitly
    py::buffer_info buf_info = buffer_buf.request();
    const uint8_t* buffer = static_cast<const uint8_t*>(buf_info.ptr);
    const int64_t* off_ptr = offsets.data();
    const int64_t* len_ptr = lengths.data();
    size_t num_seqs = static_cast<size_t>(offsets.size());
    size_t buffer_size = static_cast<size_t>(buf_info.size);

    // Apply reading frame offset to all sequences if specified
    py::array_t<int64_t> adjusted_offsets;
    py::array_t<int64_t> adjusted_lengths;

    if (reading_frame.has_value() && reading_frame.value() != 0) {
        int rf = reading_frame.value();
        adjusted_offsets = py::array_t<int64_t>(num_seqs);
        adjusted_lengths = py::array_t<int64_t>(num_seqs);
        int64_t* adj_off_ptr = adjusted_offsets.mutable_data();
        int64_t* adj_len_ptr = adjusted_lengths.mutable_data();

        for (size_t i = 0; i < num_seqs; i++) {
            adj_off_ptr[i] = off_ptr[i] + rf;
            adj_len_ptr[i] = (len_ptr[i] > rf) ? (len_ptr[i] - rf) : 0;
        }

        off_ptr = adj_off_ptr;
        len_ptr = adj_len_ptr;
    }

    validation::validate_buffer_access(buffer_size, off_ptr, len_ptr, num_seqs, "batch_tokenize_dna_shared");

    bool use_iupac = (alphabet == "iupac");
    StrandMode strand_mode = parse_strand(strand);

    if (strand_mode == StrandMode::BOTH) {
        throw std::runtime_error("For strand='both', use batch_tokenize_dna_both() which returns two arrays");
    }

    if (!enable_padding) {
        // Variable-length output
        std::vector<std::vector<int64_t>> results;
        {
            py::gil_scoped_release release;
            results = batch_tokenize_dna_variable(
                buffer, off_ptr, len_ptr, num_seqs,
                k, stride, use_iupac, strand_mode
            );
        }

        py::list py_results;
        for (const auto& seq_tokens : results) {
            py::array_t<int64_t> arr(seq_tokens.size());
            std::memcpy(arr.mutable_data(), seq_tokens.data(),
                       seq_tokens.size() * sizeof(int64_t));
            py_results.append(arr);
        }
        return py_results;

    } else {
        // Rectangular output with padding
        size_t max_length;
        if (max_len.has_value()) {
            max_length = *max_len;
        } else {
            max_length = 0;
            for (size_t i = 0; i < num_seqs; i++) {
                size_t len = static_cast<size_t>(len_ptr[i]);
                size_t num_tokens = (len >= static_cast<size_t>(k)) ? ((len - k) / stride + 1) : 0;
                max_length = std::max(max_length, num_tokens);
            }
        }

        py::array_t<int64_t> result({num_seqs, max_length});
        int64_t* out = result.mutable_data();

        {
            py::gil_scoped_release release;
            batch_tokenize_dna(buffer, off_ptr, len_ptr, num_seqs,
                              out, max_length, pad_value,
                              k, stride, use_iupac, strand_mode,
                              nullptr);
        }

        return result;
    }
}

// ============================================================================
// Single sequence DNA tokenization
// ============================================================================

py::array_t<int64_t> tokenize_dna(
    py::buffer seq_buf,
    int k,
    int stride = 3,
    std::optional<int> reading_frame = std::nullopt,
    const std::string& alphabet = "acgtn",
    const std::string& strand = "forward")
{
    // Validate parameters
    validation::validate_k(k, "tokenize_dna");
    validation::validate_stride(stride, "tokenize_dna");

    // Validate reading_frame if provided
    if (reading_frame.has_value()) {
        int rf = reading_frame.value();
        if (rf < 0 || rf > 2) {
            throw std::runtime_error("tokenize_dna: reading_frame must be 0, 1, or 2");
        }
    }

    py::buffer_info info = seq_buf.request();
    if (info.ndim != 1) {
        throw std::runtime_error("tokenize_dna: Input must be 1-dimensional");
    }

    const uint8_t* seq = static_cast<const uint8_t*>(info.ptr);
    size_t len = static_cast<size_t>(info.shape[0]);

    // Apply reading frame offset with zero overhead for frame=0 or None
    // PERFORMANCE: Only adjust pointer for frames 1 and 2
    if (reading_frame.has_value() && reading_frame.value() != 0) {
        int rf = reading_frame.value();
        if (static_cast<size_t>(rf) >= len) {
            // Reading frame offset exceeds sequence length, return empty array
            return py::array_t<int64_t>(0);
        }
        seq += rf;
        len -= rf;
    }

    bool use_iupac = (alphabet == "iupac");
    StrandMode strand_mode = parse_strand(strand);

    if (len < static_cast<size_t>(k)) {
        return py::array_t<int64_t>(0);
    }

    size_t num_tokens = (len - k) / stride + 1;
    py::array_t<int64_t> result(num_tokens);
    int64_t* out = result.mutable_data();

    const auto& table = use_iupac ? DNA_IUPAC : DNA_ACGTN;
    const auto& comp = use_iupac ? COMP_IUPAC : COMP_ACGTN;
    uint64_t base = use_iupac ? 16 : 6;
    PowerCache<32> power_cache(base);

    size_t out_idx = 0;
    for (size_t i = 0; i + k <= len && out_idx < num_tokens; i += stride) {
        uint64_t fwd = 0, rev = 0;
        for (int j = 0; j < k; j++) {
            uint8_t code = table[seq[i + j]];
            fwd = fwd * base + code;
            if (strand_mode != StrandMode::FORWARD) {
                uint8_t comp_code = comp[code];
                rev = rev + comp_code * power_cache[j];
            }
        }

        switch (strand_mode) {
            case StrandMode::FORWARD:
                out[out_idx++] = static_cast<int64_t>(fwd);
                break;
            case StrandMode::REVCOMP:
                out[num_tokens - 1 - out_idx] = static_cast<int64_t>(rev);
                out_idx++;
                break;
            case StrandMode::CANONICAL:
                out[out_idx++] = static_cast<int64_t>(std::min(fwd, rev));
                break;
            case StrandMode::BOTH:
                // For single sequence with both, just return forward
                // Use batch_tokenize_dna_both for proper both handling
                out[out_idx++] = static_cast<int64_t>(fwd);
                break;
        }
    }

    return result;
}

// ============================================================================
// Tokenize all 6 reading frames (3 forward + 3 reverse complement)
// ============================================================================

py::list tokenize_dna_all_frames(
    py::buffer seq_buf,
    int k = 3,
    int stride = 3,
    const std::string& alphabet = "acgtn")
{
    // Validate parameters
    validation::validate_k(k, "tokenize_dna_all_frames");
    validation::validate_stride(stride, "tokenize_dna_all_frames");

    py::buffer_info info = seq_buf.request();
    if (info.ndim != 1) {
        throw std::runtime_error("tokenize_dna_all_frames: Input must be 1-dimensional");
    }

    const uint8_t* seq = static_cast<const uint8_t*>(info.ptr);
    size_t len = static_cast<size_t>(info.shape[0]);

    bool use_iupac = (alphabet == "iupac");
    const auto& table = use_iupac ? DNA_IUPAC : DNA_ACGTN;
    const auto& comp = use_iupac ? COMP_IUPAC : COMP_ACGTN;
    uint64_t base = use_iupac ? 16 : 6;

    py::list result;

    // If sequence is too short for even one k-mer, return 6 empty arrays
    if (len < static_cast<size_t>(k)) {
        for (int i = 0; i < 6; i++) {
            result.append(py::array_t<int64_t>(0));
        }
        return result;
    }

    // PERFORMANCE: Compute reverse complement once
    std::vector<uint8_t> revcomp_seq(len);
    reverse_complement(seq, len, revcomp_seq.data(), use_iupac);

    // Extract 3 forward frames (no OpenMP - overhead > benefit for 3 iterations)
    for (int frame = 0; frame < 3; frame++) {
        if (static_cast<size_t>(frame) >= len) {
            result.append(py::array_t<int64_t>(0));
            continue;
        }

        const uint8_t* frame_seq = seq + frame;
        size_t frame_len = len - frame;

        if (frame_len < static_cast<size_t>(k)) {
            result.append(py::array_t<int64_t>(0));
            continue;
        }

        size_t num_tokens = (frame_len - k) / stride + 1;
        py::array_t<int64_t> tokens(num_tokens);
        int64_t* out = tokens.mutable_data();

        // Tokenize this frame
        for (size_t i = 0, out_idx = 0; i + k <= frame_len && out_idx < num_tokens; i += stride) {
            uint64_t value = 0;
            for (int j = 0; j < k; j++) {
                value = value * base + table[frame_seq[i + j]];
            }
            out[out_idx++] = static_cast<int64_t>(value);
        }

        result.append(tokens);
    }

    // Extract 3 reverse complement frames (using precomputed revcomp)
    for (int frame = 0; frame < 3; frame++) {
        if (static_cast<size_t>(frame) >= len) {
            result.append(py::array_t<int64_t>(0));
            continue;
        }

        const uint8_t* frame_seq = revcomp_seq.data() + frame;
        size_t frame_len = len - frame;

        if (frame_len < static_cast<size_t>(k)) {
            result.append(py::array_t<int64_t>(0));
            continue;
        }

        size_t num_tokens = (frame_len - k) / stride + 1;
        py::array_t<int64_t> tokens(num_tokens);
        int64_t* out = tokens.mutable_data();

        // Tokenize this frame
        for (size_t i = 0, out_idx = 0; i + k <= frame_len && out_idx < num_tokens; i += stride) {
            uint64_t value = 0;
            for (int j = 0; j < k; j++) {
                value = value * base + table[frame_seq[i + j]];
            }
            out[out_idx++] = static_cast<int64_t>(value);
        }

        result.append(tokens);
    }

    return result;
}

// ============================================================================
// Batch tokenize all 6 reading frames for multiple sequences
// ============================================================================

py::object batch_tokenize_dna_all_frames(
    py::buffer buffer_buf,
    py::array_t<int64_t> lengths,
    int k = 3,
    int stride = 3,
    const std::string& alphabet = "acgtn",
    bool enable_padding = false,
    std::optional<size_t> max_len = std::nullopt,
    int64_t pad_value = 0)
{
    // Validate parameters
    validation::validate_k(k, "batch_tokenize_dna_all_frames");
    validation::validate_stride(stride, "batch_tokenize_dna_all_frames");

    py::buffer_info buf_info = buffer_buf.request();
    if (buf_info.ndim != 1) {
        throw std::runtime_error("batch_tokenize_dna_all_frames: Buffer must be 1-dimensional");
    }

    const uint8_t* buffer = static_cast<const uint8_t*>(buf_info.ptr);
    size_t num_seqs = static_cast<size_t>(lengths.size());
    size_t buffer_size = static_cast<size_t>(buf_info.size);

    // Compute offsets from lengths
    py::array_t<int64_t> offsets = compute_offsets_from_lengths(lengths);
    const int64_t* off_ptr = offsets.data();
    const int64_t* len_ptr = lengths.data();

    // Validate buffer access
    validation::validate_buffer_access(buffer_size, off_ptr, len_ptr, num_seqs, "batch_tokenize_dna_all_frames");

    bool use_iupac = (alphabet == "iupac");
    const auto& table = use_iupac ? DNA_IUPAC : DNA_ACGTN;
    const auto& comp = use_iupac ? COMP_IUPAC : COMP_ACGTN;
    uint64_t base = use_iupac ? 16 : 6;

    if (!enable_padding) {
        // Variable-length output: List[List[ndarray]] - outer: sequences, inner: 6 frames
        std::vector<std::vector<std::vector<int64_t>>> results(num_seqs);
        for (size_t seq_idx = 0; seq_idx < num_seqs; seq_idx++) {
            results[seq_idx].resize(6);  // 6 frames per sequence
        }

        // PERFORMANCE: OpenMP parallel across sequences ONLY (not frames)
        // Parallelizing N sequences gives N-way speedup (optimal)
        // Parallelizing 6 frames adds overhead without benefit
        {
            py::gil_scoped_release release;

            #pragma omp parallel for schedule(guided)
            for (size_t seq_idx = 0; seq_idx < num_seqs; seq_idx++) {
                size_t offset = static_cast<size_t>(off_ptr[seq_idx]);
                size_t len = static_cast<size_t>(len_ptr[seq_idx]);
                const uint8_t* seq = buffer + offset;

                // If sequence is too short, all frames are empty
                if (len < static_cast<size_t>(k)) {
                    continue;
                }

                // PERFORMANCE: Each thread computes revcomp independently (no shared state)
                std::vector<uint8_t> revcomp_seq(len);
                reverse_complement(seq, len, revcomp_seq.data(), use_iupac);

                // Extract 3 forward frames
                for (int frame = 0; frame < 3; frame++) {
                    if (static_cast<size_t>(frame) >= len) continue;

                    const uint8_t* frame_seq = seq + frame;
                    size_t frame_len = len - frame;

                    if (frame_len < static_cast<size_t>(k)) continue;

                    size_t num_tokens = (frame_len - k) / stride + 1;
                    results[seq_idx][frame].reserve(num_tokens);

                    for (size_t i = 0; i + k <= frame_len; i += stride) {
                        uint64_t value = 0;
                        for (int j = 0; j < k; j++) {
                            value = value * base + table[frame_seq[i + j]];
                        }
                        results[seq_idx][frame].push_back(static_cast<int64_t>(value));
                    }
                }

                // Extract 3 reverse complement frames
                for (int frame = 0; frame < 3; frame++) {
                    if (static_cast<size_t>(frame) >= len) continue;

                    const uint8_t* frame_seq = revcomp_seq.data() + frame;
                    size_t frame_len = len - frame;

                    if (frame_len < static_cast<size_t>(k)) continue;

                    size_t num_tokens = (frame_len - k) / stride + 1;
                    results[seq_idx][3 + frame].reserve(num_tokens);

                    for (size_t i = 0; i + k <= frame_len; i += stride) {
                        uint64_t value = 0;
                        for (int j = 0; j < k; j++) {
                            value = value * base + table[frame_seq[i + j]];
                        }
                        results[seq_idx][3 + frame].push_back(static_cast<int64_t>(value));
                    }
                }
            }
        }

        // Convert to Python list of lists of arrays
        py::list py_results;
        for (size_t seq_idx = 0; seq_idx < num_seqs; seq_idx++) {
            py::list seq_frames;
            for (int frame = 0; frame < 6; frame++) {
                const auto& frame_tokens = results[seq_idx][frame];
                py::array_t<int64_t> arr(frame_tokens.size());
                if (frame_tokens.size() > 0) {
                    std::memcpy(arr.mutable_data(), frame_tokens.data(),
                               frame_tokens.size() * sizeof(int64_t));
                }
                seq_frames.append(arr);
            }
            py_results.append(seq_frames);
        }

        return py_results;

    } else {
        // Rectangular output with padding: List[ndarray] - one 2D array [6, max_len] per sequence
        // Calculate max_len if not provided
        size_t max_length = 0;
        if (max_len.has_value()) {
            max_length = *max_len;
        } else {
            // Auto-calculate max_len across all frames and sequences
            for (size_t i = 0; i < num_seqs; i++) {
                size_t len = static_cast<size_t>(len_ptr[i]);
                for (int frame = 0; frame < 3; frame++) {
                    if (static_cast<size_t>(frame) >= len) continue;
                    size_t frame_len = len - frame;
                    if (frame_len >= static_cast<size_t>(k)) {
                        size_t num_tokens = (frame_len - k) / stride + 1;
                        max_length = std::max(max_length, num_tokens);
                    }
                }
            }
        }

        // Pre-allocate all arrays BEFORE releasing GIL (pybind11 arrays need GIL)
        std::vector<py::array_t<int64_t>> arrays;
        std::vector<int64_t*> data_ptrs;
        arrays.reserve(num_seqs);
        data_ptrs.reserve(num_seqs);

        for (size_t seq_idx = 0; seq_idx < num_seqs; seq_idx++) {
            arrays.emplace_back(std::vector<size_t>{6, max_length});
            int64_t* data = arrays[seq_idx].mutable_data();
            data_ptrs.push_back(data);
            std::fill(data, data + 6 * max_length, pad_value);
        }

        // Now release GIL and do computation
        {
            py::gil_scoped_release release;

            #pragma omp parallel for schedule(guided)
            for (size_t seq_idx = 0; seq_idx < num_seqs; seq_idx++) {
                size_t offset = static_cast<size_t>(off_ptr[seq_idx]);
                size_t len = static_cast<size_t>(len_ptr[seq_idx]);
                const uint8_t* seq = buffer + offset;

                int64_t* data = data_ptrs[seq_idx];

                if (len < static_cast<size_t>(k)) {
                    continue;  // Already filled with padding
                }

                std::vector<uint8_t> revcomp_seq(len);
                reverse_complement(seq, len, revcomp_seq.data(), use_iupac);

                // Extract 3 forward frames
                for (int frame = 0; frame < 3; frame++) {
                    if (static_cast<size_t>(frame) >= len) continue;

                    const uint8_t* frame_seq = seq + frame;
                    size_t frame_len = len - frame;
                    int64_t* row = data + frame * max_length;

                    if (frame_len < static_cast<size_t>(k)) continue;

                    size_t out_idx = 0;
                    for (size_t i = 0; i + k <= frame_len && out_idx < max_length; i += stride) {
                        uint64_t value = 0;
                        for (int j = 0; j < k; j++) {
                            value = value * base + table[frame_seq[i + j]];
                        }
                        row[out_idx++] = static_cast<int64_t>(value);
                    }
                }

                // Extract 3 reverse complement frames
                for (int frame = 0; frame < 3; frame++) {
                    if (static_cast<size_t>(frame) >= len) continue;

                    const uint8_t* frame_seq = revcomp_seq.data() + frame;
                    size_t frame_len = len - frame;
                    int64_t* row = data + (3 + frame) * max_length;

                    if (frame_len < static_cast<size_t>(k)) continue;

                    size_t out_idx = 0;
                    for (size_t i = 0; i + k <= frame_len && out_idx < max_length; i += stride) {
                        uint64_t value = 0;
                        for (int j = 0; j < k; j++) {
                            value = value * base + table[frame_seq[i + j]];
                        }
                        row[out_idx++] = static_cast<int64_t>(value);
                    }
                }
            }
        }

        // Build result list (GIL is back)
        py::list py_results;
        for (size_t seq_idx = 0; seq_idx < num_seqs; seq_idx++) {
            py_results.append(arrays[seq_idx]);
        }

        return py_results;
    }
}

// ============================================================================
// Reverse complement DNA sequence
// ============================================================================

py::array_t<uint8_t> reverse_complement_dna(
    py::buffer seq_buf,
    const std::string& alphabet = "acgtn")
{
    py::buffer_info info = seq_buf.request();
    if (info.ndim != 1) {
        throw std::runtime_error("Input must be 1-dimensional");
    }

    const uint8_t* seq = static_cast<const uint8_t*>(info.ptr);
    size_t len = static_cast<size_t>(info.shape[0]);

    bool use_iupac = (alphabet == "iupac");

    py::array_t<uint8_t> result(len);
    uint8_t* out = result.mutable_data();

    reverse_complement(seq, len, out, use_iupac);

    return result;
}

// ============================================================================
// Single sequence AA tokenization
// ============================================================================

py::array_t<int64_t> tokenize_aa(
    py::buffer seq_buf,
    int k,
    int stride = 1)
{
    // Validate parameters
    validation::validate_k(k, "tokenize_aa");
    validation::validate_stride(stride, "tokenize_aa");

    py::buffer_info info = seq_buf.request();
    if (info.ndim != 1) {
        throw std::runtime_error("tokenize_aa: Input must be 1-dimensional");
    }

    const uint8_t* seq = static_cast<const uint8_t*>(info.ptr);
    size_t len = static_cast<size_t>(info.shape[0]);

    if (len < static_cast<size_t>(k)) {
        return py::array_t<int64_t>(0);  // Empty array
    }

    size_t num_tokens = (len - k) / stride + 1;
    py::array_t<int64_t> result(num_tokens);
    int64_t* out = result.mutable_data();

    constexpr uint64_t base = 29;
    size_t out_idx = 0;
    for (size_t i = 0; i + k <= len && out_idx < num_tokens; i += stride) {
        uint64_t value = 0;
        for (int j = 0; j < k; j++) {
            value = value * base + AA_TABLE[seq[i + j]];
        }
        out[out_idx++] = static_cast<int64_t>(value);
    }

    return result;
}

// ============================================================================
// Batch DNA tokenization from shared buffer (with GIL release)
// ============================================================================

py::object batch_tokenize_dna_shared(
    py::buffer buffer_buf,
    py::array_t<int64_t> lengths,
    int k,
    int stride = 3,
    std::optional<int> reading_frame = std::nullopt,
    bool enable_padding = false,
    std::optional<size_t> max_len = std::nullopt,
    int64_t pad_value = -1,
    const std::string& alphabet = "acgtn",
    const std::string& strand = "forward")
{
    // Validate parameters
    validation::validate_k(k, "batch_tokenize_dna_shared");
    validation::validate_stride(stride, "batch_tokenize_dna_shared");

    // Validate reading_frame if provided
    if (reading_frame.has_value()) {
        int rf = reading_frame.value();
        if (rf < 0 || rf > 2) {
            throw std::runtime_error("batch_tokenize_dna_shared: reading_frame must be 0, 1, or 2");
        }
    }

    py::buffer_info buf_info = buffer_buf.request();
    if (buf_info.ndim != 1) {
        throw std::runtime_error("batch_tokenize_dna_shared: Buffer must be 1-dimensional");
    }

    // Compute offsets from lengths (cumsum)
    py::array_t<int64_t> offsets = compute_offsets_from_lengths(lengths);

    // Delegate to internal helper
    return batch_tokenize_dna_shared_internal(buffer_buf, offsets, lengths,
                                              k, stride, reading_frame,
                                              enable_padding, max_len, pad_value,
                                              alphabet, strand);
}

// ============================================================================
// Batch DNA tokenization - both strands (returns tuple, with GIL release)
// ============================================================================

py::object batch_tokenize_dna_both(
    py::buffer buffer_buf,
    py::array_t<int64_t> lengths,
    int k,
    int stride = 3,
    bool enable_padding = false,
    std::optional<size_t> max_len = std::nullopt,
    int64_t pad_value = -1,
    const std::string& alphabet = "acgtn")
{
    // Validate parameters
    validation::validate_k(k, "batch_tokenize_dna_both");
    validation::validate_stride(stride, "batch_tokenize_dna_both");

    py::buffer_info buf_info = buffer_buf.request();
    if (buf_info.ndim != 1) {
        throw std::runtime_error("batch_tokenize_dna_both: Buffer must be 1-dimensional");
    }

    const uint8_t* buffer = static_cast<const uint8_t*>(buf_info.ptr);
    size_t num_seqs = static_cast<size_t>(lengths.size());
    size_t buffer_size = static_cast<size_t>(buf_info.size);

    // Compute offsets from lengths
    py::array_t<int64_t> offsets = compute_offsets_from_lengths(lengths);
    const int64_t* off_ptr = offsets.data();
    const int64_t* len_ptr = lengths.data();

    // Validate buffer access
    validation::validate_buffer_access(buffer_size, off_ptr, len_ptr, num_seqs, "batch_tokenize_dna_both");

    bool use_iupac = (alphabet == "iupac");

    if (!enable_padding) {
        // Variable-length output
        std::pair<std::vector<std::vector<int64_t>>, std::vector<std::vector<int64_t>>> results;
        {
            py::gil_scoped_release release;
            results = batch_tokenize_dna_both_variable(
                buffer, off_ptr, len_ptr, num_seqs,
                k, stride, use_iupac
            );
        }

        // Convert to Python lists of arrays
        py::list py_results_fwd;
        py::list py_results_rev;
        for (size_t i = 0; i < results.first.size(); i++) {
            const auto& fwd_tokens = results.first[i];
            const auto& rev_tokens = results.second[i];

            py::array_t<int64_t> arr_fwd(fwd_tokens.size());
            py::array_t<int64_t> arr_rev(rev_tokens.size());

            std::memcpy(arr_fwd.mutable_data(), fwd_tokens.data(),
                       fwd_tokens.size() * sizeof(int64_t));
            std::memcpy(arr_rev.mutable_data(), rev_tokens.data(),
                       rev_tokens.size() * sizeof(int64_t));

            py_results_fwd.append(arr_fwd);
            py_results_rev.append(arr_rev);
        }

        return py::make_tuple(py_results_fwd, py_results_rev);

    } else {
        // Rectangular output with padding
        size_t max_length;
        if (max_len.has_value()) {
            max_length = *max_len;
        } else {
            // Auto-calculate max_len
            max_length = 0;
            for (size_t i = 0; i < num_seqs; i++) {
                size_t len = static_cast<size_t>(len_ptr[i]);
                size_t num_tokens = (len >= static_cast<size_t>(k)) ? ((len - k) / stride + 1) : 0;
                max_length = std::max(max_length, num_tokens);
            }
        }

        py::array_t<int64_t> result_fwd({num_seqs, max_length});
        py::array_t<int64_t> result_rev({num_seqs, max_length});
        int64_t* out_fwd = result_fwd.mutable_data();
        int64_t* out_rev = result_rev.mutable_data();

        {
            py::gil_scoped_release release;
            batch_tokenize_dna_both_strands(buffer, off_ptr, len_ptr, num_seqs,
                                           out_fwd, out_rev, max_length, pad_value,
                                           k, stride, use_iupac, nullptr);
        }

        return py::make_tuple(result_fwd, result_rev);
    }
}

// ============================================================================
// Internal helper for batch_tokenize_aa with explicit offsets
// (Used by crop_and_tokenize which needs custom offsets)
// ============================================================================

py::object batch_tokenize_aa_shared_internal(
    py::buffer buffer_buf,
    py::array_t<int64_t> offsets,
    py::array_t<int64_t> lengths,
    int k,
    int stride,
    bool enable_padding,
    std::optional<size_t> max_len,
    int64_t pad_value)
{
    py::buffer_info buf_info = buffer_buf.request();
    const uint8_t* buffer = static_cast<const uint8_t*>(buf_info.ptr);
    const int64_t* off_ptr = offsets.data();
    const int64_t* len_ptr = lengths.data();
    size_t num_seqs = static_cast<size_t>(offsets.size());
    size_t buffer_size = static_cast<size_t>(buf_info.size);

    validation::validate_buffer_access(buffer_size, off_ptr, len_ptr, num_seqs, "batch_tokenize_aa_shared");

    if (!enable_padding) {
        // Variable-length output
        std::vector<std::vector<int64_t>> results;
        {
            py::gil_scoped_release release;
            results = batch_tokenize_aa_variable(
                buffer, off_ptr, len_ptr, num_seqs,
                k, stride
            );
        }

        py::list py_results;
        for (const auto& seq_tokens : results) {
            py::array_t<int64_t> arr(seq_tokens.size());
            std::memcpy(arr.mutable_data(), seq_tokens.data(),
                       seq_tokens.size() * sizeof(int64_t));
            py_results.append(arr);
        }
        return py_results;

    } else {
        // Rectangular output with padding
        size_t max_length;
        if (max_len.has_value()) {
            max_length = *max_len;
        } else {
            max_length = 0;
            for (size_t i = 0; i < num_seqs; i++) {
                size_t len = static_cast<size_t>(len_ptr[i]);
                size_t num_tokens = (len >= static_cast<size_t>(k)) ? ((len - k) / stride + 1) : 0;
                max_length = std::max(max_length, num_tokens);
            }
        }

        py::array_t<int64_t> result({num_seqs, max_length});
        int64_t* out = result.mutable_data();

        {
            py::gil_scoped_release release;
            batch_tokenize_aa(buffer, off_ptr, len_ptr, num_seqs,
                             out, max_length, pad_value,
                             k, stride, nullptr);
        }

        return result;
    }
}

// ============================================================================
// Batch AA tokenization from shared buffer (with GIL release)
// ============================================================================

py::object batch_tokenize_aa_shared(
    py::buffer buffer_buf,
    py::array_t<int64_t> lengths,
    int k,
    int stride = 1,
    bool enable_padding = false,
    std::optional<size_t> max_len = std::nullopt,
    int64_t pad_value = 0)
{
    // Validate parameters
    validation::validate_k(k, "batch_tokenize_aa_shared");
    validation::validate_stride(stride, "batch_tokenize_aa_shared");

    py::buffer_info buf_info = buffer_buf.request();
    if (buf_info.ndim != 1) {
        throw std::runtime_error("batch_tokenize_aa_shared: Buffer must be 1-dimensional");
    }

    // Compute offsets from lengths (cumsum)
    py::array_t<int64_t> offsets = compute_offsets_from_lengths(lengths);

    // Delegate to internal helper
    return batch_tokenize_aa_shared_internal(buffer_buf, offsets, lengths,
                                             k, stride, enable_padding, max_len, pad_value);
}

// ============================================================================
// Crop and tokenize DNA sequences
// ============================================================================

py::object crop_and_tokenize_dna(
    py::buffer buffer_buf,
    py::array_t<int64_t> lengths,
    py::array_t<int64_t> crop_starts,
    py::array_t<int64_t> crop_lengths,
    int k,
    int stride = 3,
    std::optional<int> reading_frame = std::nullopt,
    bool enable_padding = false,
    std::optional<size_t> max_len = std::nullopt,
    int64_t pad_value = -1,
    const std::string& alphabet = "acgtn",
    const std::string& strand = "forward")
{
    // Validate parameters
    validation::validate_k(k, "crop_and_tokenize_dna");
    validation::validate_stride(stride, "crop_and_tokenize_dna");

    // Validate reading_frame if provided
    if (reading_frame.has_value()) {
        int rf = reading_frame.value();
        if (rf < 0 || rf > 2) {
            throw std::runtime_error("crop_and_tokenize_dna: reading_frame must be 0, 1, or 2");
        }
    }

    const int64_t* crop_start_ptr = crop_starts.data();
    const int64_t* crop_len_ptr = crop_lengths.data();
    size_t num_seqs = static_cast<size_t>(lengths.size());

    // Compute offsets from lengths
    py::array_t<int64_t> offsets = compute_offsets_from_lengths(lengths);

    if (crop_starts.size() != num_seqs || crop_lengths.size() != num_seqs) {
        throw std::runtime_error(
            "crop_starts and crop_lengths must have same length as offsets");
    }

    // Create new offsets/lengths for cropped regions
    py::array_t<int64_t> new_offsets(num_seqs);
    py::array_t<int64_t> new_lengths(num_seqs);

    int64_t* new_off_ptr = new_offsets.mutable_data();
    int64_t* new_len_ptr = new_lengths.mutable_data();

    const int64_t* orig_off = offsets.data();
    const int64_t* orig_len = lengths.data();

    // Apply reading_frame offset in addition to crop_starts if specified
    int rf_offset = (reading_frame.has_value() && reading_frame.value() != 0) ? reading_frame.value() : 0;

    for (size_t i = 0; i < num_seqs; i++) {
        if (crop_start_ptr[i] < 0 || crop_len_ptr[i] < 0) {
            throw std::runtime_error("crop_starts and crop_lengths must be non-negative");
        }
        int64_t total_start = crop_start_ptr[i] + rf_offset;
        if (total_start + crop_len_ptr[i] > orig_len[i]) {
            throw std::runtime_error("crop (with reading_frame offset) extends beyond sequence boundary");
        }
        new_off_ptr[i] = orig_off[i] + total_start;
        new_len_ptr[i] = (crop_len_ptr[i] > rf_offset) ? (crop_len_ptr[i] - rf_offset) : 0;
    }

    // Delegate to internal helper (pass reading_frame=None since we already applied it)
    return batch_tokenize_dna_shared_internal(buffer_buf, new_offsets, new_lengths,
                                              k, stride, std::nullopt, enable_padding, max_len,
                                              pad_value, alphabet, strand);
}

// ============================================================================
// Crop and tokenize AA sequences
// ============================================================================

py::object crop_and_tokenize_aa(
    py::buffer buffer_buf,
    py::array_t<int64_t> lengths,
    py::array_t<int64_t> crop_starts,
    py::array_t<int64_t> crop_lengths,
    int k,
    int stride = 1,
    bool enable_padding = false,
    std::optional<size_t> max_len = std::nullopt,
    int64_t pad_value = 0)
{
    // Validate parameters
    validation::validate_k(k, "crop_and_tokenize_aa");
    validation::validate_stride(stride, "crop_and_tokenize_aa");

    const int64_t* crop_start_ptr = crop_starts.data();
    const int64_t* crop_len_ptr = crop_lengths.data();
    size_t num_seqs = static_cast<size_t>(lengths.size());

    // Compute offsets from lengths
    py::array_t<int64_t> offsets = compute_offsets_from_lengths(lengths);

    if (crop_starts.size() != num_seqs || crop_lengths.size() != num_seqs) {
        throw std::runtime_error(
            "crop_starts and crop_lengths must have same length as lengths");
    }

    // Create new offsets/lengths for cropped regions
    py::array_t<int64_t> new_offsets(num_seqs);
    py::array_t<int64_t> new_lengths(num_seqs);

    int64_t* new_off_ptr = new_offsets.mutable_data();
    int64_t* new_len_ptr = new_lengths.mutable_data();

    const int64_t* orig_off = offsets.data();
    const int64_t* orig_len = lengths.data();

    for (size_t i = 0; i < num_seqs; i++) {
        if (crop_start_ptr[i] < 0 || crop_len_ptr[i] < 0) {
            throw std::runtime_error("crop_starts and crop_lengths must be non-negative");
        }
        if (crop_start_ptr[i] + crop_len_ptr[i] > orig_len[i]) {
            throw std::runtime_error("crop extends beyond sequence boundary");
        }
        new_off_ptr[i] = orig_off[i] + crop_start_ptr[i];
        new_len_ptr[i] = crop_len_ptr[i];
    }

    // Delegate to internal helper
    return batch_tokenize_aa_shared_internal(buffer_buf, new_offsets, new_lengths,
                                             k, stride, enable_padding, max_len,
                                             pad_value);
}

// ============================================================================
// Hash tokens to fixed bucket size (with GIL release)
// ============================================================================

py::array_t<int64_t> hash_tokens(
    py::array_t<int64_t> tokens,
    int64_t num_buckets)
{
    if (num_buckets <= 0) {
        throw std::runtime_error("num_buckets must be positive");
    }

    py::buffer_info info = tokens.request();
    const int64_t* in_ptr = tokens.data();
    size_t total = static_cast<size_t>(tokens.size());

    // Create output with same shape
    std::vector<ssize_t> shape(info.shape.begin(), info.shape.end());
    py::array_t<int64_t> result(shape);
    int64_t* out_ptr = result.mutable_data();

    {
        py::gil_scoped_release release;  // Release GIL for parallel execution
        batch_hash_tokens(in_ptr, out_ptr, total, num_buckets);
    }

    return result;
}

// ============================================================================
// Module definition
// ============================================================================

PYBIND11_MODULE(_bioenc, m) {
    m.doc() = "bioenc: Fast k-mer tokenization for bioinformatics ML";

    m.def("tokenize_dna", &tokenize_dna,
          py::arg("seq"),
          py::arg("k"),
          py::arg("stride") = 3,
          py::arg("reading_frame") = py::none(),
          py::arg("alphabet") = "acgtn",
          py::arg("strand") = "forward",
          R"doc(
          Tokenize a single DNA sequence into k-mer indices.

          Parameters
          ----------
          seq : buffer
              Input sequence as uint8 array (ASCII bytes)
          k : int
              K-mer size
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
          numpy.ndarray
              Array of int64 k-mer indices
          )doc");

    m.def("reverse_complement_dna", &reverse_complement_dna,
          py::arg("seq"),
          py::arg("alphabet") = "acgtn",
          R"doc(
          Compute reverse complement of a DNA sequence.

          Parameters
          ----------
          seq : buffer
              Input sequence as uint8 array (ASCII bytes)
          alphabet : str, optional
              'acgtn' or 'iupac' (default: 'acgtn')

          Returns
          -------
          numpy.ndarray
              Reverse complement as uint8 array
          )doc");

    m.def("tokenize_dna_all_frames", &tokenize_dna_all_frames,
          py::arg("seq"),
          py::arg("k") = 3,
          py::arg("stride") = 3,
          py::arg("alphabet") = "acgtn",
          R"doc(
          Extract all 6 reading frames (3 forward + 3 reverse complement).

          Computes reverse complement once for efficiency. This function
          is optimized for single sequences (no OpenMP overhead).

          Parameters
          ----------
          seq : buffer
              Input sequence as uint8 array (ASCII bytes)
          k : int, optional
              K-mer size (default: 3 for codons)
          stride : int, optional
              Step size between k-mers (default: 3 for non-overlapping)
          alphabet : str, optional
              'acgtn' or 'iupac' (default: 'acgtn')

          Returns
          -------
          list[numpy.ndarray]
              List of 6 arrays: [fwd_frame0, fwd_frame1, fwd_frame2,
                                 rev_frame0, rev_frame1, rev_frame2]
              Each array contains int64 k-mer indices for that frame.

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
          )doc");

    m.def("batch_tokenize_dna_all_frames", &batch_tokenize_dna_all_frames,
          py::arg("buffer"),
          py::arg("lengths"),
          py::arg("k") = 3,
          py::arg("stride") = 3,
          py::arg("alphabet") = "acgtn",
          py::arg("enable_padding") = false,
          py::arg("max_len") = py::none(),
          py::arg("pad_value") = 0,
          R"doc(
          Batch tokenize all 6 reading frames for multiple sequences.

          Uses OpenMP for parallel processing across sequences (not frames).
          Releases the Python GIL during computation to allow true parallelism.

          Each thread independently computes reverse complement (no shared state).
          Optimized with schedule(guided) for load balancing with variable lengths.

          Parameters
          ----------
          buffer : buffer
              Concatenated sequences as uint8 array
          lengths : numpy.ndarray
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
              Padding value for short sequences (default: 0)

          Returns
          -------
          list[list[numpy.ndarray]] or list[numpy.ndarray]
              If enable_padding=False:
                  List of lists: result[seq_idx][frame_idx] = np.ndarray
                  Outer list = sequences, inner list = 6 frames per sequence
              If enable_padding=True:
                  List of 2D arrays: result[seq_idx] = np.ndarray[6, max_len]

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
          )doc");

    m.def("tokenize_aa", &tokenize_aa,
          py::arg("seq"),
          py::arg("k"),
          py::arg("stride") = 1,
          R"doc(
          Tokenize a single amino acid sequence into k-mer indices.

          Parameters
          ----------
          seq : buffer
              Input sequence as uint8 array (ASCII bytes)
          k : int
              K-mer size
          stride : int, optional
              Step size between k-mers (default: 1)

          Returns
          -------
          numpy.ndarray
              Array of int64 k-mer indices

          Examples
          --------
          >>> seq = np.frombuffer(b"ACDEFGHIK", dtype=np.uint8)
          >>> tokens = bioenc.tokenize_aa(seq, k=3)
          >>> tokens.shape
          (7,)
          )doc");

    m.def("batch_tokenize_dna_shared", &batch_tokenize_dna_shared,
          py::arg("buffer"),
          py::arg("lengths"),
          py::arg("k"),
          py::arg("stride") = 3,
          py::arg("reading_frame") = py::none(),
          py::arg("enable_padding") = false,
          py::arg("max_len") = py::none(),
          py::arg("pad_value") = 0,
          py::arg("alphabet") = "acgtn",
          py::arg("strand") = "forward",
          R"doc(
          Batch tokenize DNA sequences from a shared buffer.

          Uses OpenMP for parallel processing across sequences. Releases the
          Python GIL during computation to allow true parallelism.

          By default, returns variable-length arrays (one per sequence).
          Set enable_padding=True for fixed rectangular output.

          Parameters
          ----------
          buffer : buffer
              Concatenated sequences as uint8 array
          lengths : numpy.ndarray
              Length of each sequence
          k : int
              K-mer size
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
              Padding value for short sequences (default: 0)
          alphabet : str, optional
              'acgtn' or 'iupac' (default: 'acgtn')
          strand : str, optional
              'forward', 'revcomp', or 'canonical' (default: 'forward')

          Returns
          -------
          list[numpy.ndarray] or numpy.ndarray
              If enable_padding=False: List of variable-length arrays
              If enable_padding=True: 2D array of shape (num_seqs, max_len)
          )doc");

    m.def("batch_tokenize_dna_both", &batch_tokenize_dna_both,
          py::arg("buffer"),
          py::arg("lengths"),
          py::arg("k"),
          py::arg("stride") = 3,
          py::arg("enable_padding") = false,
          py::arg("max_len") = py::none(),
          py::arg("pad_value") = 0,
          py::arg("alphabet") = "acgtn",
          R"doc(
          Batch tokenize DNA sequences returning both forward and reverse complement.

          Uses OpenMP for parallel processing across sequences. Releases the
          Python GIL during computation to allow true parallelism.

          By default, returns variable-length arrays (one per sequence).
          Set enable_padding=True for fixed rectangular output.

          Parameters
          ----------
          buffer : buffer
              Concatenated sequences as uint8 array
          lengths : numpy.ndarray
              Length of each sequence
          k : int
              K-mer size
          stride : int, optional
              Step size between k-mers (default: 3 for non-overlapping codons)
          enable_padding : bool, optional
              If True, return rectangular arrays with padding (default: False)
          max_len : int, optional
              If enable_padding=True and this is set, pad to this length.
              If None, pad to actual maximum token count.
          pad_value : int, optional
              Padding value for short sequences (default: 0)
          alphabet : str, optional
              'acgtn' or 'iupac' (default: 'acgtn')

          Returns
          -------
          tuple[list[numpy.ndarray], list[numpy.ndarray]] or tuple[numpy.ndarray, numpy.ndarray]
              If enable_padding=False: Tuple of (fwd_list, rev_list) with variable-length arrays
              If enable_padding=True: Tuple of 2D arrays, each shape (num_seqs, max_len)
          )doc");

    m.def("batch_tokenize_aa_shared", &batch_tokenize_aa_shared,
          py::arg("buffer"),
          py::arg("lengths"),
          py::arg("k"),
          py::arg("stride") = 1,
          py::arg("enable_padding") = false,
          py::arg("max_len") = py::none(),
          py::arg("pad_value") = 0,
          R"doc(
          Batch tokenize amino acid sequences from a shared buffer.

          Uses OpenMP for parallel processing across sequences. Releases the
          Python GIL during computation to allow true parallelism.

          By default, returns variable-length arrays (one per sequence).
          Set enable_padding=True for fixed rectangular output.

          Parameters
          ----------
          buffer : buffer
              Concatenated sequences as uint8 array
          lengths : numpy.ndarray
              Length of each sequence
          k : int
              K-mer size
          stride : int, optional
              Step size between k-mers (default: 1)
          enable_padding : bool, optional
              If True, return rectangular array with padding (default: False)
          max_len : int, optional
              If enable_padding=True and this is set, pad to this length.
              If None, pad to actual maximum token count.
          pad_value : int, optional
              Padding value for short sequences (default: 0)

          Returns
          -------
          list[numpy.ndarray] or numpy.ndarray
              If enable_padding=False: List of variable-length arrays
              If enable_padding=True: 2D array of shape (num_seqs, max_len)
          )doc");

    m.def("crop_and_tokenize_dna", &crop_and_tokenize_dna,
          py::arg("buffer"),
          py::arg("lengths"),
          py::arg("crop_starts"),
          py::arg("crop_lengths"),
          py::arg("k"),
          py::arg("stride") = 3,
          py::arg("reading_frame") = py::none(),
          py::arg("enable_padding") = false,
          py::arg("max_len") = py::none(),
          py::arg("pad_value") = 0,
          py::arg("alphabet") = "acgtn",
          py::arg("strand") = "forward",
          R"doc(
          Crop sequences to windows, then tokenize.

          Useful for reading frames, sliding windows, and data augmentation.

          Parameters
          ----------
          buffer : buffer
              Concatenated sequences as uint8 array
          lengths : numpy.ndarray
              Length of each sequence
          crop_starts : numpy.ndarray
              Start position for crop in each sequence
          crop_lengths : numpy.ndarray
              Length of crop for each sequence
          k : int
              K-mer size
          stride : int, optional
              Step size between k-mers (default: 3 for non-overlapping codons)
          reading_frame : int, optional
              If specified, apply additional reading frame offset (0, 1, or 2) after cropping.
              Default: None
          enable_padding : bool, optional
              Return rectangular array (default: False)
          max_len : int, optional
              Max length if padding enabled
          pad_value : int, optional
              Padding value (default: 0)
          alphabet : str, optional
              'acgtn' or 'iupac' (default: 'acgtn')
          strand : str, optional
              'forward', 'revcomp', or 'canonical' (default: 'forward')

          Returns
          -------
          list[numpy.ndarray] or numpy.ndarray
              Tokenized cropped sequences
          )doc");

    m.def("crop_and_tokenize_aa", &crop_and_tokenize_aa,
          py::arg("buffer"),
          py::arg("lengths"),
          py::arg("crop_starts"),
          py::arg("crop_lengths"),
          py::arg("k"),
          py::arg("stride") = 1,
          py::arg("enable_padding") = false,
          py::arg("max_len") = py::none(),
          py::arg("pad_value") = 0,
          R"doc(
          Crop amino acid sequences to windows, then tokenize.

          Useful for windowing and data augmentation.

          Parameters
          ----------
          buffer : buffer
              Concatenated sequences as uint8 array
          lengths : numpy.ndarray
              Length of each sequence
          crop_starts : numpy.ndarray
              Start position for crop in each sequence
          crop_lengths : numpy.ndarray
              Length of crop for each sequence
          k : int
              K-mer size
          stride : int, optional
              Step size between k-mers (default: 1)
          enable_padding : bool, optional
              Return rectangular array (default: False)
          max_len : int, optional
              Max length if padding enabled
          pad_value : int, optional
              Padding value (default: 0)

          Returns
          -------
          list[numpy.ndarray] or numpy.ndarray
              Tokenized cropped sequences
          )doc");

    m.def("hash_tokens", &hash_tokens,
          py::arg("tokens"),
          py::arg("num_buckets"),
          R"doc(
          Hash token indices to a fixed number of buckets.

          Useful for large k values where vocabulary size exceeds embedding table limits.
          Preserves zero (padding) and negative values unchanged.

          Uses OpenMP for parallel processing. Releases the Python GIL during
          computation to allow true parallelism.

          Parameters
          ----------
          tokens : numpy.ndarray
              Array of int64 token indices (any shape)
          num_buckets : int
              Number of hash buckets

          Returns
          -------
          numpy.ndarray
              Hashed tokens with same shape as input
          )doc");
}
