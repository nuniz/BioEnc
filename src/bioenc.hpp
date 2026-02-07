#pragma once

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <string_view>
#include <vector>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace bioenc {

// ============================================================================
// Alphabet tables
// ============================================================================

// ACGTN alphabet (base=6): PAD=0, UNK/N=1, A=2, C=3, G=4, T=5
constexpr uint8_t DNA_ACGTN[256] = {
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, // 0-15
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, // 16-31
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, // 32-47
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, // 48-63
    1,2,1,3,1,1,1,4,1,1,1,1,1,1,1,1, // 64-79  (A=65, C=67, G=71)
    1,1,1,1,5,5,1,1,1,1,1,1,1,1,1,1, // 80-95  (T=84, U=85)
    1,2,1,3,1,1,1,4,1,1,1,1,1,1,1,1, // 96-111 (a=97, c=99, g=103)
    1,1,1,1,5,5,1,1,1,1,1,1,1,1,1,1, // 112-127 (t=116, u=117)
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
};

// IUPAC alphabet (base=16): PAD=0, UNK/N=1, A=2, C=3, G=4, T=5, U=5
// R=6, Y=7, S=8, W=9, K=10, M=11, B=12, D=13, H=14, V=15
constexpr uint8_t DNA_IUPAC[256] = {
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1, 2,12, 3,13,1,1, 4,14,1,1,10,1,11, 1,1, // @ABCDEFGHIJKLMNO
    1,1, 6, 8, 5, 5,15, 9,1, 7,1,1,1,1,1,1, // PQRSTUVWXYZ
    1, 2,12, 3,13,1,1, 4,14,1,1,10,1,11, 1,1, // `abcdefghijklmno
    1,1, 6, 8, 5, 5,15, 9,1, 7,1,1,1,1,1,1, // pqrstuvwxyz
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
};

// Complement tables for reverse complement
// ACGTN: PAD->PAD (0->0), UNK->UNK (1->1), A<->T (2<->5), C<->G (3<->4)
constexpr uint8_t COMP_ACGTN[6] = {0, 1, 5, 4, 3, 2};

// IUPAC complements
// PAD->PAD(0), UNK->UNK(1), A<->T(2<->5), C<->G(3<->4),
// R<->Y(6<->7), S<->S(8), W<->W(9), K<->M(10<->11),
// B<->V(12<->15), D<->H(13<->14)
constexpr uint8_t COMP_IUPAC[16] = {
    0,  // PAD -> PAD
    1,  // UNK -> UNK
    5,  // A -> T
    4,  // C -> G
    3,  // G -> C
    2,  // T -> A
    7,  // R(AG) -> Y(TC)
    6,  // Y(TC) -> R(AG)
    8,  // S(GC) -> S(GC)
    9,  // W(AT) -> W(AT)
    11, // K(GT) -> M(AC)
    10, // M(AC) -> K(GT)
    15, // B(CGT) -> V(ACG)
    14, // D(AGT) -> H(ACT)
    13, // H(ACT) -> D(AGT)
    12  // V(ACG) -> B(CGT)
};

// Amino acid alphabet (base=29): PAD=0, UNK/X=1
// A=2,C=3,D=4,E=5,F=6,G=7,H=8,I=9,K=10,L=11,M=12,N=13,P=14,Q=15,R=16,S=17,T=18,V=19,W=20,Y=21
// B=22,Z=23,J=24,U=25,O=26,*=27,-=28
constexpr uint8_t AA_TABLE[256] = {
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,27,1,1,28,1,1, // * at 42, - at 45
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1, 2,22, 3, 4, 5, 6, 7, 8, 9,24,10,11,12,13,26, // @ABCDEFGHIJKLMNO
    14,15,16,17,18,25,19,20, 1,21,23,1,1,1,1,1, // PQRSTUVWXYZ
    1, 2,22, 3, 4, 5, 6, 7, 8, 9,24,10,11,12,13,26, // `abcdefghijklmno
    14,15,16,17,18,25,19,20, 1,21,23,1,1,1,1,1, // pqrstuvwxyz
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
};

// ============================================================================
// Utility functions
// ============================================================================

inline uint64_t pow_u64(uint64_t base, uint64_t exp) {
    uint64_t result = 1;
    while (exp > 0) {
        if (exp & 1) result *= base;
        base *= base;
        exp >>= 1;
    }
    return result;
}

// MurmurHash3 finalizer for hashing tokens.
// Provides excellent avalanche properties: changing any input bit
// affects ~50% of output bits, making it ideal for k-mer hashing.
// This is particularly useful for large k values where vocabulary
// size (base^k) exceeds embedding table capacity.
inline uint64_t hash_mix(uint64_t key) {
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return key;
}

inline int64_t hash_kmer(int64_t value, int64_t num_buckets) {
    if (value <= 0) return value; // preserve padding (0) and negative values
    return static_cast<int64_t>(hash_mix(static_cast<uint64_t>(value)) % static_cast<uint64_t>(num_buckets));
}

// ============================================================================
// Power cache for efficient reverse complement computation
// ============================================================================

// Precomputed power lookup table to avoid O(k log k) pow_u64 calls in inner loops.
// This optimization reduces reverse complement calculation from O(k^2 log k) to O(k).
template<size_t MAX_K = 32>
class PowerCache {
public:
    PowerCache(uint64_t base) : base_(base) {
        powers_[0] = 1;
        for (size_t i = 1; i < MAX_K; i++) {
            powers_[i] = powers_[i-1] * base_;
        }
    }
    inline uint64_t operator[](size_t exp) const { return powers_[exp]; }
private:
    uint64_t base_;
    uint64_t powers_[MAX_K];
};

// ============================================================================
// Strand mode parsing
// ============================================================================

enum class StrandMode { FORWARD, REVCOMP, CANONICAL, BOTH };

inline StrandMode parse_strand(std::string_view s) {
    if (s == "forward") return StrandMode::FORWARD;
    if (s == "revcomp") return StrandMode::REVCOMP;
    if (s == "canonical") return StrandMode::CANONICAL;
    if (s == "both") return StrandMode::BOTH;
    return StrandMode::FORWARD; // default
}

// ============================================================================
// Rolling k-mer encoder (generic base-B)
// ============================================================================

template<typename Table>
class RollingIndex {
public:
    RollingIndex(const Table& table, uint64_t base, int k)
        : table_(table), base_(base), k_(k),
          high_pow_(pow_u64(base, k - 1)), value_(0), count_(0) {}

    void reset() {
        value_ = 0;
        count_ = 0;
    }

    void push(uint8_t ch) {
        uint8_t code = table_[ch];
        if (count_ >= k_) {
            // Remove oldest, not needed for fresh windows
        }
        value_ = value_ * base_ + code;
        count_++;
        if (count_ > k_) {
            // We've exceeded k, so we trim with modulo
            value_ %= pow_u64(base_, k_);
        }
    }

    // Slide: remove oldest, add new
    void slide(uint8_t old_ch, uint8_t new_ch) {
        uint8_t old_code = table_[old_ch];
        uint8_t new_code = table_[new_ch];
        value_ = (value_ - old_code * high_pow_) * base_ + new_code;
    }

    int64_t value() const { return static_cast<int64_t>(value_); }
    bool ready() const { return count_ >= k_; }

private:
    const Table& table_;
    uint64_t base_;
    int k_;
    uint64_t high_pow_;
    uint64_t value_;
    int count_;
};

// ============================================================================
// Rolling DNA encoder with reverse complement tracking
// ============================================================================

// Dual rolling encoder that simultaneously maintains forward and reverse
// complement k-mer encodings in O(1) per base, enabling efficient canonical
// k-mer extraction. This optimization is critical for stride=1 operations,
// reducing complexity from O(nk^2 log k) to O(nk) for n sequences of length k.
//
// Example: For k=8, ACGTACGT encodes to both:
//   - Forward: A*6^7 + C*6^6 + G*6^5 + ... (left-to-right)
//   - Reverse: T*6^0 + G*6^1 + C*6^2 + ... (complement, right-to-left)
// The canonical k-mer is min(forward, reverse), ensuring strand-invariant tokens.
template<typename Table, typename CompTable>
class RollingDnaCanon {
public:
    RollingDnaCanon(const Table& table, const CompTable& comp,
                    uint64_t base, int k)
        : table_(table), comp_(comp), base_(base), k_(k),
          high_pow_(pow_u64(base, k - 1)),
          fwd_(0), rev_(0), count_(0) {}

    void reset() {
        fwd_ = 0;
        rev_ = 0;
        count_ = 0;
    }

    void push(uint8_t ch) {
        uint8_t code = table_[ch];
        uint8_t comp_code = comp_[code];

        fwd_ = fwd_ * base_ + code;
        rev_ = rev_ + comp_code * (count_ < k_ ? pow_u64(base_, count_) : high_pow_ * base_);

        count_++;
        if (count_ > k_) {
            fwd_ %= pow_u64(base_, k_);
            rev_ /= base_;
        }
    }

    void slide(uint8_t old_ch, uint8_t new_ch) {
        uint8_t old_code = table_[old_ch];
        uint8_t new_code = table_[new_ch];
        uint8_t old_comp = comp_[old_code];
        uint8_t new_comp = comp_[new_code];

        fwd_ = (fwd_ - old_code * high_pow_) * base_ + new_code;
        rev_ = (rev_ - old_comp) / base_ + new_comp * high_pow_;
    }

    int64_t forward() const { return static_cast<int64_t>(fwd_); }
    int64_t revcomp() const { return static_cast<int64_t>(rev_); }
    int64_t canonical() const { return static_cast<int64_t>(std::min(fwd_, rev_)); }
    bool ready() const { return count_ >= k_; }

private:
    const Table& table_;
    const CompTable& comp_;
    uint64_t base_;
    int k_;
    uint64_t high_pow_;
    uint64_t fwd_;
    uint64_t rev_;
    int count_;
};

// ============================================================================
// Single sequence DNA tokenization
// ============================================================================

inline void tokenize_dna_seq(const uint8_t* seq, size_t len,
                             int64_t* out, size_t out_len,
                             int k, int stride,
                             bool use_iupac, StrandMode strand) {
    const auto& table = use_iupac ? DNA_IUPAC : DNA_ACGTN;
    const auto& comp = use_iupac ? COMP_IUPAC : COMP_ACGTN;
    uint64_t base = use_iupac ? 16 : 6;

    size_t num_tokens = (len >= static_cast<size_t>(k)) ? ((len - k) / stride + 1) : 0;
    size_t num_tokens_out = std::min(num_tokens, out_len);
    if (num_tokens_out == 0) {
        return;
    }

    size_t out_idx = 0;

    if (strand == StrandMode::FORWARD) {
        // Forward only - straightforward tokenization
        for (size_t i = 0; i + k <= len && out_idx < num_tokens_out; i += stride) {
            uint64_t v = 0;
            for (int j = 0; j < k; j++) {
                v = v * base + table[seq[i + j]];
            }
            out[out_idx++] = static_cast<int64_t>(v);
        }
    } else {
        // Need canonical/revcomp tracking
        PowerCache<32> power_cache(base);
        uint64_t high_pow = power_cache[k - 1];
        for (size_t i = 0; i + k <= len && out_idx < num_tokens_out; i += stride) {
            uint64_t fwd = 0, rev = 0;
            for (int j = 0; j < k; j++) {
                uint8_t code = table[seq[i + j]];
                uint8_t comp_code = comp[code];
                fwd = fwd * base + code;
                rev = rev + comp_code * power_cache[j];
            }

            switch (strand) {
                case StrandMode::FORWARD:
                    out[out_idx++] = static_cast<int64_t>(fwd);
                    break;
                case StrandMode::REVCOMP:
                    out[num_tokens_out - 1 - out_idx] = static_cast<int64_t>(rev);
                    out_idx++;
                    break;
                case StrandMode::CANONICAL:
                    out[out_idx++] = static_cast<int64_t>(std::min(fwd, rev));
                    break;
                case StrandMode::BOTH:
                    // Handled separately
                    break;
            }
        }
    }
}

// ============================================================================
// Reverse complement a sequence
// ============================================================================

inline void reverse_complement(const uint8_t* seq, size_t len,
                               uint8_t* out, bool use_iupac) {
    // Output characters for codes (indexed by code value)
    // Code: 0=PAD, 1=UNK/N, 2=A, 3=C, 4=G, 5=T
    static const char BASES_ACGTN[] = "\0NACGT";
    // Code: 0=PAD, 1=UNK/N, 2=A, 3=C, 4=G, 5=T, 6=R, 7=Y, 8=S, 9=W, 10=K, 11=M, 12=B, 13=D, 14=H, 15=V
    static const char BASES_IUPAC[] = "\0NACGTRYSWKMBDHV";

    const auto& table = use_iupac ? DNA_IUPAC : DNA_ACGTN;
    const auto& comp = use_iupac ? COMP_IUPAC : COMP_ACGTN;
    const char* bases = use_iupac ? BASES_IUPAC : BASES_ACGTN;

    for (size_t i = 0; i < len; i++) {
        uint8_t code = table[seq[len - 1 - i]];
        uint8_t comp_code = comp[code];
        out[i] = static_cast<uint8_t>(bases[comp_code]);
    }
}

// ============================================================================
// Batch DNA tokenization from shared buffer (OpenMP parallelized)
// ============================================================================

inline void batch_tokenize_dna(
    const uint8_t* buffer,
    const int64_t* offsets,
    const int64_t* lengths,
    size_t num_seqs,
    int64_t* out,
    size_t max_len,
    int64_t pad_value,
    int k, int stride,
    bool use_iupac,
    StrandMode strand,
    const int64_t* start_indices)  // nullable, defaults to 0
{
    const auto& table = use_iupac ? DNA_IUPAC : DNA_ACGTN;
    const auto& comp = use_iupac ? COMP_IUPAC : COMP_ACGTN;
    uint64_t base = use_iupac ? 16 : 6;
    PowerCache<32> power_cache(base);

    #pragma omp parallel for schedule(guided)
    for (size_t seq_idx = 0; seq_idx < num_seqs; seq_idx++) {
        int64_t* row = out + seq_idx * max_len;

        size_t offset = static_cast<size_t>(offsets[seq_idx]);
        size_t len = static_cast<size_t>(lengths[seq_idx]);
        const uint8_t* seq = buffer + offset;

        size_t start = start_indices ? static_cast<size_t>(start_indices[seq_idx]) : 0;
        size_t available = (len > start) ? (len - start) : 0;
        size_t num_tokens = (available >= static_cast<size_t>(k)) ? ((available - k) / stride + 1) : 0;
        size_t num_tokens_out = std::min(num_tokens, max_len);

        if (num_tokens_out == 0) {
            // Sequence too short (or start beyond end), fill entire row with padding
            std::fill(row, row + max_len, pad_value);
            continue;
        }

        size_t out_idx = 0;

        // Optimized path for stride=1 with canonical/revcomp using rolling window
        if (stride == 1 && strand != StrandMode::FORWARD) {
            RollingDnaCanon<decltype(table), decltype(comp)> roller(table, comp, base, k);
            for (size_t i = start; i < len && out_idx < num_tokens_out; i++) {
                roller.push(seq[i]);
                if (roller.ready()) {
                    switch (strand) {
                        case StrandMode::REVCOMP:
                            row[num_tokens_out - 1 - out_idx] = roller.revcomp();
                            out_idx++;
                            break;
                        case StrandMode::CANONICAL:
                            row[out_idx++] = roller.canonical();
                            break;
                        default:
                            break;
                    }
                }
            }
        } else {
            // Naive loop for stride > 1 or forward-only
            for (size_t i = start; i + k <= len && out_idx < num_tokens_out; i += stride) {
                uint64_t fwd = 0, rev = 0;
                for (int j = 0; j < k; j++) {
                    uint8_t code = table[seq[i + j]];
                    fwd = fwd * base + code;
                    if (strand != StrandMode::FORWARD) {
                        uint8_t comp_code = comp[code];
                        rev = rev + comp_code * power_cache[j];
                    }
                }

                switch (strand) {
                    case StrandMode::FORWARD:
                        row[out_idx++] = static_cast<int64_t>(fwd);
                        break;
                    case StrandMode::REVCOMP:
                        row[num_tokens_out - 1 - out_idx] = static_cast<int64_t>(rev);
                        out_idx++;
                        break;
                    case StrandMode::CANONICAL:
                        row[out_idx++] = static_cast<int64_t>(std::min(fwd, rev));
                        break;
                    case StrandMode::BOTH:
                        // Handled in separate function
                        break;
                }
            }
        }

        // Fill remaining slots with padding
        if (num_tokens_out < max_len) {
            std::fill(row + num_tokens_out, row + max_len, pad_value);
        }
    }
}

// Batch DNA both strands (returns forward and revcomp separately, OpenMP parallelized)
inline void batch_tokenize_dna_both_strands(
    const uint8_t* buffer,
    const int64_t* offsets,
    const int64_t* lengths,
    size_t num_seqs,
    int64_t* out_fwd,
    int64_t* out_rev,
    size_t max_len,
    int64_t pad_value,
    int k, int stride,
    bool use_iupac,
    const int64_t* start_indices)  // nullable, defaults to 0
{
    const auto& table = use_iupac ? DNA_IUPAC : DNA_ACGTN;
    const auto& comp = use_iupac ? COMP_IUPAC : COMP_ACGTN;
    uint64_t base = use_iupac ? 16 : 6;
    PowerCache<32> power_cache(base);

    #pragma omp parallel for schedule(guided)
    for (size_t seq_idx = 0; seq_idx < num_seqs; seq_idx++) {
        int64_t* row_fwd = out_fwd + seq_idx * max_len;
        int64_t* row_rev = out_rev + seq_idx * max_len;

        size_t offset = static_cast<size_t>(offsets[seq_idx]);
        size_t len = static_cast<size_t>(lengths[seq_idx]);
        const uint8_t* seq = buffer + offset;

        size_t start = start_indices ? static_cast<size_t>(start_indices[seq_idx]) : 0;
        size_t available = (len > start) ? (len - start) : 0;
        size_t num_tokens = (available >= static_cast<size_t>(k)) ? ((available - k) / stride + 1) : 0;
        size_t num_tokens_out = std::min(num_tokens, max_len);

        if (num_tokens_out == 0) {
            // Sequence too short (or start beyond end), fill entire rows with padding
            std::fill(row_fwd, row_fwd + max_len, pad_value);
            std::fill(row_rev, row_rev + max_len, pad_value);
            continue;
        }

        size_t out_idx = 0;

        // Optimized path for stride=1 using rolling window
        if (stride == 1) {
            RollingDnaCanon<decltype(table), decltype(comp)> roller(table, comp, base, k);
            for (size_t i = start; i < len && out_idx < num_tokens_out; i++) {
                roller.push(seq[i]);
                if (roller.ready()) {
                    row_fwd[out_idx] = roller.forward();
                    row_rev[num_tokens_out - 1 - out_idx] = roller.revcomp();
                    out_idx++;
                }
            }
        } else {
            // Naive loop for stride > 1
            for (size_t i = start; i + k <= len && out_idx < num_tokens_out; i += stride) {
                uint64_t fwd = 0, rev = 0;
                for (int j = 0; j < k; j++) {
                    uint8_t code = table[seq[i + j]];
                    uint8_t comp_code = comp[code];
                    fwd = fwd * base + code;
                    rev = rev + comp_code * power_cache[j];
                }
                row_fwd[out_idx] = static_cast<int64_t>(fwd);
                row_rev[num_tokens_out - 1 - out_idx] = static_cast<int64_t>(rev);
                out_idx++;
            }
        }

        // Fill remaining slots with padding
        if (num_tokens_out < max_len) {
            std::fill(row_fwd + num_tokens_out, row_fwd + max_len, pad_value);
            std::fill(row_rev + num_tokens_out, row_rev + max_len, pad_value);
        }
    }
}

// ============================================================================
// Batch AA tokenization from shared buffer (OpenMP parallelized)
// ============================================================================

inline void batch_tokenize_aa(
    const uint8_t* buffer,
    const int64_t* offsets,
    const int64_t* lengths,
    size_t num_seqs,
    int64_t* out,
    size_t max_len,
    int64_t pad_value,
    int k, int stride,
    const int64_t* start_indices)  // nullable, defaults to 0
{
    constexpr uint64_t base = 29;

    #pragma omp parallel for schedule(guided)
    for (size_t seq_idx = 0; seq_idx < num_seqs; seq_idx++) {
        int64_t* row = out + seq_idx * max_len;
        std::fill(row, row + max_len, pad_value);

        size_t offset = static_cast<size_t>(offsets[seq_idx]);
        size_t len = static_cast<size_t>(lengths[seq_idx]);
        const uint8_t* seq = buffer + offset;

        if (len < static_cast<size_t>(k)) continue;

        size_t start = start_indices ? static_cast<size_t>(start_indices[seq_idx]) : 0;

        size_t out_idx = 0;
        for (size_t i = start; i + k <= len && out_idx < max_len; i += stride) {
            uint64_t value = 0;
            for (int j = 0; j < k; j++) {
                value = value * base + AA_TABLE[seq[i + j]];
            }
            row[out_idx++] = static_cast<int64_t>(value);
        }
    }
}

// ============================================================================
// Batch DNA tokenization - variable length output (returns vector of vectors)
// ============================================================================

inline std::vector<std::vector<int64_t>> batch_tokenize_dna_variable(
    const uint8_t* buffer,
    const int64_t* offsets,
    const int64_t* lengths,
    size_t num_seqs,
    int k, int stride,
    bool use_iupac,
    StrandMode strand)
{
    const auto& table = use_iupac ? DNA_IUPAC : DNA_ACGTN;
    const auto& comp = use_iupac ? COMP_IUPAC : COMP_ACGTN;
    uint64_t base = use_iupac ? 16 : 6;
    PowerCache<32> power_cache(base);

    std::vector<std::vector<int64_t>> results(num_seqs);

    #pragma omp parallel for schedule(guided)
    for (size_t seq_idx = 0; seq_idx < num_seqs; seq_idx++) {
        size_t offset = static_cast<size_t>(offsets[seq_idx]);
        size_t len = static_cast<size_t>(lengths[seq_idx]);
        const uint8_t* seq = buffer + offset;

        std::vector<int64_t>& seq_tokens = results[seq_idx];

        if (len < static_cast<size_t>(k)) {
            continue;  // Empty vector for too-short sequences
        }

        // Calculate number of tokens
        size_t num_tokens = (len - k) / stride + 1;
        seq_tokens.reserve(num_tokens);

        // Optimized path for stride=1 with canonical/revcomp using rolling window
        if (stride == 1 && strand != StrandMode::FORWARD) {
            RollingDnaCanon<decltype(table), decltype(comp)> roller(table, comp, base, k);
            for (size_t i = 0; i < len; i++) {
                roller.push(seq[i]);
                if (roller.ready()) {
                    switch (strand) {
                        case StrandMode::REVCOMP:
                            seq_tokens.push_back(roller.revcomp());
                            break;
                        case StrandMode::CANONICAL:
                            seq_tokens.push_back(roller.canonical());
                            break;
                        default:
                            break;
                    }
                }
            }
        } else {
            // Naive loop for stride > 1 or forward-only
            for (size_t i = 0; i + k <= len; i += stride) {
                uint64_t fwd = 0, rev = 0;
                for (int j = 0; j < k; j++) {
                    uint8_t code = table[seq[i + j]];
                    fwd = fwd * base + code;
                    if (strand != StrandMode::FORWARD) {
                        uint8_t comp_code = comp[code];
                        rev = rev + comp_code * power_cache[j];
                    }
                }

                switch (strand) {
                    case StrandMode::FORWARD:
                        seq_tokens.push_back(static_cast<int64_t>(fwd));
                        break;
                    case StrandMode::REVCOMP:
                        seq_tokens.push_back(static_cast<int64_t>(rev));
                        break;
                    case StrandMode::CANONICAL:
                        seq_tokens.push_back(static_cast<int64_t>(std::min(fwd, rev)));
                        break;
                    case StrandMode::BOTH:
                        // Handled in separate function
                        break;
                }
            }
        }
    }

    return results;
}

// ============================================================================
// Batch DNA both strands - variable length output (returns pair of vector of vectors)
// ============================================================================

inline std::pair<std::vector<std::vector<int64_t>>, std::vector<std::vector<int64_t>>>
batch_tokenize_dna_both_variable(
    const uint8_t* buffer,
    const int64_t* offsets,
    const int64_t* lengths,
    size_t num_seqs,
    int k, int stride,
    bool use_iupac)
{
    const auto& table = use_iupac ? DNA_IUPAC : DNA_ACGTN;
    const auto& comp = use_iupac ? COMP_IUPAC : COMP_ACGTN;
    uint64_t base = use_iupac ? 16 : 6;
    PowerCache<32> power_cache(base);

    std::vector<std::vector<int64_t>> results_fwd(num_seqs);
    std::vector<std::vector<int64_t>> results_rev(num_seqs);

    #pragma omp parallel for schedule(guided)
    for (size_t seq_idx = 0; seq_idx < num_seqs; seq_idx++) {
        size_t offset = static_cast<size_t>(offsets[seq_idx]);
        size_t len = static_cast<size_t>(lengths[seq_idx]);
        const uint8_t* seq = buffer + offset;

        std::vector<int64_t>& fwd_tokens = results_fwd[seq_idx];
        std::vector<int64_t>& rev_tokens = results_rev[seq_idx];

        if (len < static_cast<size_t>(k)) {
            continue;  // Empty vectors for too-short sequences
        }

        size_t num_tokens = (len - k) / stride + 1;
        fwd_tokens.reserve(num_tokens);
        rev_tokens.reserve(num_tokens);

        // Optimized path for stride=1 using rolling window
        if (stride == 1) {
            RollingDnaCanon<decltype(table), decltype(comp)> roller(table, comp, base, k);
            for (size_t i = 0; i < len; i++) {
                roller.push(seq[i]);
                if (roller.ready()) {
                    fwd_tokens.push_back(roller.forward());
                    rev_tokens.push_back(roller.revcomp());
                }
            }
        } else {
            // Naive loop for stride > 1
            for (size_t i = 0; i + k <= len; i += stride) {
                uint64_t fwd = 0, rev = 0;
                for (int j = 0; j < k; j++) {
                    uint8_t code = table[seq[i + j]];
                    uint8_t comp_code = comp[code];
                    fwd = fwd * base + code;
                    rev = rev + comp_code * power_cache[j];
                }
                fwd_tokens.push_back(static_cast<int64_t>(fwd));
                rev_tokens.push_back(static_cast<int64_t>(rev));
            }
        }

        // Reverse to return tokens in reverse-complement order
        std::reverse(rev_tokens.begin(), rev_tokens.end());
    }

    return std::make_pair(results_fwd, results_rev);
}

// ============================================================================
// Batch AA tokenization - variable length output (returns vector of vectors)
// ============================================================================

inline std::vector<std::vector<int64_t>> batch_tokenize_aa_variable(
    const uint8_t* buffer,
    const int64_t* offsets,
    const int64_t* lengths,
    size_t num_seqs,
    int k, int stride)
{
    constexpr uint64_t base = 29;

    std::vector<std::vector<int64_t>> results(num_seqs);

    #pragma omp parallel for schedule(guided)
    for (size_t seq_idx = 0; seq_idx < num_seqs; seq_idx++) {
        size_t offset = static_cast<size_t>(offsets[seq_idx]);
        size_t len = static_cast<size_t>(lengths[seq_idx]);
        const uint8_t* seq = buffer + offset;

        std::vector<int64_t>& seq_tokens = results[seq_idx];

        if (len < static_cast<size_t>(k)) {
            continue;  // Empty vector for too-short sequences
        }

        size_t num_tokens = (len - k) / stride + 1;
        seq_tokens.reserve(num_tokens);

        for (size_t i = 0; i + k <= len; i += stride) {
            uint64_t value = 0;
            for (int j = 0; j < k; j++) {
                value = value * base + AA_TABLE[seq[i + j]];
            }
            seq_tokens.push_back(static_cast<int64_t>(value));
        }
    }

    return results;
}

// ============================================================================
// Batch hash tokens (OpenMP parallelized)
// ============================================================================

inline void batch_hash_tokens(
    const int64_t* in,
    int64_t* out,
    size_t total,
    int64_t num_buckets)
{
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < total; i++) {
        out[i] = hash_kmer(in[i], num_buckets);
    }
}

} // namespace bioenc
