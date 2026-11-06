#pragma once
#include <cstddef>
#include <cstdint>

namespace mlaa
{

enum class SimdLevel : std::uint8_t {
    None = 0,
    SSE2,
    AVX,
    AVX2,
    AVX512
};

struct CpuTopology {
    std::size_t l1d_size{0};
    std::size_t l2_size{0};
    std::size_t l3_size{0};
    std::size_t line_size{64};

    std::size_t physical_cores{1};
    std::size_t logical_cores{1};

    bool has_avx{false};
    bool has_avx2{false};
    bool has_fma{false};
    bool has_avx512f{false};    // AVX-512 Foundation
    bool has_avx512vl{false};   // Vector Length Extensions
    bool has_avx512bw{false};   // Byte and Word Instructions
    bool has_avx512dq{false};   // Doubleword and Quadword Instructions
    bool has_avx512cd{false};   // Conflict Detection

    SimdLevel highest_simd{SimdLevel::None};

    [[nodiscard]] const char* simd_name() const noexcept {
        switch (highest_simd) {
            case SimdLevel::AVX512: return "AVX-512";
            case SimdLevel::AVX2:   return "AVX2";
            case SimdLevel::AVX:    return "AVX";
            case SimdLevel::SSE2:   return "SSE2";
            default:                return "None / Baseline";
        }
    }

    // (Thread-Safe Magic Static)
    static const CpuTopology& get() noexcept;
};

} // namespace mlaa