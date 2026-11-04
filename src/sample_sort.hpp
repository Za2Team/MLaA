#pragma once

#include <mlaa/sample_sort.hpp>
#include <mlaa/CpuTopology.hpp>
#include <immintrin.h>
#include <cstring>
#include <algorithm>
#include <concepts>
#include <type_traits>

namespace mlaa::detail {

template <typename T>
struct alignas(64) CacheTuningParams {
    size_t l1d_size = 32768;
    size_t l2_size = 1048576;
    size_t l3_size = 16777216;
    size_t line_size = 64;
    size_t base_case_threshold = 1024;

    static const CacheTuningParams& get() noexcept {
        static const CacheTuningParams instance = []() {
            CacheTuningParams params;
            const auto& cpu = mlaa::CpuTopology::get();
            if (cpu.l1d_size > 0) params.l1d_size = cpu.l1d_size;
            if (cpu.l2_size > 0) params.l2_size = cpu.l2_size;
            if (cpu.l3_size > 0) params.l3_size = cpu.l3_size;
            if (cpu.line_size > 0) params.line_size = cpu.line_size;

            // محاسبه آستانه بیسکیس متناسب با اندازه بایت دیتا
            size_t optimal_base = (params.l1d_size / (2 * sizeof(T)));
            if (optimal_base < 512) optimal_base = 512;
            if (optimal_base > 4096) optimal_base = 4096;
            params.base_case_threshold = optimal_base;

            return params;
        }();
        return instance;
    }
};

template <typename T>
inline size_t get_base_case_threshold() noexcept {
    return CacheTuningParams<T>::get().base_case_threshold;
}

template <typename T, int K>
struct alignas(64) DecisionTree {
    static_assert(K == 256 || K == 16, "K must be 256 or 16");
    static constexpr int LOG_K = (K == 256) ? 8 : 4;
    T tree[K];

    MLAA_FORCE_INLINE void build(const T* MLAA_RESTRICT splitters) noexcept {
        int node = 1;
        for (int d = 0; d < LOG_K; ++d) {
            const int step = K >> (d + 1);
            const int count = 1 << d;
            for (int j = 0; j < count; ++j) {
                tree[node++] = splitters[(2 * j + 1) * step - 1];
            }
        }
    }

    MLAA_FORCE_INLINE int classify_single(T val) const noexcept {
        int b = 1;
        MLAA_UNROLL
        for (int i = 0; i < LOG_K; ++i) {
            b = (b << 1) | static_cast<int>(val > tree[b]);
        }
        return b - K;
    }

    // (8-Way Branchless ILP)
    MLAA_FORCE_INLINE void classify_8(const T* MLAA_RESTRICT src_chunk,
                                      uint8_t* MLAA_RESTRICT oracle_chunk) const noexcept {
        const T v0 = src_chunk[0], v1 = src_chunk[1], v2 = src_chunk[2], v3 = src_chunk[3];
        const T v4 = src_chunk[4], v5 = src_chunk[5], v6 = src_chunk[6], v7 = src_chunk[7];

        const T root = tree[1];
        int b0 = 2 + static_cast<int>(v0 > root);
        int b1 = 2 + static_cast<int>(v1 > root);
        int b2 = 2 + static_cast<int>(v2 > root);
        int b3 = 2 + static_cast<int>(v3 > root);
        int b4 = 2 + static_cast<int>(v4 > root);
        int b5 = 2 + static_cast<int>(v5 > root);
        int b6 = 2 + static_cast<int>(v6 > root);
        int b7 = 2 + static_cast<int>(v7 > root);

        MLAA_UNROLL
        for (int lvl = 1; lvl < LOG_K; ++lvl) {
            b0 = (b0 << 1) | static_cast<int>(v0 > tree[b0]);
            b1 = (b1 << 1) | static_cast<int>(v1 > tree[b1]);
            b2 = (b2 << 1) | static_cast<int>(v2 > tree[b2]);
            b3 = (b3 << 1) | static_cast<int>(v3 > tree[b3]);
            b4 = (b4 << 1) | static_cast<int>(v4 > tree[b4]);
            b5 = (b5 << 1) | static_cast<int>(v5 > tree[b5]);
            b6 = (b6 << 1) | static_cast<int>(v6 > tree[b6]);
            b7 = (b7 << 1) | static_cast<int>(v7 > tree[b7]);
        }

        oracle_chunk[0] = static_cast<uint8_t>(b0 - K);
        oracle_chunk[1] = static_cast<uint8_t>(b1 - K);
        oracle_chunk[2] = static_cast<uint8_t>(b2 - K);
        oracle_chunk[3] = static_cast<uint8_t>(b3 - K);
        oracle_chunk[4] = static_cast<uint8_t>(b4 - K);
        oracle_chunk[5] = static_cast<uint8_t>(b5 - K);
        oracle_chunk[6] = static_cast<uint8_t>(b6 - K);
        oracle_chunk[7] = static_cast<uint8_t>(b7 - K);
    }
};

} // namespace mlaa::detail