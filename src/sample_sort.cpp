#include "sample_sort.hpp"

namespace mlaa {

namespace {

// تابع بازگشتی است و نباید force inline باشد تا GCC خطای recursive inlining ندهد
void sample_sort_internal_impl(int64_t* MLAA_RESTRICT src,
                               int64_t* MLAA_RESTRICT tmp,
                               uint8_t* MLAA_RESTRICT oracle,
                               size_t n,
                               const detail::CacheTuningParams& topo) {
    if (n <= topo.base_case_threshold) {
        std::sort(src, src + n);
        return;
    }

    if (n < 32768) {
        constexpr int K = 16;
        constexpr int ALPHA = 16;
        constexpr size_t NUM_SAMPLES = K * ALPHA - 1;

        alignas(64) int64_t samples[NUM_SAMPLES];
        const size_t step = n / NUM_SAMPLES;

        MLAA_UNROLL
        for (size_t i = 0; i < NUM_SAMPLES; ++i) {
            samples[i] = src[i * step];
        }
        std::sort(samples, samples + NUM_SAMPLES);

        alignas(64) int64_t splitters[K - 1];
        for (int i = 0; i < K - 1; ++i) {
            splitters[i] = samples[(i + 1) * ALPHA - 1];
        }

        if (splitters[0] == splitters[K - 2]) {
            std::sort(src, src + n);
            return;
        }

        alignas(64) detail::DecisionTree<K> tree;
        tree.build(splitters);

        size_t bucket_counts[K] = {0};
        const size_t n_vec = n & ~size_t(7);
        size_t i = 0;

        for (; i < n_vec; i += 8) {
            if ((i & 63) == 0 && i + 64 < n) {
                _mm_prefetch(reinterpret_cast<const char*>(src + i + 64), _MM_HINT_T0);
            }
            tree.classify_8(src + i, oracle + i);
            bucket_counts[oracle[i]]++;
            bucket_counts[oracle[i + 1]]++;
            bucket_counts[oracle[i + 2]]++;
            bucket_counts[oracle[i + 3]]++;
            bucket_counts[oracle[i + 4]]++;
            bucket_counts[oracle[i + 5]]++;
            bucket_counts[oracle[i + 6]]++;
            bucket_counts[oracle[i + 7]]++;
        }
        for (; i < n; ++i) {
            uint8_t b = static_cast<uint8_t>(tree.classify_single(src[i]));
            oracle[i] = b;
            bucket_counts[b]++;
        }

        size_t bucket_offsets[K + 1] = {0};
        for (int b = 0; b < K; ++b) {
            bucket_offsets[b + 1] = bucket_offsets[b] + bucket_counts[b];
        }
        size_t write_ptrs[K];
        std::memcpy(write_ptrs, bucket_offsets, sizeof(write_ptrs));

        size_t j = 0;
        const size_t n_dist = n & ~size_t(3);
        for (; j < n_dist; j += 4) {
            tmp[write_ptrs[oracle[j]]++]     = src[j];
            tmp[write_ptrs[oracle[j + 1]]++] = src[j + 1];
            tmp[write_ptrs[oracle[j + 2]]++] = src[j + 2];
            tmp[write_ptrs[oracle[j + 3]]++] = src[j + 3];
        }
        for (; j < n; ++j) {
            tmp[write_ptrs[oracle[j]]++] = src[j];
        }

        for (int b = 0; b < K; ++b) {
            const size_t start = bucket_offsets[b];
            const size_t len = bucket_offsets[b + 1] - start;
            if (len <= 1) {
                if (len == 1) src[start] = tmp[start];
                continue;
            }

            if (len <= topo.base_case_threshold) {
                std::memcpy(src + start, tmp + start, len * sizeof(int64_t));
                std::sort(src + start, src + start + len);
            } else {
                sample_sort_internal_impl(tmp + start, src + start, oracle + start, len, topo);
                std::memcpy(src + start, tmp + start, len * sizeof(int64_t));
            }
        }
    } else {
        constexpr int K = 256;
        constexpr int ALPHA = 16;
        constexpr size_t NUM_SAMPLES = K * ALPHA - 1;

        alignas(64) int64_t samples[NUM_SAMPLES];
        const size_t step = n / NUM_SAMPLES;
        for (size_t i = 0; i < NUM_SAMPLES; ++i) {
            samples[i] = src[i * step];
        }
        std::sort(samples, samples + NUM_SAMPLES);

        alignas(64) int64_t splitters[K - 1];
        for (int i = 0; i < K - 1; ++i) {
            splitters[i] = samples[(i + 1) * ALPHA - 1];
        }

        if (splitters[0] == splitters[K - 2]) {
            std::sort(src, src + n);
            return;
        }

        alignas(64) detail::DecisionTree<K> tree;
        tree.build(splitters);

        size_t bucket_counts[K] = {0};
        const size_t n_vec = n & ~size_t(7);
        size_t i = 0;

        for (; i < n_vec; i += 8) {
            if ((i & 63) == 0 && i + 64 < n) {
                _mm_prefetch(reinterpret_cast<const char*>(src + i + 64), _MM_HINT_T0);
            }
            tree.classify_8(src + i, oracle + i);
            bucket_counts[oracle[i]]++;
            bucket_counts[oracle[i + 1]]++;
            bucket_counts[oracle[i + 2]]++;
            bucket_counts[oracle[i + 3]]++;
            bucket_counts[oracle[i + 4]]++;
            bucket_counts[oracle[i + 5]]++;
            bucket_counts[oracle[i + 6]]++;
            bucket_counts[oracle[i + 7]]++;
        }
        for (; i < n; ++i) {
            uint8_t b = static_cast<uint8_t>(tree.classify_single(src[i]));
            oracle[i] = b;
            bucket_counts[b]++;
        }

        size_t bucket_offsets[K + 1] = {0};
        for (int b = 0; b < K; ++b) {
            bucket_offsets[b + 1] = bucket_offsets[b] + bucket_counts[b];
        }
        size_t write_ptrs[K];
        std::memcpy(write_ptrs, bucket_offsets, sizeof(write_ptrs));

        size_t j = 0;
        const size_t n_dist = n & ~size_t(3);
        for (; j < n_dist; j += 4) {
            tmp[write_ptrs[oracle[j]]++]     = src[j];
            tmp[write_ptrs[oracle[j + 1]]++] = src[j + 1];
            tmp[write_ptrs[oracle[j + 2]]++] = src[j + 2];
            tmp[write_ptrs[oracle[j + 3]]++] = src[j + 3];
        }
        for (; j < n; ++j) {
            tmp[write_ptrs[oracle[j]]++] = src[j];
        }

        for (int b = 0; b < K; ++b) {
            const size_t start = bucket_offsets[b];
            const size_t len = bucket_offsets[b + 1] - start;
            if (len <= 1) {
                if (len == 1) src[start] = tmp[start];
                continue;
            }

            if (len <= topo.base_case_threshold) {
                std::memcpy(src + start, tmp + start, len * sizeof(int64_t));
                std::sort(src + start, src + start + len);
            } else {
                sample_sort_internal_impl(tmp + start, src + start, oracle + start, len, topo);
                std::memcpy(src + start, tmp + start, len * sizeof(int64_t));
            }
        }
    }
}

} // anonymous namespace

void sample_sort_internal(int64_t* MLAA_RESTRICT src,
                          int64_t* MLAA_RESTRICT tmp,
                          uint8_t* MLAA_RESTRICT oracle,
                          size_t n) {
    const auto& topo = detail::CacheTuningParams::get();
    sample_sort_internal_impl(src, tmp, oracle, n, topo);
}

void sample_sort_internal(int64_t* MLAA_RESTRICT src,
                          int64_t* MLAA_RESTRICT tmp,
                          size_t n) {
    const auto& topo = detail::CacheTuningParams::get();
    if (n <= topo.base_case_threshold) {
        std::sort(src, src + n);
        return;
    }
    detail::HeapBuffer<uint8_t> oracle_buffer(n);
    sample_sort_internal_impl(src, tmp, oracle_buffer.get(), n, topo);
}

} // namespace mlaa