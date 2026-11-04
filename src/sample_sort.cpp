#include "sample_sort.hpp"

namespace mlaa {

namespace {

template <typename T>
void sample_sort_internal_impl(T* MLAA_RESTRICT src,
                               T* MLAA_RESTRICT tmp,
                               uint8_t* MLAA_RESTRICT oracle,
                               size_t n,
                               const detail::CacheTuningParams<T>& topo) {
    if (n <= topo.base_case_threshold) {
        std::sort(src, src + n);
        return;
    }

    constexpr size_t PREFETCH_DISTANCE = 512 / sizeof(T);

    if (n < 32768) {
        constexpr int K = 16;
        constexpr int ALPHA = 16;
        constexpr size_t NUM_SAMPLES = K * ALPHA - 1;

        alignas(64) T samples[NUM_SAMPLES];
        const size_t step = n / NUM_SAMPLES;

        MLAA_UNROLL
        for (size_t i = 0; i < NUM_SAMPLES; ++i) {
            samples[i] = src[i * step];
        }
        std::sort(samples, samples + NUM_SAMPLES);

        alignas(64) T splitters[K - 1];
        for (int i = 0; i < K - 1; ++i) {
            splitters[i] = samples[(i + 1) * ALPHA - 1];
        }

        if (splitters[0] == splitters[K - 2]) {
            std::sort(src, src + n);
            return;
        }

        alignas(64) detail::DecisionTree<T, K> tree;
        tree.build(splitters);

        size_t bucket_counts[K] = {0};
        const size_t n_vec = n & ~size_t(7);
        size_t i = 0;

        for (; i < n_vec; i += 8) {
            if ((i & 63) == 0 && i + PREFETCH_DISTANCE < n) {
                _mm_prefetch(reinterpret_cast<const char*>(src + i + PREFETCH_DISTANCE), _MM_HINT_T0);
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
                if constexpr (std::is_trivially_copyable_v<T>) {
                    std::memcpy(src + start, tmp + start, len * sizeof(T));
                } else {
                    std::move(tmp + start, tmp + start + len, src + start);
                }
                std::sort(src + start, src + start + len);
            } else {
                sample_sort_internal_impl(tmp + start, src + start, oracle + start, len, topo);
                if constexpr (std::is_trivially_copyable_v<T>) {
                    std::memcpy(src + start, tmp + start, len * sizeof(T));
                } else {
                    std::move(tmp + start, tmp + start + len, src + start);
                }
            }
        }
    } else {
        constexpr int K = 256;
        constexpr int ALPHA = 16;
        constexpr size_t NUM_SAMPLES = K * ALPHA - 1;

        alignas(64) T samples[NUM_SAMPLES];
        const size_t step = n / NUM_SAMPLES;
        for (size_t i = 0; i < NUM_SAMPLES; ++i) {
            samples[i] = src[i * step];
        }
        std::sort(samples, samples + NUM_SAMPLES);

        alignas(64) T splitters[K - 1];
        for (int i = 0; i < K - 1; ++i) {
            splitters[i] = samples[(i + 1) * ALPHA - 1];
        }

        if (splitters[0] == splitters[K - 2]) {
            std::sort(src, src + n);
            return;
        }

        alignas(64) detail::DecisionTree<T, K> tree;
        tree.build(splitters);

        size_t bucket_counts[K] = {0};
        const size_t n_vec = n & ~size_t(7);
        size_t i = 0;

        for (; i < n_vec; i += 8) {
            if ((i & 63) == 0 && i + PREFETCH_DISTANCE < n) {
                _mm_prefetch(reinterpret_cast<const char*>(src + i + PREFETCH_DISTANCE), _MM_HINT_T0);
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
                if constexpr (std::is_trivially_copyable_v<T>) {
                    std::memcpy(src + start, tmp + start, len * sizeof(T));
                } else {
                    std::move(tmp + start, tmp + start + len, src + start);
                }
                std::sort(src + start, src + start + len);
            } else {
                sample_sort_internal_impl(tmp + start, src + start, oracle + start, len, topo);
                if constexpr (std::is_trivially_copyable_v<T>) {
                    std::memcpy(src + start, tmp + start, len * sizeof(T));
                } else {
                    std::move(tmp + start, tmp + start + len, src + start);
                }
            }
        }
    }
}

} // anonymous namespace

template <typename T>
void sample_sort_internal(T* MLAA_RESTRICT src,
                          T* MLAA_RESTRICT tmp,
                          uint8_t* MLAA_RESTRICT oracle,
                          size_t n) {
    const auto& topo = detail::CacheTuningParams<T>::get();
    sample_sort_internal_impl(src, tmp, oracle, n, topo);
}

template <typename T>
void sample_sort_internal(T* MLAA_RESTRICT src,
                          T* MLAA_RESTRICT tmp,
                          size_t n) {
    const auto& topo = detail::CacheTuningParams<T>::get();
    if (n <= topo.base_case_threshold) {
        std::sort(src, src + n);
        return;
    }
    detail::HeapBuffer<uint8_t> oracle_buffer(n);
    sample_sort_internal_impl(src, tmp, oracle_buffer.get(), n, topo);
}

// Explicit Template Instantiations
#define INSTANTIATE_SAMPLE_SORT(Type)                                                      \
    template void sample_sort_internal<Type>(Type*, Type*, uint8_t*, size_t);             \
    template void sample_sort_internal<Type>(Type*, Type*, size_t);                       \
    template size_t detail::get_base_case_threshold<Type>() noexcept;

INSTANTIATE_SAMPLE_SORT(int64_t)
INSTANTIATE_SAMPLE_SORT(uint64_t)
INSTANTIATE_SAMPLE_SORT(int32_t)
INSTANTIATE_SAMPLE_SORT(uint32_t)
INSTANTIATE_SAMPLE_SORT(double)
INSTANTIATE_SAMPLE_SORT(float)

#undef INSTANTIATE_SAMPLE_SORT

} // namespace mlaa