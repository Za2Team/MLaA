#pragma once

#include <mlaa/common.hpp>
#include <mlaa/CpuTopology.hpp>
#include <mlaa/sample_sort.hpp>

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <concepts>
#include <limits>
#include <utility>

namespace mlaa {

struct alignas(32) LeafModel {
    double slope;
    double intercept;
    uint32_t max_error;    
    uint32_t sample_count; 
};

template <typename T>
requires std::is_arithmetic_v<T>
class FastRMI {
public:
    double root_slope{0.0};
    double root_intercept{0.0};
    uint32_t num_leaves{0};
    size_t total_size{0};
    double sampling_ratio{1.0};
    double inv_sampling_ratio{1.0};
    bool degenerate_duplicates{false}; 

    std::vector<LeafModel> leaves;


    FastRMI() = default;

    explicit FastRMI(const std::vector<T>& raw_data, size_t manual_leaves = 0) {
        build_from_raw(raw_data, manual_leaves);
    }

    FastRMI(const std::vector<T>& sorted_sample, size_t full_data_size, size_t manual_leaves, bool is_sorted = true) {
        if (!is_sorted) {
            std::vector<T> copy_sample = sorted_sample;
            sample_sort(copy_sample.begin(), copy_sample.end());
            build_from_sorted_sample(copy_sample, full_data_size, manual_leaves);
        } else {
            build_from_sorted_sample(sorted_sample, full_data_size, manual_leaves);
        }
    }

    void build_from_raw(const std::vector<T>& raw_data, size_t manual_leaves = 0) {
        total_size = raw_data.size();
        if (total_size == 0) [[unlikely]] return;

        const auto& cpu = CpuTopology::get();

        const size_t max_sample_bytes = (cpu.l3_size > 0) ? (cpu.l3_size / 2) : (8 * 1024 * 1024);
        const size_t max_sample_elements = std::max<size_t>(2048, max_sample_bytes / sizeof(T));

        size_t sample_size = 0;
        if (total_size <= 2048) {
            sample_size = total_size;
        } else {
            size_t target = total_size / 64;
            sample_size = std::clamp<size_t>(target, 2048, max_sample_elements);
            if (sample_size > total_size) sample_size = total_size;
        }

        std::vector<T> sample;
        sample.reserve(sample_size);

        if (sample_size == total_size) {
            sample = raw_data;
        } else {
            const double step = static_cast<double>(total_size - 1) / static_cast<double>(sample_size - 1);
            for (size_t i = 0; i < sample_size; ++i) {
                size_t src_idx = static_cast<size_t>(i * step);
                sample.push_back(raw_data[src_idx]);
            }
        }

        sample_sort(sample.begin(), sample.end());

        build_from_sorted_sample(sample, total_size, manual_leaves);
    }

    void build_from_sorted_sample(const std::vector<T>& sample, size_t full_data_size, size_t manual_leaves = 0) {
        const size_t n_sample = sample.size();
        if (n_sample == 0) [[unlikely]] return;

        total_size = (full_data_size >= n_sample) ? full_data_size : n_sample;
        sampling_ratio = static_cast<double>(n_sample) / static_cast<double>(total_size);
        inv_sampling_ratio = 1.0 / sampling_ratio;

        if (manual_leaves > 0) {
            num_leaves = static_cast<uint32_t>(std::clamp<size_t>(manual_leaves, 2, 16384));
        } else {
            size_t est = n_sample / 64;
            num_leaves = static_cast<uint32_t>(std::clamp<size_t>(est, 16, 8192));
        }

        leaves.resize(num_leaves);

        const T min_val = sample.front();
        const T max_val = sample.back();

        if (max_val == min_val) [[unlikely]] {
            root_slope = 0.0;
            root_intercept = 0.0;
            degenerate_duplicates = true;

            const double mid_pos = static_cast<double>(total_size) * 0.5;
            const uint32_t half_err = static_cast<uint32_t>((total_size + 1) / 2);

            for (uint32_t l = 0; l < num_leaves; ++l) {
                leaves[l] = LeafModel{0.0, mid_pos, half_err, static_cast<uint32_t>(n_sample)};
            }
            return;
        }

        const double key_range = static_cast<double>(max_val) - static_cast<double>(min_val);
        root_slope = static_cast<double>(num_leaves - 1) / key_range;
        root_intercept = -static_cast<double>(min_val) * root_slope;

        const double key_step = key_range / static_cast<double>(num_leaves - 1);

        std::vector<size_t> leaf_start(num_leaves + 1, 0);
        size_t cur_sample = 0;
        size_t duplicate_count = 0;

        for (uint32_t l = 0; l < num_leaves; ++l) {
            leaf_start[l] = cur_sample;
            const double next_thresh = static_cast<double>(min_val) + static_cast<double>(l + 1) * key_step;

            while (cur_sample + 4 <= n_sample && static_cast<double>(sample[cur_sample + 3]) < next_thresh) {
                if (sample[cur_sample] == sample[cur_sample + 1]) duplicate_count++;
                if (sample[cur_sample + 1] == sample[cur_sample + 2]) duplicate_count++;
                if (sample[cur_sample + 2] == sample[cur_sample + 3]) duplicate_count++;
                cur_sample += 4;
            }
            while (cur_sample < n_sample && static_cast<double>(sample[cur_sample]) < next_thresh) {
                if (cur_sample > 0 && sample[cur_sample] == sample[cur_sample - 1]) duplicate_count++;
                cur_sample++;
            }
        }
        leaf_start[num_leaves] = n_sample;

        if (duplicate_count > (n_sample * 7) / 10) {
            degenerate_duplicates = true;
        }

        for (uint32_t l = 0; l < num_leaves; ++l) {
            const size_t start = leaf_start[l];
            const size_t end = leaf_start[l + 1];
            const size_t count = end - start;

            if (count == 0) {
                const double pos = static_cast<double>(start) * inv_sampling_ratio;
                const uint32_t safe_err = static_cast<uint32_t>(inv_sampling_ratio) + 2;
                leaves[l] = LeafModel{0.0, pos, safe_err, 0};
                continue;
            }

            const T* ptr = sample.data() + start;
            const double x_diff = static_cast<double>(ptr[count - 1]) - static_cast<double>(ptr[0]);

            double sample_slope = 0.0;
            double sample_intercept = 0.0;
            uint32_t sample_max_err = 0;

            if (count < 16 || x_diff <= 1e-9) {
                if (x_diff > 1e-9) {
                    sample_slope = static_cast<double>(count - 1) / x_diff;
                    sample_intercept = static_cast<double>(start) - static_cast<double>(ptr[0]) * sample_slope;
                    sample_max_err = compute_max_error_scalar(ptr, count, start, sample_slope, sample_intercept);
                } else {
                    sample_slope = 0.0;
                    sample_intercept = static_cast<double>(start) + static_cast<double>(count - 1) * 0.5;
                    sample_max_err = static_cast<uint32_t>((count + 1) / 2);
                }
            } else {
                if constexpr (sizeof(T) == 4) {
                    if (x_diff < 16777216.0) {
                        compute_ols_simd_32bit(ptr, count, start, sample_slope, sample_intercept);
                        sample_max_err = compute_max_error_simd_32bit(ptr, count, start, sample_slope, sample_intercept);
                    } else {
                        compute_ols_simd_64bit(ptr, count, start, sample_slope, sample_intercept);
                        sample_max_err = compute_max_error_simd_64bit(ptr, count, start, sample_slope, sample_intercept);
                    }
                } else {
                    compute_ols_simd_64bit(ptr, count, start, sample_slope, sample_intercept);
                    sample_max_err = compute_max_error_simd_64bit(ptr, count, start, sample_slope, sample_intercept);
                }
            }
            const double scaled_slope = sample_slope * inv_sampling_ratio;
            const double scaled_intercept = sample_intercept * inv_sampling_ratio;
            const uint32_t scaled_error = static_cast<uint32_t>(std::ceil(sample_max_err * inv_sampling_ratio)) + 2;

            leaves[l] = LeafModel{scaled_slope, scaled_intercept, scaled_error, static_cast<uint32_t>(count)};
        }
    }

   
    [[nodiscard]] MLAA_FORCE_INLINE std::pair<size_t, uint32_t> predict(T key) const noexcept {
        const double k = static_cast<double>(key);

        int64_t leaf_idx = static_cast<int64_t>(std::fma(k, root_slope, root_intercept));
        if (leaf_idx < 0) leaf_idx = 0;
        if (static_cast<uint32_t>(leaf_idx) >= num_leaves) leaf_idx = num_leaves - 1;

        const LeafModel& leaf = leaves[leaf_idx];

        const double raw_pos = std::fma(k, leaf.slope, leaf.intercept);
        
        size_t real_pos = 0;
        if (raw_pos > 0.0) [[likely]] {
            real_pos = static_cast<size_t>(raw_pos);
            if (real_pos >= total_size) [[unlikely]] {
                real_pos = total_size > 0 ? (total_size - 1) : 0;
            }
        }

        return {real_pos, leaf.max_error};
    }

    [[nodiscard]] MLAA_FORCE_INLINE bool should_fallback_to_samplesort() const noexcept {
        return degenerate_duplicates;
    }

private:

    MLAA_FORCE_INLINE static __m256 load_8_to_ps(const T* ptr) noexcept {
        if constexpr (std::is_same_v<T, float>) {
            return _mm256_loadu_ps(reinterpret_cast<const float*>(ptr));
        } else if constexpr (sizeof(T) == 4 && std::is_integral_v<T>) {
            __m256i iv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
            return _mm256_cvtepi32_ps(iv);
        } else {
            alignas(32) float tmp[8];
            MLAA_UNROLL
            for (int k = 0; k < 8; ++k) tmp[k] = static_cast<float>(ptr[k]);
            return _mm256_load_ps(tmp);
        }
    }

    MLAA_FORCE_INLINE static __m256d load_4_to_pd(const T* ptr) noexcept {
        if constexpr (std::is_same_v<T, double>) {
            return _mm256_loadu_pd(reinterpret_cast<const double*>(ptr));
        } else if constexpr (sizeof(T) == 4 && std::is_integral_v<T>) {
            __m128i half = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
            return _mm256_cvtepi32_pd(half);
        } else if constexpr (std::is_same_v<T, float>) {
            __m128 half = _mm_loadu_ps(reinterpret_cast<const float*>(ptr));
            return _mm256_cvtps_pd(half);
        } else {
            return _mm256_set_pd(
                static_cast<double>(ptr[3]),
                static_cast<double>(ptr[2]),
                static_cast<double>(ptr[1]),
                static_cast<double>(ptr[0])
            );
        }
    }
    void compute_ols_simd_32bit(const T* MLAA_RESTRICT data, size_t count, size_t start,
                                double& out_slope, double& out_intercept) noexcept {
        const float x0 = static_cast<float>(data[0]);
        const __m256 x0_vec = _mm256_set1_ps(x0);

        __m256 sum_x_vec  = _mm256_setzero_ps();
        __m256 sum_xx_vec = _mm256_setzero_ps();
        __m256 sum_xy_vec = _mm256_setzero_ps();

        __m256 y_idx = _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f);
        const __m256 y_inc8 = _mm256_set1_ps(8.0f);

        size_t i = 0;
        for (; i + 8 <= count; i += 8) {
            __m256 raw_x = load_8_to_ps(data + i);
            __m256 x_val = _mm256_sub_ps(raw_x, x0_vec);

            sum_x_vec  = _mm256_add_ps(sum_x_vec, x_val);
            sum_xx_vec = _mm256_fmadd_ps(x_val, x_val, sum_xx_vec);
            sum_xy_vec = _mm256_fmadd_ps(x_val, y_idx, sum_xy_vec);
            y_idx      = _mm256_add_ps(y_idx, y_inc8);
        }

        double sum_x  = static_cast<double>(horizontal_add_avx(sum_x_vec));
        double sum_xx = static_cast<double>(horizontal_add_avx(sum_xx_vec));
        double sum_xy = static_cast<double>(horizontal_add_avx(sum_xy_vec));

        for (; i < count; ++i) {
            const double x = static_cast<double>(data[i]) - static_cast<double>(x0);
            const double y = static_cast<double>(i);
            sum_x  += x;
            sum_xx += x * x;
            sum_xy += x * y;
        }

        const double n = static_cast<double>(count);
        const double sum_y = 0.5 * n * (n - 1.0);
        const double denom = n * sum_xx - sum_x * sum_x;

        if (std::abs(denom) > 1e-12) {
            out_slope = (n * sum_xy - sum_x * sum_y) / denom;
            const double mean_y = static_cast<double>(start) + (sum_y / n);
            const double mean_x = static_cast<double>(x0) + (sum_x / n);
            out_intercept = mean_y - out_slope * mean_x;
        } else {
            out_slope = 0.0;
            out_intercept = static_cast<double>(start) + 0.5 * (n - 1.0);
        }
    }
    uint32_t compute_max_error_simd_32bit(const T* MLAA_RESTRICT data, size_t count, size_t start,
                                          double slope, double intercept) noexcept {
        const float slope_f = static_cast<float>(slope);
        const float intercept_f = static_cast<float>(intercept);
        const __m256 slope_vec = _mm256_set1_ps(slope_f);
        const __m256 intercept_vec = _mm256_set1_ps(intercept_f);
        const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

        __m256 y_actual = _mm256_set_ps(
            static_cast<float>(start + 7), static_cast<float>(start + 6),
            static_cast<float>(start + 5), static_cast<float>(start + 4),
            static_cast<float>(start + 3), static_cast<float>(start + 2),
            static_cast<float>(start + 1), static_cast<float>(start + 0)
        );
        const __m256 y_inc8 = _mm256_set1_ps(8.0f);
        __m256 max_err_vec  = _mm256_setzero_ps();

        size_t i = 0;
        for (; i + 8 <= count; i += 8) {
            __m256 x_val = load_8_to_ps(data + i);
            __m256 pred = _mm256_fmadd_ps(slope_vec, x_val, intercept_vec);
            __m256 diff = _mm256_sub_ps(pred, y_actual);
            __m256 abs_diff = _mm256_and_ps(diff, abs_mask);

            max_err_vec = _mm256_max_ps(max_err_vec, abs_diff);
            y_actual    = _mm256_add_ps(y_actual, y_inc8);
        }

        float max_err = horizontal_max_avx(max_err_vec);

        for (; i < count; ++i) {
            const double pred = slope * static_cast<double>(data[i]) + intercept;
            const double actual = static_cast<double>(start + i);
            const double diff = std::abs(pred - actual);
            if (diff > max_err) max_err = static_cast<float>(diff);
        }

        return static_cast<uint32_t>(std::ceil(max_err));
    }

    void compute_ols_simd_64bit(const T* MLAA_RESTRICT data, size_t count, size_t start,
                                double& out_slope, double& out_intercept) noexcept {
        const double x0 = static_cast<double>(data[0]);
        const __m256d x0_vec = _mm256_set1_pd(x0);

        __m256d sum_x_vec  = _mm256_setzero_pd();
        __m256d sum_xx_vec = _mm256_setzero_pd();
        __m256d sum_xy_vec = _mm256_setzero_pd();

        __m256d y_idx = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);
        const __m256d y_inc4 = _mm256_set1_pd(4.0);

        size_t i = 0;
        for (; i + 4 <= count; i += 4) {
            __m256d raw_x = load_4_to_pd(data + i);
            __m256d x_val = _mm256_sub_pd(raw_x, x0_vec);

            sum_x_vec  = _mm256_add_pd(sum_x_vec, x_val);
            sum_xx_vec = _mm256_fmadd_pd(x_val, x_val, sum_xx_vec);
            sum_xy_vec = _mm256_fmadd_pd(x_val, y_idx, sum_xy_vec);
            y_idx      = _mm256_add_pd(y_idx, y_inc4);
        }

        double sum_x  = horizontal_add_avx_pd(sum_x_vec);
        double sum_xx = horizontal_add_avx_pd(sum_xx_vec);
        double sum_xy = horizontal_add_avx_pd(sum_xy_vec);

        for (; i < count; ++i) {
            const double x = static_cast<double>(data[i]) - x0;
            const double y = static_cast<double>(i);
            sum_x  += x;
            sum_xx += x * x;
            sum_xy += x * y;
        }

        const double n = static_cast<double>(count);
        const double sum_y = 0.5 * n * (n - 1.0);
        const double denom = n * sum_xx - sum_x * sum_x;

        if (std::abs(denom) > 1e-12) {
            out_slope = (n * sum_xy - sum_x * sum_y) / denom;
            const double mean_y = static_cast<double>(start) + (sum_y / n);
            const double mean_x = x0 + (sum_x / n);
            out_intercept = mean_y - out_slope * mean_x;
        } else {
            out_slope = 0.0;
            out_intercept = static_cast<double>(start) + 0.5 * (n - 1.0);
        }
    }

    uint32_t compute_max_error_simd_64bit(const T* MLAA_RESTRICT data, size_t count, size_t start,
                                          double slope, double intercept) noexcept {
        const __m256d slope_vec = _mm256_set1_pd(slope);
        const __m256d intercept_vec = _mm256_set1_pd(intercept);
        const __m256d sign_mask = _mm256_set1_pd(-0.0);

        __m256d y_actual = _mm256_set_pd(
            static_cast<double>(start + 3), static_cast<double>(start + 2),
            static_cast<double>(start + 1), static_cast<double>(start + 0)
        );
        const __m256d y_inc4 = _mm256_set1_pd(4.0);
        __m256d max_err_vec  = _mm256_setzero_pd();

        size_t i = 0;
        for (; i + 4 <= count; i += 4) {
            __m256d x_val = load_4_to_pd(data + i);
            __m256d pred = _mm256_fmadd_pd(slope_vec, x_val, intercept_vec);
            __m256d diff = _mm256_sub_pd(pred, y_actual);
            __m256d abs_diff = _mm256_andnot_pd(sign_mask, diff);

            max_err_vec = _mm256_max_pd(max_err_vec, abs_diff);
            y_actual    = _mm256_add_pd(y_actual, y_inc4);
        }

        double max_err = horizontal_max_avx_pd(max_err_vec);

        for (; i < count; ++i) {
            const double pred = slope * static_cast<double>(data[i]) + intercept;
            const double actual = static_cast<double>(start + i);
            const double diff = std::abs(pred - actual);
            if (diff > max_err) max_err = diff;
        }

        return static_cast<uint32_t>(std::ceil(max_err));
    }

    uint32_t compute_max_error_scalar(const T* MLAA_RESTRICT data, size_t count, size_t start,
                                      double slope, double intercept) noexcept {
        double max_err = 0.0;
        for (size_t i = 0; i < count; ++i) {
            const double pred = slope * static_cast<double>(data[i]) + intercept;
            const double actual = static_cast<double>(start + i);
            const double diff = std::abs(pred - actual);
            if (diff > max_err) max_err = diff;
        }
        return static_cast<uint32_t>(std::ceil(max_err));
    }
};

} // namespace mlaa