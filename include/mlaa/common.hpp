#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <concepts>
#include <limits>
#include <immintrin.h>

#if defined(_MSC_VER)
    #define MLAA_FORCE_INLINE [[msvc::forceinline]] inline
    #define MLAA_RESTRICT __restrict
    #define MLAA_UNROLL
#elif defined(__clang__)
    #define MLAA_FORCE_INLINE __attribute__((always_inline)) inline
    #define MLAA_RESTRICT __restrict__
    #define MLAA_UNROLL _Pragma("unroll")
#elif defined(__GNUC__)
    #define MLAA_FORCE_INLINE __attribute__((always_inline)) inline
    #define MLAA_RESTRICT __restrict__
    #define MLAA_UNROLL _Pragma("GCC unroll 8")
#else
    #define MLAA_FORCE_INLINE inline
    #define MLAA_RESTRICT
    #define MLAA_UNROLL
#endif

namespace mlaa {

MLAA_FORCE_INLINE float horizontal_add_avx(__m256 v) noexcept {
    __m128 vlow  = _mm256_castps256_ps128(v);        
    __m128 vhigh = _mm256_extractf128_ps(v, 1);      
    __m128 sum4  = _mm_add_ps(vlow, vhigh);          

    __m128 shuf  = _mm_movehdup_ps(sum4);            
    __m128 sum2  = _mm_add_ps(sum4, shuf);          

    __m128 shuf2 = _mm_movehl_ps(shuf, sum2);        
    __m128 sum1  = _mm_add_ss(sum2, shuf2);         
    return _mm_cvtss_f32(sum1); 
}

MLAA_FORCE_INLINE double horizontal_add_avx_pd(__m256d v) noexcept {
    __m128d vlow  = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1);
    __m128d sum2  = _mm_add_pd(vlow, vhigh);
    __m128d shuf  = _mm_unpackhi_pd(sum2, sum2);
    __m128d sum1  = _mm_add_sd(sum2, shuf);
    return _mm_cvtsd_f64(sum1);
}

MLAA_FORCE_INLINE float horizontal_max_avx(__m256 v) noexcept {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 max4  = _mm_max_ps(vlow, vhigh);

    __m128 shuf  = _mm_movehdup_ps(max4);
    __m128 max2  = _mm_max_ps(max4, shuf);

    __m128 shuf2 = _mm_movehl_ps(shuf, max2);
    __m128 max1  = _mm_max_ss(max2, shuf2);
    return _mm_cvtss_f32(max1);
}

MLAA_FORCE_INLINE double horizontal_max_avx_pd(__m256d v) noexcept {
    __m128d vlow  = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1);
    __m128d max2  = _mm_max_pd(vlow, vhigh);
    __m128d shuf  = _mm_unpackhi_pd(max2, max2);
    __m128d max1  = _mm_max_sd(max2, shuf);
    return _mm_cvtsd_f64(max1);
}

} // namespace mlaa