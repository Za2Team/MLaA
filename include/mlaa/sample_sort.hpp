#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <algorithm>
#include <concepts>
#include <type_traits>

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

namespace detail {

// بافر سبک RAII با عملکرد فوق‌سریع و بدون سربار Zero-fill یا تداخل با لینکر
template <typename T>
struct HeapBuffer {
    T* ptr;
    explicit HeapBuffer(size_t n) : ptr(new T[n]) {}
    ~HeapBuffer() { delete[] ptr; }

    HeapBuffer(const HeapBuffer&) = delete;
    HeapBuffer& operator=(const HeapBuffer&) = delete;

    HeapBuffer(HeapBuffer&& o) noexcept : ptr(o.ptr) { o.ptr = nullptr; }
    HeapBuffer& operator=(HeapBuffer&& o) noexcept {
        if (this != &o) {
            delete[] ptr;
            ptr = o.ptr;
            o.ptr = nullptr;
        }
        return *this;
    }

    MLAA_FORCE_INLINE T* get() const noexcept { return ptr; }
    MLAA_FORCE_INLINE T* data() const noexcept { return ptr; }
};

} // namespace detail

void sample_sort_internal(int64_t* MLAA_RESTRICT src,
                          int64_t* MLAA_RESTRICT tmp,
                          uint8_t* MLAA_RESTRICT oracle,
                          size_t n);

void sample_sort_internal(int64_t* MLAA_RESTRICT src,
                          int64_t* MLAA_RESTRICT tmp,
                          size_t n);

template <std::random_access_iterator RandomIt>
MLAA_FORCE_INLINE void sample_sort(RandomIt begin, RandomIt end) {
    const size_t n = static_cast<size_t>(std::distance(begin, end));
    if (n <= 1) [[unlikely]] return;

    using ValueType = typename std::iterator_traits<RandomIt>::value_type;
    static_assert(std::is_same_v<ValueType, int64_t>, "Currently optimized strictly for int64_t");

    int64_t* src = &(*begin);

    constexpr size_t BASE_THRESHOLD = 1024;
    if (n <= BASE_THRESHOLD) [[unlikely]] {
        std::sort(src, src + n);
        return;
    }

    detail::HeapBuffer<int64_t> tmp_buffer(n);
    detail::HeapBuffer<uint8_t> oracle_buffer(n);

    sample_sort_internal(src, tmp_buffer.get(), oracle_buffer.get(), n);
}

} // namespace mlaa