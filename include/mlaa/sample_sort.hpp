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

template <typename T>
concept SupportedFastNumeric = 
    std::same_as<T, int64_t>  ||
    std::same_as<T, uint64_t> ||
    std::same_as<T, int32_t>  ||
    std::same_as<T, uint32_t> ||
    std::same_as<T, double>   ||
    std::same_as<T, float>;

template <typename T>
concept SortableComparable = std::totally_ordered<T>;

namespace detail {

// RAII
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

template <typename T>
size_t get_base_case_threshold() noexcept;

} // namespace detail

template <typename T>
void sample_sort_internal(T* MLAA_RESTRICT src,
                          T* MLAA_RESTRICT tmp,
                          uint8_t* MLAA_RESTRICT oracle,
                          size_t n);

template <typename T>
void sample_sort_internal(T* MLAA_RESTRICT src,
                          T* MLAA_RESTRICT tmp,
                          size_t n);

// (Compile-Time Type Dispatcher)
template <std::random_access_iterator RandomIt>
MLAA_FORCE_INLINE void sample_sort(RandomIt begin, RandomIt end) {
    const size_t n = static_cast<size_t>(std::distance(begin, end));
    if (n <= 1) [[unlikely]] return;

    using ValueType = typename std::iterator_traits<RandomIt>::value_type;

    if constexpr (SupportedFastNumeric<ValueType>) {
        ValueType* src = &(*begin);
        const size_t base_threshold = detail::get_base_case_threshold<ValueType>();

        if (n <= base_threshold) [[unlikely]] {
            std::sort(src, src + n);
            return;
        }

        detail::HeapBuffer<ValueType> tmp_buffer(n);
        detail::HeapBuffer<uint8_t> oracle_buffer(n);

        sample_sort_internal<ValueType>(src, tmp_buffer.get(), oracle_buffer.get(), n);
    }
    else if constexpr (SortableComparable<ValueType>) {
        std::sort(begin, end);
    } else {
        static_assert(SortableComparable<ValueType>, 
                      "Type must support total ordering via operator< to be sorted.");
    }
}

} // namespace mlaa