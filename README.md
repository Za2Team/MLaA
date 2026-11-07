# MLaA

MLaA is a C++20 library for cache-aware sorting and learned-index experiments. It combines a specialized sample sort for common numeric types with CPU-topology inspection and a piecewise-linear Recursive Model Index (FastRMI).

> **Status:** Early-stage project (version 0.0.1). APIs and performance characteristics may change.

## Features

- **`mlaa::sample_sort`** — cache-tuned sample sort with an optimized path for `int32_t`, `uint32_t`, `int64_t`, `uint64_t`, `float`, and `double`. Other totally ordered types use `std::sort`.
- **`mlaa::FastRMI<T>`** — builds a learned index over sorted data (or samples from raw data) and returns a predicted position with an error bound. Refine the result with a bounded binary search for an exact match.
- **`mlaa::CpuTopology`** — reports CPU core counts, cache sizes, cache-line size, and detected SIMD capabilities.
- CMake build, CTest test target, and a bundled 1-million-element benchmark dataset.

## Requirements

- CMake 3.20 or newer
- A C++20 compiler
- An x86 CPU/toolchain supporting AVX2; the implementation also uses FMA instructions. The supplied CMake configuration enables AVX2 and, on non-MSVC toolchains, `-march=native`. Build for a machine compatible with the generated binary.

## Build and test

```bash
git clone https://github.com/Za2Team/MLaA.git
cd MLaA
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build --output-on-failure
```
