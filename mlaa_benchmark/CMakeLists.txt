cmake_minimum_required(VERSION 3.20)
project(SortingDataBenchmark CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Fetch Google Benchmark automatically
FetchContent_Declare(
    google_benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.9.1
)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(google_benchmark)

add_executable(mlaa_benchmark main.cpp)

# Link benchmark library
target_link_libraries(mlaa_benchmark PRIVATE benchmark::benchmark)

# Extreme performance flags
if(MSVC)
    target_compile_options(mlaa_benchmark PRIVATE /O2 /Oi /Ot /GL)
    target_link_options(mlaa_benchmark PRIVATE /LTCG)
else()
    target_compile_options(mlaa_benchmark PRIVATE -O3 -march=native -flto)
    target_link_options(mlaa_benchmark PRIVATE -flto)
endif()