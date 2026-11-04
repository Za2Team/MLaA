cmake_minimum_required(VERSION 3.20)
project(LearnedBenchmark LANGUAGES CXX)

# استاندارد C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# تنظیمات بهینه‌سازی پردازنده
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(-O3 -march=native -Wall -Wextra)
endif()

include(FetchContent)

# غیرفعال کردن تست‌های داخلی برای سرعت ساخت
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)

# دانلود مستقیم فایل فشرده بدون نیاز به هیچ‌گونه وابستگی به Git
FetchContent_Declare(
    googlebenchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.9.1.zip
)
FetchContent_MakeAvailable(googlebenchmark)

# ساخت فایل اجرایی
add_executable(bench_runner bench_main.cpp)

# لینک کتابخانه‌های لازم در ویندوز
target_link_libraries(bench_runner PRIVATE benchmark::benchmark)
if(WIN32)
    target_link_libraries(bench_runner PRIVATE shlwapi ws2_32)
endif()