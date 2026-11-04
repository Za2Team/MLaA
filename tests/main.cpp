#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <cstdint>

#include <mlaa/CpuTopology.hpp>
#include <mlaa/sample_sort.hpp> // استفاده از هدر عمومی تابع اصلی

int main() {
    const auto& cpu = mlaa::CpuTopology::get();

    std::cout << "========================================\n";
    std::cout << "           CPU HARDWARE TOPOLOGY        \n";
    std::cout << "========================================\n";
    std::cout << "Physical Cores : " << cpu.physical_cores << "\n";
    std::cout << "Logical Cores  : " << cpu.logical_cores << "\n";
    std::cout << "----------------------------------------\n";
    std::cout << "L1 Data Cache  : " << (cpu.l1d_size / 1024) << " KB\n";
    std::cout << "L2 Cache       : " << (cpu.l2_size / 1024) << " KB\n";
    std::cout << "L3 Cache       : " << (cpu.l3_size / (1024 * 1024)) << " MB\n";
    std::cout << "Cache Line     : " << cpu.line_size << " bytes\n";
    std::cout << "----------------------------------------\n";
    std::cout << "Highest SIMD   : " << cpu.simd_name() << "\n";
    std::cout << "AVX Supported  : " << (cpu.has_avx ? "Yes" : "No") << "\n";
    std::cout << "AVX2 Supported : " << (cpu.has_avx2 ? "Yes" : "No") << "\n";
    std::cout << "FMA Supported  : " << (cpu.has_fma ? "Yes" : "No") << "\n";
    std::cout << "AVX-512F       : " << (cpu.has_avx512f ? "Yes" : "No") << "\n";
    std::cout << "AVX-512VL/BW/DQ: " << (cpu.has_avx512vl && cpu.has_avx512bw && cpu.has_avx512dq ? "Yes" : "No") << "\n";
    std::cout << "========================================\n";

    // =========================================================================
    //        BENCHMARK: sample_sort
    // =========================================================================
    std::cout << "\n========================================\n";
    std::cout << "        BENCHMARK: SAMPLE SORT          \n";
    std::cout << "========================================\n";

    constexpr size_t BENCHMARK_ELEMENTS = 200'000;
    std::string bin_path = R"(C:\Users\Lenovo\Desktop\Za2Net Projects\C++\mlaa\benchmark\gaussian_1m.bin)";

    // بررسی مسیر نسبی در صورت اجرا از دایرکتوری build
    if (!std::filesystem::exists(bin_path)) {
        bin_path = "../benchmark/gaussian_1m.bin";
    }

    std::ifstream file(bin_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not open dataset at:\n  " << bin_path << "\n";
        return 1;
    }

    std::vector<int64_t> raw_dataset(BENCHMARK_ELEMENTS);
    file.read(reinterpret_cast<char*>(raw_dataset.data()), BENCHMARK_ELEMENTS * sizeof(int64_t));
    const std::streamsize bytes_read = file.gcount();
    file.close();

    if (bytes_read != static_cast<std::streamsize>(BENCHMARK_ELEMENTS * sizeof(int64_t))) {
        std::cerr << "[ERROR] Expected to read " << BENCHMARK_ELEMENTS * sizeof(int64_t)
                  << " bytes, but read " << bytes_read << " bytes.\n";
        return 1;
    }

    std::cout << "Elements Loaded : " << BENCHMARK_ELEMENTS << " int64_t (~" 
              << (BENCHMARK_ELEMENTS * sizeof(int64_t) / 1024) << " KB)\n";
    std::cout << "Target Memory   : Fitting completely inside L2/L3 Cache\n";
    std::cout << "----------------------------------------\n";

    // ۱. بررسی صحت عملکرد الگوریتم (Correctness Test)
    {
        std::vector<int64_t> test_arr = raw_dataset;

        // فراخوانی مستقیم API اصلی
        mlaa::sample_sort(test_arr.begin(), test_arr.end());

        if (!std::is_sorted(test_arr.begin(), test_arr.end())) {
            std::cerr << "[FAIL] Output is NOT sorted correctly!\n";
            return 1;
        }
        std::cout << "Verification    : PASSED (std::is_sorted == true)\n";
    }

    // ۲. بنچمارک دقیق ۱۰ تکرار برای mlaa::sample_sort
    constexpr int ITERATIONS = 10;
    std::vector<double> sample_sort_times;
    sample_sort_times.reserve(ITERATIONS);

    std::vector<int64_t> work_buffer(BENCHMARK_ELEMENTS);

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        std::memcpy(work_buffer.data(), raw_dataset.data(), BENCHMARK_ELEMENTS * sizeof(int64_t));

        const auto start = std::chrono::high_resolution_clock::now();
        mlaa::sample_sort(work_buffer.begin(), work_buffer.end());
        const auto end = std::chrono::high_resolution_clock::now();

        const double elapsed_us = std::chrono::duration<double, std::micro>(end - start).count();
        sample_sort_times.push_back(elapsed_us);
    }

    const double min_ss_us = *std::min_element(sample_sort_times.begin(), sample_sort_times.end());
    const double avg_ss_us = std::accumulate(sample_sort_times.begin(), sample_sort_times.end(), 0.0) / ITERATIONS;

    // ۳. بنچمارک مقایسه‌ای با std::sort بر روی همان دیتا
    std::vector<double> std_sort_times;
    std_sort_times.reserve(ITERATIONS);

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        std::memcpy(work_buffer.data(), raw_dataset.data(), BENCHMARK_ELEMENTS * sizeof(int64_t));

        const auto start = std::chrono::high_resolution_clock::now();
        std::sort(work_buffer.begin(), work_buffer.end());
        const auto end = std::chrono::high_resolution_clock::now();

        const double elapsed_us = std::chrono::duration<double, std::micro>(end - start).count();
        std_sort_times.push_back(elapsed_us);
    }

    const double min_std_us = *std::min_element(std_sort_times.begin(), std_sort_times.end());
    const double avg_std_us = std::accumulate(std_sort_times.begin(), std_sort_times.end(), 0.0) / ITERATIONS;

    const double speedup = min_std_us / min_ss_us;
    const double throughput_melem = (BENCHMARK_ELEMENTS / (min_ss_us * 1e-6)) / 1e6;

    std::cout << "----------------------------------------\n";
    std::cout << "mlaa::sample_sort (Min) : " << min_ss_us << " us  (" << (min_ss_us / 1000.0) << " ms)\n";
    std::cout << "mlaa::sample_sort (Avg) : " << avg_ss_us << " us\n";
    std::cout << "std::sort         (Min) : " << min_std_us << " us  (" << (min_std_us / 1000.0) << " ms)\n";
    std::cout << "Throughput              : " << throughput_melem << " M elements/sec\n";
    std::cout << "Performance Gain        : " << speedup << "x faster than std::sort\n";
    std::cout << "========================================\n";

    return 0;
}