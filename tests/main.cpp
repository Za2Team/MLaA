#include <iostream>
#include <vector>
#include <mlaa/CpuTopology.hpp>
#include "benchmark_suite.hpp"

int main() {
    // 1. Hardware inspection
    const auto& cpu = mlaa::CpuTopology::get();

    std::cout << "========================================\n";
    std::cout << "           CPU HARDWARE TOPOLOGY        \n";
    std::cout << "========================================\n";
    std::cout << "Physical Cores : " << cpu.physical_cores << "\n";
    std::cout << "Logical Cores  : " << cpu.logical_cores << "\n";
    std::cout << "L1 Data Cache  : " << (cpu.l1d_size / 1024) << " KB\n";
    std::cout << "L2 Cache       : " << (cpu.l2_size / 1024) << " KB\n";
    std::cout << "L3 Cache       : " << (cpu.l3_size / (1024 * 1024)) << " MB\n";
    std::cout << "SIMD Support   : " << cpu.simd_name() << "\n";
    std::cout << "========================================\n\n";

    std::vector<mlaa::bench::TestResult> results;
    results.reserve(12);

    // 2. Run all 12 dataset benchmarks
    // int32_t
    results.push_back(mlaa::bench::execute_case<int32_t>("data_uniform_int32.bin", "int32_t", "Uniform"));
    results.push_back(mlaa::bench::execute_case<int32_t>("data_gaussian_int32.bin", "int32_t", "Gaussian"));

    // uint32_t
    results.push_back(mlaa::bench::execute_case<uint32_t>("data_uniform_uint32.bin", "uint32_t", "Uniform"));
    results.push_back(mlaa::bench::execute_case<uint32_t>("data_gaussian_uint32.bin", "uint32_t", "Gaussian"));

    // int64_t
    results.push_back(mlaa::bench::execute_case<int64_t>("data_uniform_int64.bin", "int64_t", "Uniform"));
    results.push_back(mlaa::bench::execute_case<int64_t>("data_gaussian_int64.bin", "int64_t", "Gaussian"));

    // uint64_t
    results.push_back(mlaa::bench::execute_case<uint64_t>("data_uniform_uint64.bin", "uint64_t", "Uniform"));
    results.push_back(mlaa::bench::execute_case<uint64_t>("data_gaussian_uint64.bin", "uint64_t", "Gaussian"));

    // float
    results.push_back(mlaa::bench::execute_case<float>("data_uniform_float.bin", "float", "Uniform"));
    results.push_back(mlaa::bench::execute_case<float>("data_gaussian_float.bin", "float", "Gaussian"));

    // double
    results.push_back(mlaa::bench::execute_case<double>("data_uniform_double.bin", "double", "Uniform"));
    results.push_back(mlaa::bench::execute_case<double>("data_gaussian_double.bin", "double", "Gaussian"));

    // 3. Print final comparative summary
    mlaa::bench::print_summary_report(results);

    // Return non-zero if any test failed
    for (const auto& r : results) {
        if (!r.sort_passed || !r.rmi_passed) return 1;
    }

    return 0;
}