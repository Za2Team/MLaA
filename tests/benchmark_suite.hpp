#pragma once

#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <string>
#include <iomanip>
#include <cstdint>

#include <mlaa/sample_sort.hpp>
#include <mlaa/FastRMI.hpp>

namespace mlaa::bench {

struct TestResult {
    std::string filename;
    std::string type_name;
    std::string dist_name;
    bool sort_passed = false;
    bool rmi_passed  = false;
    size_t error_bound = 0;
    double sort_latency_ms = 0.0;
    double rmi_latency_ns  = 0.0;
};

// Locate dataset file across paths
inline std::string resolve_path(const std::string& filename) {
    const std::vector<std::string> search_paths = {
        filename,
        "../" + filename,
        "../../" + filename,
        "benchmark/" + filename,
        "../benchmark/" + filename,
        "../../benchmark/" + filename,
        R"(C:\Users\Lenovo\Desktop\Za2Net Projects\C++\mlaa\mlaa_benchmark\data\)" + filename
    };

    for (const auto& path : search_paths) {
        if (std::filesystem::exists(path)) {
            return path;
        }
    }
    return filename;
}

// Single file benchmark executor
template <typename T>
TestResult execute_case(const std::string& filename, const std::string& type_name, const std::string& dist_name) {
    TestResult result{filename, type_name, dist_name};
    const std::string full_path = resolve_path(filename);

    if (!std::filesystem::exists(full_path)) {
        std::cerr << "[ERROR] Missing dataset: " << full_path << "\n";
        return result;
    }

    const size_t file_bytes = std::filesystem::file_size(full_path);
    const size_t num_elements = file_bytes / sizeof(T);

    std::vector<T> dataset(num_elements);
    std::ifstream file(full_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open: " << full_path << "\n";
        return result;
    }

    file.read(reinterpret_cast<char*>(dataset.data()), file_bytes);
    file.close();

    // 1. Benchmark & verify Sample Sort
    const auto t_sort_start = std::chrono::high_resolution_clock::now();
    mlaa::sample_sort(dataset.begin(), dataset.end());
    const auto t_sort_end = std::chrono::high_resolution_clock::now();

    result.sort_latency_ms = std::chrono::duration<double, std::milli>(t_sort_end - t_sort_start).count();
    result.sort_passed = std::is_sorted(dataset.begin(), dataset.end());

    if (!result.sort_passed) {
        std::cerr << "[FAIL] Sort failed for: " << filename << "\n";
        return result;
    }

    // 2. Build FastRMI
    mlaa::FastRMI<T> rmi(dataset);

    // 3. Query validation at 55% offset
    const size_t target_index = static_cast<size_t>(num_elements * 0.55);
    const T query_key = dataset[target_index];

    const auto t_query_start = std::chrono::high_resolution_clock::now();
    auto [pred_pos, error_bound] = rmi.predict(query_key);
    const auto t_query_end = std::chrono::high_resolution_clock::now();

    result.rmi_latency_ns = std::chrono::duration<double, std::nano>(t_query_end - t_query_start).count();
    result.error_bound = error_bound;

    // 4. Bounded window check
    const size_t bound_min = (pred_pos > error_bound) ? (pred_pos - error_bound) : 0;
    const size_t bound_max = std::min(static_cast<size_t>(pred_pos + error_bound), num_elements - 1);

    auto it = std::lower_bound(dataset.begin() + bound_min, dataset.begin() + bound_max + 1, query_key);
    result.rmi_passed = (it != (dataset.begin() + bound_max + 1)) && (*it == query_key);

    // Individual report
    std::cout << "------------------------------------------------------------\n";
    std::cout << "Dataset: " << filename << " (" << type_name << ", " << dist_name << ")\n";
    std::cout << "Elements: " << num_elements << " | Elements Size: " << (file_bytes / (1024 * 1024)) << " MB\n";
    std::cout << "Sort Time: " << result.sort_latency_ms << " ms | Sort Status: " << (result.sort_passed ? "PASS" : "FAIL") << "\n";
    std::cout << "Target Index: " << target_index << " | Key: " << query_key << "\n";
    std::cout << "Predicted: " << pred_pos << " | Error Window: +/- " << error_bound << "\n";
    std::cout << "RMI Latency: " << result.rmi_latency_ns << " ns | Verification: " << (result.rmi_passed ? "PASSED" : "FAILED") << "\n";

    return result;
}

// Print comprehensive tabular summary
inline void print_summary_report(const std::vector<TestResult>& results) {
    std::cout << "\n=========================================================================================\n";
    std::cout << "                                AGGREGATED BENCHMARK SUMMARY                             \n";
    std::cout << "=========================================================================================\n";
    std::cout << std::left 
              << std::setw(30) << "Filename"
              << std::setw(10) << "Type"
              << std::setw(11) << "Dist"
              << std::setw(12) << "Sort (ms)"
              << std::setw(14) << "RMI Error"
              << std::setw(12) << "RMI (ns)"
              << std::setw(8)  << "Status" << "\n";
    std::cout << "-----------------------------------------------------------------------------------------\n";

    bool all_passed = true;
    for (const auto& r : results) {
        const bool ok = r.sort_passed && r.rmi_passed;
        if (!ok) all_passed = false;

        std::cout << std::left 
                  << std::setw(30) << r.filename
                  << std::setw(10) << r.type_name
                  << std::setw(11) << r.dist_name
                  << std::setw(12) << std::fixed << std::setprecision(2) << r.sort_latency_ms
                  << std::setw(14) << r.error_bound
                  << std::setw(12) << std::fixed << std::setprecision(1) << r.rmi_latency_ns
                  << std::setw(8)  << (ok ? "PASSED" : "FAILED") << "\n";
    }
    std::cout << "=========================================================================================\n";
    std::cout << "OVERALL STATUS: " << (all_passed ? "[ALL 12 TESTS PASSED]" : "[TEST SUITE FAILED]") << "\n";
    std::cout << "=========================================================================================\n";
}

} // namespace mlaa::bench