#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <cstdint>

#include <mlaa/CpuTopology.hpp>
#include <mlaa/sample_sort.hpp>
#include <mlaa/FastRMI.hpp>

int main() {
    // 1. Hardware topology inspection
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
    std::cout << "========================================\n";

    // 2. Load dataset (1,000,000 int64_t elements = 8 MB)
    constexpr size_t DATASET_SIZE = 1'000'000;
    std::string bin_path = R"(C:\C++\mlaa\benchmark\gaussian_1m.bin)";

    if (!std::filesystem::exists(bin_path)) {
        bin_path = "../benchmark/gaussian_1m.bin";
    }

    std::ifstream file(bin_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Failed to open dataset: " << bin_path << "\n";
        return 1;
    }

    std::vector<int64_t> dataset(DATASET_SIZE);
    file.read(reinterpret_cast<char*>(dataset.data()), DATASET_SIZE * sizeof(int64_t));
    const std::streamsize bytes_read = file.gcount();
    file.close();

    if (bytes_read != static_cast<std::streamsize>(DATASET_SIZE * sizeof(int64_t))) {
        std::cerr << "[ERROR] Partial read: expected " << DATASET_SIZE * sizeof(int64_t)
                  << " bytes, got " << bytes_read << " bytes.\n";
        return 1;
    }

    std::cout << "\nDataset Loaded  : " << DATASET_SIZE << " int64_t elements (~"
              << (DATASET_SIZE * sizeof(int64_t) / (1024 * 1024)) << " MB)\n";

    // 3. Single-pass sorting via mlaa::sample_sort
    mlaa::sample_sort(dataset.begin(), dataset.end());

    if (!std::is_sorted(dataset.begin(), dataset.end())) {
        std::cerr << "[FAIL] Output array is NOT sorted correctly!\n";
        return 1;
    }
    std::cout << "Sort Verification: PASSED (std::is_sorted == true)\n";

    // 4. Construct FastRMI on sorted data
    mlaa::FastRMI<int64_t> rmi(dataset);

    // 5. Query validation for index 550,000
    constexpr size_t TARGET_INDEX = 550'000;
    const int64_t query_key = dataset[TARGET_INDEX];

    // Measure raw prediction latency
    const auto t_start = std::chrono::high_resolution_clock::now();
    auto [pred_pos, error_bound] = rmi.predict(query_key);
    const auto t_end = std::chrono::high_resolution_clock::now();

    const double latency_ns = std::chrono::duration<double, std::nano>(t_end - t_start).count();

    // 6. Calculate bounded search range
    const size_t bound_min = (pred_pos > error_bound) ? (pred_pos - error_bound) : 0;
    const size_t bound_max = std::min(static_cast<size_t>(pred_pos + error_bound), DATASET_SIZE - 1);

    // Local refinement search within bounded window
    auto it = std::lower_bound(dataset.begin() + bound_min, dataset.begin() + bound_max + 1, query_key);
    const size_t resolved_index = std::distance(dataset.begin(), it);
    const bool is_range_valid = (it != (dataset.begin() + bound_max + 1)) && (*it == query_key);

    // 7. Output evaluation report
    std::cout << "\n========================================\n";
    std::cout << "         FastRMI PREDICTION REPORT      \n";
    std::cout << "========================================\n";
    std::cout << "Target Index       : " << TARGET_INDEX << "\n";
    std::cout << "Query Key Value    : " << query_key << "\n";
    std::cout << "Predicted Index    : " << pred_pos << "\n";
    std::cout << "Error Bound (+/-)  : " << error_bound << "\n";
    std::cout << "Search Window      : [" << bound_min << ", " << bound_max << "]\n";
    std::cout << "Resolved Index     : " << resolved_index << "\n";
    std::cout << "Prediction Latency : " << latency_ns << " ns\n";
    std::cout << "Range Verification : " << (is_range_valid ? "PASSED" : "FAILED") << "\n";
    std::cout << "========================================\n";

    return is_range_valid ? 0 : 1;
}


/*
========================================
           CPU HARDWARE TOPOLOGY        
========================================
Physical Cores : 6
Logical Cores  : 12
----------------------------------------
L1 Data Cache  : 32 KB
L2 Cache       : 512 KB
L3 Cache       : 4 MB
Cache Line     : 64 bytes
----------------------------------------
Highest SIMD   : AVX2
AVX Supported  : Yes
AVX2 Supported : Yes
FMA Supported  : Yes
========================================

Dataset Loaded  : 1000000 int64_t elements (~7 MB)
Sort Verification: PASSED (std::is_sorted == true)

========================================
         FastRMI PREDICTION REPORT      
========================================
Target Index       : 550000
Query Key Value    : 512435
Predicted Index    : 549967
Error Bound (+/-)  : 66
Search Window      : [549901, 550033]
Resolved Index     : 549998
Prediction Latency : 300 ns
Range Verification : PASSED
========================================
*/