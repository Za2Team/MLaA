#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <string>
#include <cmath>
#include <cstdint>
#include <benchmark/benchmark.h>

constexpr size_t NUM_ELEMENTS = 10'000'000;
constexpr size_t UNIQUE_POOL  = 2'000'000; // Forces duplicates in 10M elements

// Direct binary dump to disk via single buffered write
template <typename T>
void dump_to_binary(const std::string& filename, const std::vector<T>& data) {
    std::ofstream out(filename, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::cerr << "Failed to open: " << filename << '\n';
        return;
    }
    out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
}

// Generate Uniform data with duplicates
template <typename T>
void generate_uniform(const std::string& filename) {
    std::vector<T> data(NUM_ELEMENTS);
    std::mt19937_64 rng(1337);

    if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<int64_t> dist(0, static_cast<int64_t>(UNIQUE_POOL));
        for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
            data[i] = static_cast<T>(dist(rng));
        }
    } else {
        std::uniform_real_distribution<double> dist(0.0, 100'000.0);
        for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
            // Truncate to force duplicate floating point numbers
            data[i] = static_cast<T>(std::round(dist(rng) * 100.0) / 100.0);
        }
    }

    dump_to_binary(filename, data);
}

// Generate Gaussian data with high density and duplicates around the mean
template <typename T>
void generate_gaussian(const std::string& filename) {
    std::vector<T> data(NUM_ELEMENTS);
    std::mt19937_64 rng(7331);

    const double mean = 1'000'000.0;
    const double stddev = 150'000.0;
    std::normal_distribution<double> dist(mean, stddev);

    for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
        double val = dist(rng);
        if constexpr (std::is_integral_v<T>) {
            data[i] = static_cast<T>(std::llround(val));
        } else {
            // Round to single decimal place to guarantee duplicates
            data[i] = static_cast<T>(std::round(val * 10.0) / 10.0);
        }
    }

    dump_to_binary(filename, data);
}

int main(int argc, char** argv) {
    std::cout << "Generating dataset files (10M elements each)...\n";

    // int32_t
    generate_uniform<int32_t>("data_uniform_int32.bin");
    generate_gaussian<int32_t>("data_gaussian_int32.bin");

    // uint32_t
    generate_uniform<uint32_t>("data_uniform_uint32.bin");
    generate_gaussian<uint32_t>("data_gaussian_uint32.bin");

    // int64_t
    generate_uniform<int64_t>("data_uniform_int64.bin");
    generate_gaussian<int64_t>("data_gaussian_int64.bin");

    // uint64_t
    generate_uniform<uint64_t>("data_uniform_uint64.bin");
    generate_gaussian<uint64_t>("data_gaussian_uint64.bin");

    // float
    generate_uniform<float>("data_uniform_float.bin");
    generate_gaussian<float>("data_gaussian_float.bin");

    // double
    generate_uniform<double>("data_uniform_double.bin");
    generate_gaussian<double>("data_gaussian_double.bin");

    std::cout << "All 12 binary files generated successfully.\n";

    // Initialize and run Google Benchmark harness if arguments were passed
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}