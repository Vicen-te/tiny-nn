

// Copyright (c) 2025 Vicente Brisa Saez
// Github: Vicen-te
// License: MIT


// ===================== MAIN HEADER =====================
#include "../headers/benchmark.hpp"


// ===================== STANDARD HEADERS =====================
#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <cmath>
#include <string_view>
#include <span>


// ===================== PROJECT HEADERS =====================
#include "../headers/Timer.hpp"
#include "../headers/Model.hpp"
#include "../headers/GpuLayer.hpp"
#include "../headers/CpuLayer.hpp"


namespace
{

    /**
     * @brief Generate a random input vector for testing the model.
     *
     * @param input_size Number of elements in the vector.
     * @param seed Random seed for reproducibility (default: 12345).
     * @return Vector of floats in [0, 1].
     */
    std::vector<float> generate_random_input
    (
        size_t input_size,
        unsigned int seed = 12345
    )
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> input(input_size);
        for (auto& val : input) val = dist(rng);
        return input;
    }

    /**
     * @brief Compare two numeric sequences element-wise within 
     * a given tolerance and print detailed mismatches.
     * 
     * This function checks each corresponding element of the two sequences. 
     * If the absolute difference exceeds the specified tolerance, 
     * it prints the index, CPU value, GPU value, absolute difference,
     * and relative difference. It helps identify numerical discrepancies 
     * between CPU and GPU computations.
     *
     * @param a First numeric sequence (e.g., CPU output), passed as std::span<const float>.
     * @param b Second numeric sequence (e.g., GPU output), passed as std::span<const float>.
     * @param tol Tolerance threshold for differences (default: 1e-4).
     * @return True if all differences are within tolerance, false otherwise.
     */
    bool vectors_equal_verbose
    (
        std::span<const float> a,
        std::span<const float> b,
        float tol = 1e-4f
    )
    {
        if (a.size() != b.size())
        {
            std::cout << "Vectors have different sizes: " << a.size() << " vs " << b.size() << "\n";
            return false;
        }

        bool equal = true;
        for (size_t i = 0; i < a.size(); ++i)
        {
            float diff_abs = std::fabs(a[i] - b[i]);
            // avoid division by 0
            float diff_rel = diff_abs / (std::fabs(a[i]) + 1e-8f);

            if (diff_abs > tol)
            {
                std::cout << "Index " << i
                    << ": CPU=" << a[i]
                    << " | GPU=" << b[i]
                    << " | abs_diff=" << diff_abs
                    << " | rel_diff=" << diff_rel
                    << " > tol=" << tol
                    << "\n";
                equal = false;
            }
        }

        if (equal)
            std::cout << "All differences within tolerance " << tol << "\n";
        else
            std::cout << "Some differences exceed tolerance " << tol << "\n";

        return equal;
    }


    /**
     * @brief Write tabular numeric data to a CSV file.
     *
     * @param path    Output file path.
     * @param header  Column header names.
     * @param rows    2D array of numeric data (each inner vector is one row).
     *
     * The function overwrites any existing file at the given path.
     */
    void write_csv
    (
        const std::filesystem::path& path,
        std::span<const std::string_view> header,
        const std::vector<std::vector<double>>& rows
    )
    {
        std::ofstream ofs(path);
        if (!ofs.is_open())
            throw std::runtime_error("Failed to open file for writing: " + path.string());

        // Write CSV header
        for (size_t i = 0; i < header.size(); ++i)
        {
            ofs << header[i];
            if (i + 1 < header.size()) ofs << ',';
        }
        ofs << '\n';

        // Write data rows
        for (const auto& row : rows)
        {
            for (size_t j = 0; j < row.size(); ++j)
            {
                ofs << std::fixed << std::setprecision(6) << row[j];
                if (j + 1 < row.size()) ofs << ',';
            }
            ofs << '\n';
        }
    }

    /**
     * @brief Measure and write the CPU vs CUDA inference benchmark.
     *
     * Runs multiple inferences on both CPU and GPU, measures execution time,
     * and stores the results in a CSV file.
     *
     * @param cpu_model Reference to CPU model.
     * @param gpu_model Reference to GPU model.
     * @param input Input vector to feed the models.
     * @param bench_path Path to CSV file for benchmark results.
     */
    void compare_inference_performance
    (
        Model<cpu::Layer>& cpu_model,
        Model<gpu::Layer>& gpu_model,
        const std::vector<float>& input,
        const std::filesystem::path& bench_path
    )
    {
        const int warmup = 5;
        const int reps = 200;
        Timer timer;

        // Warm-up CPU and GPU to stabilize performance
        for (int i = 0; i < warmup; ++i)
        {
            (void)cpu_model.inference(input);
            (void)gpu_model.inference(input);
        }

        // Measure CPU time (parallel)
        timer.reset();
        for (int i = 0; i < reps; ++i)
            (void)cpu_model.inference(input);
        double cpu_ms = timer.elapsed_milliseconds() / reps;

        // Measure CUDA time
        timer.reset();
        for (int i = 0; i < reps; ++i)
            (void)gpu_model.inference(input);
        double cuda_ms = timer.elapsed_milliseconds() / reps;

        std::cout << "Average times (ms): CPU-par=" << cpu_ms
            << " | CUDA-reuse=" << cuda_ms << "\n";

        // Write benchmark results to CSV
        std::array<std::string_view, 4> header = { "input", "output", "t_cpu_ms", "t_cuda_ms" };
        std::vector<std::vector<double>> rows =
        { { (double)cpu_model.get_input_size(), (double)cpu_model.get_output_size(), cpu_ms, cuda_ms}};

        write_csv(bench_path, header, rows);
        std::cout << "Results written to " << bench_path << "\n";
    }

}

void benchmark::run
(
    const std::filesystem::path& model_path, 
    const std::filesystem::path& bench_path
)
{
    // Load Model
    std::cout << "Loading model from: " << model_path << "\n";

    Model<cpu::Layer> cpu_model;
    Model<gpu::Layer> gpu_model;

    cpu_model.from_json(model_path);
    gpu_model.from_json(model_path);

    // Generate random input
    const std::vector<float> input = generate_random_input(cpu_model.get_input_size());

    // CPU vs CUDA correctness check
    std::vector<float> cpu_out = cpu_model.inference(input);
    std::vector<float> cuda_out = gpu_model.inference(input);

    // Print CPU output
    std::cout << "CPU output:\n";
    for (size_t i = 0; i < cpu_out.size(); ++i)
        std::cout << cpu_out[i] << " ";
    std::cout << std::endl << std::endl;

    // Print CUDA output
    std::cout << "CUDA output:\n";
    for (size_t i = 0; i < cuda_out.size(); ++i)
        std::cout << cuda_out[i] << " ";
    std::cout << std::endl << std::endl;

    // Check element-wise equality with tolerance
    bool equals = vectors_equal_verbose(cpu_out, cuda_out, 8.3e-3f);
    std::cout << "Inference (Forward) CPU vs CUDA match? " << (equals ? "YES" : "NO") << "\n";

    // Performance benchmark
    compare_inference_performance(cpu_model, gpu_model, input, bench_path);
}