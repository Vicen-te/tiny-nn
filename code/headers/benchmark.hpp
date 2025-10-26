

// Copyright (c) 2025 Vicente Brisa Saez
// Github: Vicen-te
// License: MIT


#pragma once


// ===================== STANDARD HEADERS =====================
#include <filesystem>


/**
 * @brief Namespace for benchmarking utilities.
 *
 * Provides functions to compare CPU vs GPU model outputs,
 * measure inference times, and save results to CSV.
 */
namespace benchmark
{

    /**
     * @brief Run a CPU vs CUDA benchmark for a model.
     *
     * Loads the model from a JSON file, runs inference on a random input,
     * checks correctness between CPU and GPU outputs, and records average
     * inference times in a CSV file.
     *
     * @param model_path Path to the JSON file containing the model.
     * @param bench_path Path to the CSV file where results will be saved.
     */
	void run
	(
		const std::filesystem::path& model_path, 
		const std::filesystem::path& bench_path
	);

}