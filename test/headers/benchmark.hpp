

// Copyright (c) 2025 Vicente Brisa Saez
// Github: Vicen-te
// License: MIT


#pragma once


// ===================== STANDARD HEADERS =====================
#include <filesystem>
#include <span>


namespace benchmark
{

    /**
     * @brief Compare a model's CPU and GPU implementations.
     *
     * Loads a model from a JSON file, generates a random input vector,
     * runs multiple inferences on both CPU and GPU, checks that their
     * outputs are numerically consistent within a tolerance, and records
     * the average inference times in a CSV file.
     *
     * This function is useful for validating correctness between
     * CPU and GPU implementations and for benchmarking their performance.
     *
     * @param model_path Path to the JSON file containing the model definition.
     * @param bench_path Path to the CSV file where benchmark results will be saved.
     */
    void compare_models
	(
		const std::filesystem::path& model_path, 
		const std::filesystem::path& bench_path
	);

    /**
     * @brief Verify the correctness of a model inference using one-hot encoded labels.
     *
     * This function compares the model’s predicted output vector (probabilities or scores)
     * against the expected one-hot label vector, determines both the predicted and
     * true class indices, prints the result, and returns whether the prediction is correct.
     *
     * @param output           Model output vector (e.g., 10 values for MNIST digits 0–9).
     * @param sample           Input sample vector (currently unused, but can be useful for debugging or visualization).
     * @param expected_labels  One-hot encoded ground truth label vector.
     * @return True if the predicted class matches the expected class, false otherwise.
     */
    [[nodiscard]] bool verify_inference
    (
        std::span<const float> output,
        std::span<const float> expected_labels
    );
}