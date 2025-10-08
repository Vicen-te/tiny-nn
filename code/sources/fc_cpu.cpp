
// Copyright (c) 2025 Vicente Brisa Saez
// GitHub: Vicen-te
// License: MIT

#include "../headers/fc_cpu.hpp"
#include <future>
#include <thread>
#include <numeric>
#include <algorithm>
#include <execution>
#include <cmath>

/**
 * @brief Perform a fully connected (dense) layer forward pass on CPU (sequential version).
 *
 * This function computes:
 *     y_i = ReLU(b_i + sum_j(W_ij * x_j))
 *
 * @param W Weight matrix (dimensions: out × in)
 * @param b Bias vector (length: out)
 * @param x Input vector (length: in)
 * @return Output vector (length: out)
 */
std::vector<float> fc_forward_seq(const Matrix<float>& W,
                                  const std::vector<float>& b,
                                  const std::vector<float>& x)
{
    size_t out_dim = W.rows;
    size_t in_dim  = W.cols;

    std::vector<float> y(out_dim, 0.0f);

    for (size_t i = 0; i < out_dim; ++i)
    {
        float sum = b[i];
        const float* w_row = W.ptr() + i * in_dim;

        for (size_t j = 0; j < in_dim; ++j)
            sum += w_row[j] * x[j];

        // ReLU activation
        y[i] = (sum > 0.0f) ? sum : 0.0f;
    }

    return y;
}

/**
 * @brief Perform a fully connected (dense) layer forward pass on CPU (parallel version).
 *
 * Uses `std::execution::par` if supported, otherwise falls back to
 * manual multi-threaded chunking with `std::async`.
 *
 * @param W Weight matrix (dimensions: out × in)
 * @param b Bias vector (length: out)
 * @param x Input vector (length: in)
 * @return Output vector (length: out)
 */
std::vector<float> fc_forward_par(const Matrix<float>& W,
                                  const std::vector<float>& b,
                                  const std::vector<float>& x)
{
    size_t out_dim = W.rows;
    std::vector<float> y(out_dim, 0.0f);

    // Attempt to use parallel STL (if supported by the compiler and runtime)
    try
    {
        std::vector<size_t> indices(out_dim);
        std::iota(indices.begin(), indices.end(), 0ull);

        std::transform(std::execution::par, indices.begin(), indices.end(), y.begin(),
            [&](size_t i)
            {
                float sum = b[i];
                const float* w_row = W.ptr() + i * W.cols;
                for (size_t j = 0; j < W.cols; ++j)
                    sum += w_row[j] * x[j];
                return (sum > 0.0f) ? sum : 0.0f; //< ReLU
            });

        return y;
    }
    catch (...)
    {
        // Fallback to std::async-based parallelization if execution policy is unsupported
    }

    // Determine number of threads to use
    unsigned int n_threads = std::thread::hardware_concurrency();
    if (n_threads == 0) n_threads = 1;  //< fallback if hardware_concurrency not available
    n_threads = std::min<unsigned>(n_threads, static_cast<unsigned>(out_dim));

    size_t chunk_size = (out_dim + n_threads - 1) / n_threads;
    std::vector<std::future<void>> futures;

    // Launch worker threads for each chunk
    for (unsigned t = 0; t < n_threads; ++t)
    {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, out_dim);

        futures.push_back(std::async(std::launch::async, [&, start, end]()
        {
            for (size_t i = start; i < end; ++i)
            {
                float sum = b[i];
                const float* w_row = W.ptr() + i * W.cols;

                for (size_t j = 0; j < W.cols; ++j)
                    sum += w_row[j] * x[j];

                y[i] = (sum > 0.0f) ? sum : 0.0f; //< ReLU
            }
        }));
    }

    // Wait for all threads to finish
    for (auto& f : futures)
        f.get();

    return y;
}
