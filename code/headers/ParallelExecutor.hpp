

// Copyright (c) 2025 Vicente Brisa Saez
// GitHub: Vicen-te
// License: MIT


#pragma once


// ===================== STANDARD HEADERS =====================
#include <vector>
#include <thread>
#include <future>
#include <numeric>
#include <execution>
#include <algorithm>
#include <functional>


/**
 * @brief Utility class for parallel execution (STL or async fallback).
 *
 * This class provides a unified interface to execute loops in parallel.
 * It first tries to use the C++17 parallel STL (`std::execution::par`).
 * If the platform or compiler doesn't support it, it falls back to
 * asynchronous multithreading using `std::async`.
 *
 * Example usage:
 * @code
 * ParallelExecutor::Run(N, [&](size_t i) {
 *     // Do work on index i
 * });
 * @endcode
 */
class ParallelExecutor
{

public:


    /**
     * @brief Executes a loop in parallel over a given range.
     *
     * @param total Number of iterations to perform.
     * @param task  A lambda function taking a `size_t` index.
     */
    static void run(size_t num_items, const std::function<void(size_t)>& task)
    {
        if (num_items == 0) return;

        // Attempt to use parallel STL (if supported by the compiler and runtime)
        try
        {
            std::vector<size_t> indices(num_items);
            std::iota(indices.begin(), indices.end(), 0ull);

            std::for_each(std::execution::par, indices.begin(), indices.end(),
            [&](size_t i)
            {
                task(i);
            });

            return; //< Success with parallel STL
        }
        catch (...)
        {
            // Fallback to std::async-based parallelization if execution policy is unsupported
        }

        // Determine number of threads
        unsigned int n_threads = std::thread::hardware_concurrency();
        if (n_threads == 0) n_threads = 1;  //< Fallback if hardware_concurrency not available
        n_threads = std::min<unsigned>(n_threads, static_cast<unsigned>(num_items));

        size_t chunk_size = (num_items + n_threads - 1) / n_threads;
        std::vector<std::future<void>> futures;
        futures.reserve(n_threads);

        // Launch worker threads for each chunk
        for (unsigned t = 0; t < n_threads; ++t)
        {
            size_t start = t * chunk_size;
            size_t end = std::min(start + chunk_size, num_items);

            futures.push_back(std::async(std::launch::async, [&, start, end]()
                {
                    for (size_t i = start; i < end; ++i)
                        task(i);
                }));
        }

        // Wait for all threads to finish
        for (auto& f : futures)
            f.get();
    }

};
