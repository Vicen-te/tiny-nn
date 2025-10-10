

// Copyright (c) 2025 Vicente Brisa Saez
// GitHub: Vicen-te
// License: MIT


#pragma once


// ===================== STANDARD LIBRARIES =====================
#include <chrono>
#include <iomanip>


/**
 * @brief Simple high-resolution timer utility for benchmarking.
 *
 * Measures elapsed time in milliseconds using std::chrono::high_resolution_clock.
 */
struct Timer
{
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point start_time;

    /// Constructor — automatically starts the timer
    inline Timer() : start_time(Clock::now()) {}

    /// Reset the timer to the current time
    inline void reset() { start_time = Clock::now(); }

    /// Get the elapsed time in milliseconds since the last reset
    inline double elapsed_ms() const
    {
        return std::chrono::duration<double, std::milli>(Clock::now() - start_time).count();
    }
};
