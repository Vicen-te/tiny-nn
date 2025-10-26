

// Copyright (c) 2025 Vicente Brisa Saez
// GitHub: Vicen-te
// License: MIT


#pragma once


// ===================== STANDARD HEADERS =====================
#include <chrono>


/**
 * @brief Simple high-resolution timer utility for benchmarking.
 *
 * Uses std::chrono::high_resolution_clock to measure elapsed time
 * with millisecond precision. Useful for profiling code sections.
 */
class Timer
{

private:


    // ===================== MEMBERS =====================
    using Clock = std::chrono::high_resolution_clock;   ///< Alias for high-resolution clock
    using TimePoint = Clock::time_point;                ///< Alias for a time point

    TimePoint start_time;                               ///< Start time of the timer  



public:


    // ===================== CONSTRUCTORS =====================
    /// Constructor — automatically starts the timer
    inline Timer() : start_time(Clock::now()) {}


    // ===================== MUTATORS =====================
    /// Reset the timer to the current time
    inline void reset() { start_time = Clock::now(); }


    // ===================== ACCESSORS =====================
    /**
     * @brief Get elapsed time in milliseconds since last reset.
     * @return Elapsed time in milliseconds (double)
     */
    [[nodiscard]] inline double elapsed_milliseconds() const noexcept
    {
        return std::chrono::duration<double, std::milli>(Clock::now() - start_time).count();
    }

    /**
    * @brief Get elapsed time in seconds since last reset.
    * @return Elapsed time in seconds (double)
    */
    [[nodiscard]] inline double elapsed_seconds() const noexcept
    {
        return std::chrono::duration<double>(Clock::now() - start_time).count();
    }

};
