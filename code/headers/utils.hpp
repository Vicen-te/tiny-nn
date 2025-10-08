
// Copyright (c) 2025 Vicente Brisa Saez
// GitHub: Vicen-te
// License: MIT

#pragma once

#include <chrono>
#include <fstream>
#include <vector>
#include <string>
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

/**
 * @brief Write tabular numeric data to a CSV file.
 *
 * @param path    Output file path.
 * @param header  Column header names.
 * @param rows    2D array of numeric data (each inner vector is one row).
 *
 * The function overwrites any existing file at the given path.
 */
inline void write_csv(
    const std::string& path,
    const std::vector<std::string>& header,
    const std::vector<std::vector<double>>& rows)
{
    std::ofstream ofs(path);
    if (!ofs.is_open())
        throw std::runtime_error("Failed to open file for writing: " + path);

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
