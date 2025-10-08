
// Copyright (c) 2025 Vicente Brisa Saez
// GitHub: Vicen-te
// License: MIT

#pragma once

#include <vector>
#include <cstddef>
#include <type_traits>
#include <cassert>

/**
 * @brief Concept to restrict the Matrix type to floating-point values.
 */
template<typename T>
concept Float = std::is_floating_point_v<T>;

/**
 * @brief A simple, row-major dense matrix container.
 *
 * @tparam T Floating-point type (e.g., float, double)
 *
 * This structure provides a minimal and efficient representation
 * of a dense 2D matrix, stored in **row-major order**. It supports
 * basic element access and resizing.
 */
template<Float T = float>
struct Matrix
{
    size_t rows = 0;        ///< Number of rows in the matrix
    size_t cols = 0;        ///< Number of columns in the matrix
    std::vector<T> data;    ///< Underlying data buffer in row-major order

    /// @brief Default constructor (creates an empty matrix)
    Matrix() = default;

    /**
     * @brief Construct a matrix with given dimensions.
     * @param r Number of rows
     * @param c Number of columns
     */
    Matrix(size_t r, size_t c) { resize(r, c); }

    /**
     * @brief Resize the matrix and reset all values to zero.
     * @param r New number of rows
     * @param c New number of columns
     */
    void resize(size_t r, size_t c)
    {
        rows = r;
        cols = c;
        data.assign(r * c, static_cast<T>(0));
    }

    /**
     * @brief Get a raw pointer to the underlying data.
     * @return Non-const pointer to the first element, or nullptr if empty.
     */
    inline T* ptr() { return data.empty() ? nullptr : data.data(); }

    /**
     * @brief Get a raw const pointer to the underlying data.
     * @return Const pointer to the first element, or nullptr if empty.
     */
    inline const T* ptr() const { return data.empty() ? nullptr : data.data(); }

    /**
     * @brief Element access (mutable).
     * @param i Row index (0-based)
     * @param j Column index (0-based)
     * @return Reference to element at position (i, j)
     */
    inline T& operator()(size_t i, size_t j)
    {
        assert(i < rows && j < cols);
        return data[i * cols + j];
    }

    /**
     * @brief Element access (read-only).
     * @param i Row index (0-based)
     * @param j Column index (0-based)
     * @return Const reference to element at position (i, j)
     */
    inline const T& operator()(size_t i, size_t j) const
    {
        assert(i < rows && j < cols);
        return data[i * cols + j];
    }
};
