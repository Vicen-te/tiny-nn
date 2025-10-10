

// Copyright (c) 2025 Vicente Brisa Saez
// GitHub: Vicen-te
// License: MIT


#pragma once


// ===================== STANDARD LIBRARIES =====================
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
class Matrix
{

private:

    size_t n_rows = 0;      ///< Number of rows in the matrix
    size_t n_cols = 0;      ///< Number of columns in the matrix
    std::vector<T> data;    ///< Underlying data buffer in row-major order

public:

    // ===================== CONSTRUCTORS =====================

    /// @brief Default constructor (creates an empty matrix)
    Matrix() = default;

    /**
     * @brief Construct a matrix with given dimensions.
     * @param r Number of rows
     * @param c Number of columns
     */
    Matrix(size_t r, size_t c) { resize(r, c); }


    // ===================== ACCESSORS =====================

    /// Number of rows per image
    inline size_t rows() const { return n_rows; }

    /// Number of columns per image
    inline size_t cols() const { return n_cols; }

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


    // ===================== MUTATORS =====================

    /**
     * @brief Resize the matrix and reset all values to zero.
     * @param r New number of rows
     * @param c New number of columns
     */
    void resize(size_t r, size_t c)
    {
        n_rows = r;
        n_cols = c;
        data.assign(r * c, static_cast<T>(0));
    }


    // ===================== ELEMENT ACCESS =====================

    /**
     * @brief Element access (mutable).
     * @param i Row index (0-based)
     * @param j Column index (0-based)
     * @return Reference to element at position (i, j)
     */
    inline T& operator()(size_t i, size_t j)
    {
        assert(i < n_rows && j < n_cols);
        return data[i * n_cols + j];
    }

    /**
     * @brief Element access (read-only).
     * @param i Row index (0-based)
     * @param j Column index (0-based)
     * @return Const reference to element at position (i, j)
     */
    inline const T& operator()(size_t i, size_t j) const
    {
        assert(i < n_rows && j < n_cols);
        return data[i * n_cols + j];
    }
};
