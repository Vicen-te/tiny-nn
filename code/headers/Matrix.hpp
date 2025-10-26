

// Copyright (c) 2025 Vicente Brisa Saez
// GitHub: Vicen-te
// License: MIT


#pragma once


// ===================== STANDARD HEADERS =====================
#include <vector>
#include <cassert>
#include <stdexcept> 


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
 * Provides a minimal, efficient representation of a dense 2D matrix.
 * Stored in **row-major order**. Supports element access, resizing,
 * and basic operations like fill or copy from vector.
 */
template<Float T = float>
class Matrix
{

private:


    // ===================== MEMBERS =====================
    size_t n_rows = 0;      ///< Number of rows
    size_t n_cols = 0;      ///< Number of columns
    std::vector<T> data;    ///< Row-major data buffer



public:


    // ===================== CONSTRUCTORS =====================
    /// @brief Default constructor (creates an empty matrix)
    Matrix() = default;

    /**
     * @brief Construct a matrix with given dimensions.
     * @param rows Number of rows
     * @param cols Number of columns
     */
    Matrix(size_t rows, size_t cols) { resize(rows, cols); }


    // ===================== ACCESSORS =====================

    /// Get the number of rows
    [[nodiscard]] inline size_t rows() const { return n_rows; }

    /// Get the number of columns
    [[nodiscard]] inline size_t cols() const { return n_cols; }

    /**
     * @brief Get a raw pointer to the underlying data (mutable).
     * @return Pointer to the first element, or nullptr if empty.
     */
    [[nodiscard]] inline T* ptr() { return data.empty() ? nullptr : data.data(); }

    /**
     * @brief Get a raw const pointer to the underlying data (read-only).
     * @return Const pointer to the first element, or nullptr if empty.
     */
    [[nodiscard]] inline const T* ptr() const { return data.empty() ? nullptr : data.data(); }


    // ===================== MUTATORS =====================

    /**
     * @brief Resize the matrix and reset all values to zero.
     * @param rows New number of rows
     * @param cols New number of columns
     */
    inline void resize(size_t rows, size_t cols)
    {
        n_rows = rows;
        n_cols = cols;
        data.resize(rows * cols, static_cast<T>(0));
    }

    /**
     * @brief Copy data from a std::vector into the matrix.
     *        The vector size must match rows() * cols().
     * @param new_data Vector containing the data to copy
     */
    inline void copy_from_vector(const std::vector<T>& new_data)
    {
        if (new_data.size() != n_rows * n_cols)
            throw std::runtime_error("Vector size does not match matrix dimensions");

        std::memcpy(ptr(), new_data.data(), new_data.size() * sizeof(T));
    }

    /**
     * @brief Fill the matrix with a specific value and resize it.
     * @param rows New number of rows
     * @param cols New number of columns
     * @param value The value to assign to all elements
     */
    inline void assign(size_t rows, size_t cols, T value)
    {
        n_rows = rows;
        n_cols = cols;
        data.assign(rows * cols, value);
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
