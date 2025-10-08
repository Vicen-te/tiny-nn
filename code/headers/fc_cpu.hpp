
// Copyright (c) 2025 Vicente Brisa Saez
// GitHub: Vicen-te
// License: MIT

#pragma once

#include "matrix.hpp"
#include <vector>

/**
 * @brief Perform a sequential CPU forward pass of a fully-connected (dense) layer.
 *
 * Computes: y = ReLU(W * x + b)
 *
 * @param W Weight matrix of size [out × in]
 * @param b Bias vector of length [out]
 * @param x Input vector of length [in]
 * @return Output vector of length [out]
 */
std::vector<float> fc_forward_seq(
    const Matrix<float>& W,
    const std::vector<float>& b,
    const std::vector<float>& x
);

/**
 * @brief Perform a parallel CPU forward pass of a fully-connected (dense) layer.
 *
 * Uses either std::execution::par or std::async-based multithreading
 * to compute: y = ReLU(W * x + b)
 *
 * @param W Weight matrix of size [out × in]
 * @param b Bias vector of length [out]
 * @param x Input vector of length [in]
 * @return Output vector of length [out]
 */
std::vector<float> fc_forward_par(
    const Matrix<float>& W,
    const std::vector<float>& b,
    const std::vector<float>& x
);
