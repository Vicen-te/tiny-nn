
// Copyright (c) 2025 Vicente Brisa Saez
// GitHub: Vicen-te
// License: MIT

#pragma once

#include "matrix.hpp"
#include <vector>
#include <string>

/**
 * @brief Represents a single fully-connected layer in a neural network.
 *
 * Each layer contains:
 *  - Input and output dimensions
 *  - Weight matrix (out × in)
 *  - Bias vector (out)
 *  - Optional activation type and layer name
 */
struct Layer
{
    size_t in = 0;                 ///< Number of input neurons
    size_t out = 0;                ///< Number of output neurons
    Matrix<float> W;               ///< Weight matrix of shape [out × in]
    std::vector<float> b;          ///< Bias vector of length [out]
    std::string activation;        ///< Activation function name (e.g. "relu", "sigmoid")
    std::string name;              ///< Optional layer name or identifier
};

/**
 * @brief Represents a neural network model composed of multiple layers.
 *
 * Contains:
 *  - Sequential list of layers
 *  - Cached input/output sizes (inferred from first and last layers)
 */
struct Model
{
    std::vector<Layer> layers;     ///< Ordered list of fully-connected layers
    size_t input_size = 0;         ///< Size of model input vector
    size_t output_size = 0;        ///< Size of model output vector
};
