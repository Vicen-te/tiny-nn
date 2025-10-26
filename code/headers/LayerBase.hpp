

// Copyright (c) 2025 Vicente Brisa Saez
// Github: Vicen-te
// License: MIT


#pragma once


// ===================== STANDARD HEADERS =====================
#include <vector>
#include <string_view>


// ===================== PROJECT HEADERS =====================
#include "Matrix.hpp"
#include "ActivationType.hpp"
#include "json.hpp"

using json = nlohmann::json;
using ActivationType = activation::ActivationType;


/**
 * @brief Base class for fully connected neural network layers.
 *
 * Provides weight/bias storage, activation type, and
 * serialization functionality. Derived classes (CPU/GPU)
 * implement forward/backward computations.
 */
class LayerBase
{

private:


    // ===================== HELPERS =====================
    /**
     * @brief Initialize the layer's weights randomly.
     *
     * Uses He initialization for ReLU and Xavier/Glorot for other activations.
     * Weights are drawn uniformly from [-limit, +limit]. Biases are not initialized.
     */
    void initialize_weights();



protected:


    // ===================== MEMBERS =====================
    size_t in = 0;              ///< Number of input neurons
    size_t out = 0;             ///< Number of output neurons

    Matrix<float> W;            ///< Weight matrix of size [out, in]
    std::vector<float> b;       ///< Bias vector of length 'out'

    std::string name;           ///< Optional layer name

    /// Activation function type (e.g., RELU, SOFTMAX)
    ActivationType activation = ActivationType::NONE;  



public:


    // ===================== CONSTRUCTORS =====================
    LayerBase() = default;


    /**
     * @brief Construct a fully connected layer.
     * 
     * Allocates weight matrix and biases, initializing weights
     * with He (for ReLU) or Xavier/Glorot (for others).
     *
     * Biases are small positive for ReLU, zero otherwise.
     * 
     * @param input_dim Number of input neurons
     * @param output_dim Number of output neurons
     * @param activation_type Activation function type for this layer
     * @param layer_name Optional name for the layer
     *
     */
    LayerBase
    (
        size_t input_dim,
        size_t output_dim,
        ActivationType activation_type,
        std::string_view layer_name = ""
    );


    // ===================== JSON SERIALIZATION =====================

     /**
     * @brief Serialize layer parameters (weights, biases, activation, name) to JSON.
     * @return JSON object representing the layer
     */
    json to_json() const;

    /**
     * @brief Load layer parameters from a JSON object.
     * @param layer JSON object containing previously saved layer data
     *
     */
    virtual void from_json(const json& layer);

};
