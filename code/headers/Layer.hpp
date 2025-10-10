


// Copyright (c) 2025 Vicente Brisa Saez
// Github: Vicen-te
// License: MIT


#pragma once


// ===================== STANDARD LIBRARIES =====================
#include <vector>
#include <string>


// ===================== CUSTOM HEADERS =====================
#include "matrix.hpp"
#include "json.hpp"

using json = nlohmann::json;


/**
 * @brief Fully connected neural network layer.
 *
 * Stores input/output sizes, weights, biases, activation function, and optional name.
 * Supports random initialization and JSON serialization/deserialization.
 */
struct Layer
{
    size_t in = 0;               //< Number of input neurons
    size_t out = 0;              //< Number of output neurons
    Matrix<float> W;             //< Weight matrix of size [out, in]
    std::vector<float> b;        //< Bias vector
    std::string activation;      //< Activation function name (e.g., "relu", "softmax")
    std::string name;            //< Optional layer name


    // ===================== CONSTRUCTORS =====================
    Layer() = default;

    /**
     * @brief Construct a layer with random weight initialization.
     *
     * Initializes weights with uniform distribution in [-sqrt(1/in), sqrt(1/in)]
     * and biases to zero.
     *
     * @param in_ Number of input neurons
     * @param out_ Number of output neurons
     * @param act Activation function name
     * @param name_ Optional layer name
     */
    Layer(size_t in_, size_t out_, const std::string& act, const std::string& name_ = "");


    // ===================== JSON SERIALIZATION =====================

    /**
     * @brief Serialize layer to JSON.
     * @return JSON object containing all layer parameters
     */
    json to_json() const;

    /**
     * @brief Load layer parameters from a JSON object
     * @param layer JSON object containing layer parameters
     */
    void from_json(const json& layer);
};
