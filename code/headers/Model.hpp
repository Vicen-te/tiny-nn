

// Copyright (c) 2025 Vicente Brisa Saez
// GitHub: Vicen-te
// License: MIT


#pragma once


// ===================== STANDARD LIBRARIES =====================
#include <vector>
#include <string>


// ===================== CUSTOM HEADERS =====================
#include "layer.hpp"


/**
 * @brief Represents a fully connected neural network model.
 *
 * Supports adding layers dynamically and saving/loading the model
 * in JSON format for serialization.
 */
class Model
{
public:

    std::vector<Layer> layers;  ///< All layers in the model
    size_t input_size = 0;      ///< Number of input neurons
    size_t output_size = 0;     ///< Number of output neurons


    // ===================== LAYER MANAGEMENT =====================
    /**
     * @brief Add a fully connected layer to the model.
     *
     * @param in Number of input neurons
     * @param out Number of output neurons
     * @param activation Activation function name ("relu", "softmax", etc.)
     * @param name Optional name for the layer
     *
     * Updates input_size for the first layer and output_size for the last layer.
     */
    inline void add_layer(size_t in, size_t out, const std::string& activation, const std::string& name = "")
    {
        layers.emplace_back(in, out, activation, name);

        // Set input size for the first layer
        if (layers.size() == 1)
            input_size = in;

        // Update output size every time a layer is added
        output_size = out;
    }


    // ===================== SERIALIZATION =====================
    /**
     * @brief Save the model architecture and weights to a JSON file.
     *
     * @param path Path to the output JSON file.
     *
     * Serializes input/output sizes and all layers.
     * Throws std::runtime_error if file cannot be opened.
     */
    void save_json(const std::string& path) const;

    /**
     * @brief Load model architecture and weights from a JSON file.
     *
     * @param path Path to the input JSON file.
     *
     * Clears any existing layers before loading. Throws std::runtime_error
     * if the file cannot be opened or parsed.
     */
    void load_json(const std::string& path);
};
