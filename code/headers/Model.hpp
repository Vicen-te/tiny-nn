

// Copyright (c) 2025 Vicente Brisa Saez
// GitHub: Vicen-te
// License: MIT


#pragma once


// ===================== STANDARD HEADERS =====================
#include <vector>
#include <filesystem>
#include <fstream>


// ===================== PROJECT HEADERS =====================
#include "LayerBase.hpp"


/**
 * @brief Fully connected neural network model.
 *
 * Supports dynamic layer addition and JSON serialization.
 *
 * @tparam LayerType Must derive from LayerBase.
 */
template <typename LayerType>
requires std::is_base_of_v<LayerBase, LayerType>
class Model
{

private:


    // ===================== MEMBERS =====================
    std::vector<std::shared_ptr<LayerType>> layers;  ///< Layers of the network
    size_t output_size = 0;     ///< Number of output neurons


    // ===================== METADATA =====================
    size_t input_size = 0;      ///< Number of input neurons


public:

    bool cpu_parallel = false;  ///< Flag for CPU parallel execution


    // ===================== ACCESSORS =====================
    /**
     * @brief Get the number of input neurons of the model.
     * @return Number of input neurons
     */
    [[nodiscard]] inline size_t get_input_size() const { return input_size; }

    /**
     * @brief Get the number of output neurons of the model.
     * @return Number of output neurons
     */

    [[nodiscard]] inline size_t get_output_size() const { return output_size; }


    // ===================== LAYER MANAGEMENT =====================
    /**
     * @brief Add a fully connected layer to the model.
     * 
     * Automatically updates model input_size for first layer and output_size
     * for the last layer.
     * 
     * @param input_dim Number of input neurons for this layer
     * @param output_dim Number of output neurons for this layer
     * @param activation_type Activation type (RELU, SOFTMAX, etc.)
     * @param layer_name Optional descriptive layer name
     *
     */
    void add_layer
    (
        size_t input_dim,
        size_t output_dim,
        ActivationType activation_type,
        const std::string& layer_name = ""
    )
    {
        std::shared_ptr<LayerType> layer = std::make_shared<LayerType>(input_dim, output_dim, activation_type, layer_name);
        layers.push_back(layer);

        // Set input size for the first layer
        if (layers.size() == 1)
            input_size = input_dim;

        // Update output size every time a layer is added
        output_size = output_dim;
    }


    // ===================== TRAINING & INTERFACE =====================
    /**
     * @brief Performs forward pass, backward pass, gradient averaging, weight updates,
     * and computes cross-entropy loss.
     * 
     * @param X Input training vectors
     * @param Y Target output vectors
     * @param epochs Number of epochs
     * @param lr Learning rate
     * @param num_batches Number of mini-batches
     *
     */
    void train
    (
        const std::vector<std::vector<float>>& X,
        const std::vector<std::vector<float>>& Y,
        int epochs, float lr, int num_batches
    );

    /**
     * @brief Perform inference using the trained network.
     * @param X Input vector
     * @return Output vector after forward propagation
     *
     * CPU and GPU versions differ in implementation:
     * - CPU: sequential forward pass through all layers
     * - GPU: forward pass using tensors and cuBLAS
     */
    [[nodiscard]] std::vector<float> inference(const std::vector<float>& X);


    // ===================== SERIALIZATION =====================
    /**
     * @brief Save the model architecture to a JSON file.
     *
     * This includes both metadata and trainable parameters:
     *
     * Metadata (architecture info, does NOT change during training):
     * - Layer name
     * - Activation type
     * - Input and output size
     *
     * Trainable parameters (change during training):
     * - Weights
     * - Biases
     *
     * @param path Path to the output JSON file.
     * @param pretty Pretty-print JSON if true
     *
     */
    void to_json(const std::filesystem::path& path, bool pretty = true) const
    {
        json json_model;
        json_model["input_size"] = input_size;
        json_model["output_size"] = output_size;
        json_model["layers"] = json::array();

        // Serialize each layer
        for (const std::shared_ptr<LayerType>& Layer : layers)
            json_model["layers"].push_back(Layer->to_json());

        std::ofstream ofs(path);
        if (!ofs.is_open())
            throw std::runtime_error("Cannot open file for writing: " + path.string());

        // Pretty-print JSON with 4-space indentation
        ofs << (pretty ? json_model.dump(4) : json_model.dump()); 
    }

    /**
     * @brief Load model architecture from a JSON file.
     * 
     * Clears existing data and restores both metadata and parameters.
     *
     * Metadata (architecture info):
     * - Layer name
     * - Activation type
     * - Input and output size
     *
     * Trainable parameters:
     * - Weights
     * - Biases
     * 
     * @param path Path to the input JSON file.
     *
     */
    void from_json(const std::filesystem::path& path)
    {
        std::ifstream ifs(path);
        if (!ifs.is_open())
            throw std::runtime_error("Cannot open file for reading: " + path.string());

        json json_model;
        ifs >> json_model;

        input_size = json_model["input_size"];
        output_size = json_model["output_size"];

        layers.clear(); //< Remove existing layers

        // Deserialize each layer
        for (const json& json_layer : json_model["layers"])
        {
            std::shared_ptr<LayerType> layer = std::make_shared<LayerType>();
            layer->from_json(json_layer);
            layers.push_back(layer);
        }
    }

};
