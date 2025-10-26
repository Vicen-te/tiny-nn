

// Copyright (c) 2025 Vicente Brisa Saez
// Github: Vicen-te
// License: MIT


// ===================== MAIN HEADER =====================
#include "../headers/LayerBase.hpp"


// ===================== STANDARD HEADERS =====================
#include <random>


void LayerBase::initialize_weights()
{
    std::random_device rd;  // Random seed
    std::mt19937 gen(rd()); // Mersenne Twister engine for reproducible randomness

    // Compute initialization limit depending on activation function
    float limit = (activation == ActivationType::RELU)
        ? std::sqrt(2.0f / in)  // He initialization (suitable for ReLU)
        : std::sqrt(1.0f / in); // Xavier for others (for sigmoid/tanh, etc.)

    std::uniform_real_distribution<float> dis(-limit, limit);

    // Fill weight matrix W with random values
    for (size_t i = 0; i < out; ++i)
        for (size_t j = 0; j < in; ++j)
            W(i, j) = dis(gen);
}

LayerBase::LayerBase
(   
    size_t input_dim,
    size_t output_dim, 
    ActivationType activation_type, 
    std::string_view layer_name
)
    : in(input_dim), out(output_dim), name(layer_name), activation(activation_type)
{
    W.resize(out, in); //< Allocate weight matrix

    // Small positive bias for ReLU, zero bias otherwise
    b.assign(out, (activation == ActivationType::RELU) ? 0.01f : 0.0f);

    // Initialize weights using appropriate strategy
    initialize_weights();
}

json LayerBase::to_json() const
{
    json json_layer;
    json_layer["in"] = in;
    json_layer["out"] = out;
    json_layer["name"] = name;
    json_layer["activation"] = activation;
    json_layer["W"] = json::array();

    // Serialize weight matrix row by row
    for (size_t i = 0; i < out; ++i)
    {
        json row = json::array();
        for (size_t j = 0; j < in; ++j)
            row.push_back(W(i, j));

        json_layer["W"].push_back(row);
    }

    json_layer["b"] = b; //< Store biases
    return json_layer;
}

void LayerBase::from_json(const json& json_layer)
{
    in = json_layer["in"];
    out = json_layer["out"];
    name = json_layer["name"];
    activation = json_layer["activation"];

    W.resize(out, in);

    // Validate weight dimensions
     
    // Number of rows
    if (!json_layer["W"].is_array() || json_layer["W"].size() != out) 
        throw std::runtime_error("Weight row size mismatch in layer: " + name);

    // Number of cols 
    if (!json_layer["W"][out - 1].is_array() || 
        json_layer["W"][out - 1].size() != in) 
        throw std::runtime_error
        (
            "Weight column size mismatch in layer: " + name + 
            " at row " + std::to_string(out - 1)
        );

    // Load weight values
    for (size_t i = 0; i < out; ++i)
        for (size_t j = 0; j < in; ++j)
            W(i, j) = json_layer["W"][i][j];

    // Validate and load biases
    if (json_layer["b"].size() != out)
        throw std::runtime_error("Bias size mismatch in layer: " + name);

    b = json_layer["b"].get<std::vector<float>>(); //< Load bias vector
}
