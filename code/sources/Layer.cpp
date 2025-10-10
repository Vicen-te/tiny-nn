

// Layer.cpp
// Copyright (c) 2025 Vicente Brisa Saez
// GitHub: Vicen-te
// License: MIT


// ===================== MAIN HEADER =====================
#include "../headers/Layer.hpp"


// ===================== STANDARD LIBRARIES =====================
#include <cmath>
#include <random>
#include <stdexcept>


Layer::Layer(size_t in_, size_t out_, const std::string& act, const std::string& name_)
    : in(in_), out(out_), activation(act), name(name_)
{
    W.resize(out, in);

    // Random initialization using uniform distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    float limit = std::sqrt(1.0f / in);
    std::uniform_real_distribution<float> dis(-limit, limit);

    for (size_t i = 0; i < out; ++i)
        for (size_t j = 0; j < in; ++j)
            W(i, j) = dis(gen);

    // Initialize biases to zero
    b.assign(out, 0.0f);
}

// ===================== JSON SERIALIZATION =====================
json Layer::to_json() const
{
    json jl;
    jl["in"] = in;
    jl["out"] = out;
    jl["activation"] = activation;
    jl["name"] = name;
    jl["W"] = json::array();

    // Copy each row of the weight matrix
    for (size_t i = 0; i < out; ++i)
    {
        json row = json::array();
        for (size_t j = 0; j < in; ++j)
            row.push_back(W(i, j));

        jl["W"].push_back(row);
    }

    jl["b"] = b; // Store biases
    return jl;
}

/**
 * @brief Load layer parameters from a JSON object
 * @param layer JSON object containing layer parameters
 */
void Layer::from_json(const json& layer)
{
    in = layer["in"];
    out = layer["out"];
    activation = layer["activation"];
    name = layer["name"];

    W.resize(out, in);

    for (size_t i = 0; i < out; ++i)
        for (size_t j = 0; j < in; ++j)
            W(i, j) = layer["W"][i][j];

    b = layer["b"].get<std::vector<float>>(); // Load biases
}
