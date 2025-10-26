

// Layer.cpp
// Copyright (c) 2025 Vicente Brisa Saez
// GitHub: Vicen-te
// License: MIT


// ===================== MAIN HEADER =====================
#include "../headers/CpuLayer.hpp"

using namespace activation;


// ===================== STANDARD HEADERS =====================
#include <cmath>
#include <random>
#include <stdexcept>
#include <memory>
#include <future>
#include <iostream>
#include <algorithm>


namespace cpu
{
    float Layer::forward_neuron(size_t i, std::span<const float> X)
    {
        float z = b[i]; 
        const float* w_row = W.ptr() + i * in;

        // Z = X * W^T + b
        for (size_t j = 0; j < in; ++j)
            z += X[j] * w_row[j];

        return z;  //< Pre-activation value
    }

    inline void Layer::grad_neuron
    (
        size_t i,
        std::span<const float> delta,
        std::span<const float> a_prev
    )
    {
        // db = delta
        const float d = delta[i];
        db[i] += d;

        // dW = delta * a_prev^T
        for (size_t j = 0; j < in; ++j)
            dW(i, j) += d * a_prev[j];
    }

    inline float Layer::delta_prev_neuron
    (
        size_t i,
        std::span<const float> delta,
        std::span<const float> a_prev,
        ActivationType act_prev
    ) const
    {
        // delta_prev = delta * W * d_activation
        float sum = 0.0f;
        for (size_t j = 0; j < out; ++j)
            sum += delta[j] * W(j, i);

        return sum * activate_derivative(a_prev[i], act_prev);
    }

    void Layer::average_neuron(size_t i, size_t batch_size)
    {
        db[i] /= batch_size;        //< Average bias gradient
        for (size_t j = 0; j < in; ++j)
            dW(i, j) /= batch_size; //< Average weight gradients
    }

    void Layer::sgd_update(size_t i, float lr)
    {
        // Update weights
        for (size_t j = 0; j < in; ++j)
            W(i, j) -= lr * dW(i, j);

        // Update bias
        b[i] -= lr * db[i];
    }

    void Layer::init()
    {
        dW.resize(out, in);
        db.resize(out);
    }

    void Layer::reset_gradients()
    {
        db.assign(out, 0.0f);       //< Reset bias gradients
        dW.assign(out, in, 0.0f);   //< Reset weight gradients
    }

    void Layer::from_json(const json& layer)
    {
        LayerBase::from_json(layer); //< Load weights and biases
        init();
    }

    Layer::Layer
    (
        size_t input_dim,
        size_t output_dim,
        ActivationType activation_type,
        std::string_view layer_name
    )
        : LayerBase(input_dim, output_dim, activation_type, layer_name)
    {
        init();
    }

    const std::vector<float> Layer::forward(std::span<const float> X, bool parallel)
    {
        std::vector<float> Z(out, 0.0f);

        // Compute pre-activation for all neurons (optionally parallelized)
        run_for(out, [&](size_t i) { Z[i] = forward_neuron(i, X); }, parallel);

        // Return activated values
        if (activation == ActivationType::SOFTMAX) 
            return softmax(Z);
        else
        {
            std::vector<float> A(out, 0.0f);
            run_for(out, [&](size_t i) { A[i] = activate(Z[i], activation); }, parallel);
            return A; 
        }
    }

    const std::vector<float> Layer::backward
    (
        std::span<const float> delta_next,
        std::span<const float> a_prev, 
        bool delta_prev_needed, 
        bool parallel
    )
    {
        // Compute dW and db for all neurons in this layer
        run_for(out, [&](size_t i) { grad_neuron(i, delta_next, a_prev); }, parallel);

        if (delta_prev_needed)
        {
            // Compute delta for previous layer neurons
            std::vector<float> delta_prev(in, 0.0f);
            run_for(in, [&](size_t i)
                {
                    delta_prev[i] = delta_prev_neuron(i, delta_next, a_prev, activation);

                }, parallel);
            return delta_prev;
        }

        // No delta_prev required
        return std::vector<float>();
    }

    void Layer::average_gradients(size_t batch_size, bool parallel)
    {
        run_for(out, [&](size_t i) { average_neuron(i, batch_size); }, parallel);
    }

    void Layer::update(float lr, bool parallel)
    {
        run_for(out, [&](size_t i) { sgd_update(i, lr); }, parallel);
        reset_gradients();
    }

}