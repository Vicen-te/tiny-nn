

// Copyright (c) 2025 Vicente Brisa Saez
// GitHub: Vicen-te
// License: MIT


#pragma once


// ===================== MAIN HEADER =====================
#include "../headers/fc_cpu.hpp"


// ===================== STANDARD LIBRARIES =====================
#include <vector>
#include <iostream>
#include <future>
#include <thread>
#include <numeric>
#include <algorithm>
#include <execution>
#include <random>
#include <cmath>              //< For exp, log, etc.


// ===================== CUSTOM HEADERS =====================
#include "../headers/model.hpp"
#include "../headers/matrix.hpp"  


namespace cpu
{
    // ================================================================
    //                     ACTIVATION FUNCTION
    // ================================================================
   
    std::vector<float> softmax(const std::vector<float>& z)
    {
        std::vector<float> res = z;  //< Copy to work in-place

        // Find the maximum value in the vector
        float max_val = *std::ranges::max_element(res); //< For numerical stability

        float sum = 0.0f; 

        // Compute exponentials of (score - max_val) to prevent numerical overflow
        // While accumulating their sum for normalization
        for (auto& score : res) sum += (score = std::exp(score - max_val)); //< Exponentiate and sum
        for (auto& score : res) score /= sum; //< Normalize

        return res;
    }


    // ================================================================
    //                     LOSS FUNCTION
    // ================================================================

    float cross_entropy(const std::vector<float>& y_true, const std::vector<float>& y_pred)
    {
        float loss = 0.0f;
        constexpr float eps = 1e-8f; //< Small epsilon to prevent log(0) and numerical instability

        for (size_t i = 0; i < y_true.size(); ++i)
            loss -= y_true[i] * std::log(std::clamp(y_pred[i], eps, 1.0f));

        return loss;
    }


    // ================================================================
    //                 FULLY CONNECTED LAYER OPERATIONS
    // ================================================================

    std::vector<float> forward_seq(const Matrix<float>& W,
        const std::vector<float>& b,
        const std::vector<float>& x)
    {
        size_t out_dim = W.rows();
        size_t in_dim = W.cols();

        std::vector<float> z(out_dim, 0.0f);

        for (size_t i = 0; i < out_dim; ++i)
        {
            float sum = b[i];
            const float* w_row = W.ptr() + i * in_dim;

            for (size_t j = 0; j < in_dim; ++j)
                sum += w_row[j] * x[j];

            z[i] = sum;
        }

        return z;
    }

    std::vector<float> forward_par(const Matrix<float>& W,
        const std::vector<float>& b,
        const std::vector<float>& x)
    {
        size_t out_dim = W.rows();
        std::vector<float> y(out_dim, 0.0f);

        // Attempt to use parallel STL (if supported by the compiler and runtime)
        try
        {
            std::vector<size_t> indices(out_dim);
            std::iota(indices.begin(), indices.end(), 0ull);

            std::transform(std::execution::par, indices.begin(), indices.end(), y.begin(),
                [&](size_t i)
                {
                    float sum = b[i];
                    const float* w_row = W.ptr() + i * W.cols();
                    for (size_t j = 0; j < W.cols(); ++j)
                        sum += w_row[j] * x[j];
                    return (sum > 0.0f) ? sum : 0.0f; //< ReLU
                });

            return y;
        }
        catch (...)
        {
            // Fallback to std::async-based parallelization if execution policy is unsupported
        }

        // Determine number of threads to use
        unsigned int n_threads = std::thread::hardware_concurrency();
        if (n_threads == 0) n_threads = 1;  //< Fallback if hardware_concurrency not available
        n_threads = std::min<unsigned>(n_threads, static_cast<unsigned>(out_dim));

        size_t chunk_size = (out_dim + n_threads - 1) / n_threads;
        std::vector<std::future<void>> futures;

        // Launch worker threads for each chunk
        for (unsigned t = 0; t < n_threads; ++t)
        {
            size_t start = t * chunk_size;
            size_t end = std::min(start + chunk_size, out_dim);

            futures.push_back(std::async(std::launch::async, [&, start, end]()
                {
                    for (size_t i = start; i < end; ++i)
                    {
                        float sum = b[i];
                        const float* w_row = W.ptr() + i * W.cols();

                        for (size_t j = 0; j < W.cols(); ++j)
                            sum += w_row[j] * x[j];

                        y[i] = (sum > 0.0f) ? sum : 0.0f; //< ReLU
                    }
                }
            ));
        }

        // Wait for all threads to finish
        for (auto& f : futures)
            f.get();

        return y;
    }


    // ================================================================
    //                FORWARD AND BACKWARD PROPAGATION
    // ================================================================

    std::vector<std::vector<float>> forward_cpu(const Model& model, const std::vector<float>& x)
    {
        std::vector<std::vector<float>> activations;
        std::vector<float> cur = x;
        activations.push_back(cur); //< Save input layer as first activation

        for (const Layer& layer : model.layers)
        {
            cur = forward_seq(layer.W, layer.b, cur); //< Linear transformation: z = W*x + b

            if (layer.activation == "relu")
            {
                for (auto& v : cur) v = relu(v);
            }
            else if (layer.activation == "softmax")
                cur = softmax(cur);

            activations.push_back(cur);
        }
        return activations; //< Return all activations, 0: input, 1: first hidden layer, ... 
    }

    Gradients backward_cpu
    (
        Model& model,
        const std::vector<std::vector<float>>& A,
        const std::vector<float>& y_true
    )
    {
        Gradients grads;
        grads.dW.resize(model.layers.size()); 
        grads.db.resize(model.layers.size());

        std::vector<float> delta;           
        const size_t num_layers = model.layers.size();

        //                  output layer
        delta.resize(model.layers[num_layers - 1].out);


        // Compute gradient of the loss for the last layer.
        // For softmax + cross-entropy, this simplifies to: delta = softmax_output - one_hot_labels
        for (size_t i = 0; i < delta.size(); ++i)
            delta[i] = A.back()[i] - y_true[i];


        // Backpropagate through all layers
        for (int layer = (int)num_layers - 1; layer >= 0; --layer)
        {
            const Matrix<float>& W = model.layers[layer].W;
            const std::vector<float>& b = model.layers[layer].b;
            const std::vector<float>& a_prev = A[layer];

            Matrix<float>& dW = grads.dW[layer];
            std::vector<float>& db = grads.db[layer];

            dW.resize(W.rows(), W.cols());
            db.assign(W.rows(), 0);

            // Compute weight and bias gradients
            // dW = delta * a_prev^T
            // db = delta
            for (size_t i = 0; i < W.rows(); ++i)
            {
                for (size_t j = 0; j < W.cols(); ++j)
                    dW(i, j) += delta[i] * a_prev[j];

                db[i] += delta[i];
            }

            // Compute delta for previous layer
            // delta_prev = W^T * delta .* d_activation
            if (layer > 0)
            {
                std::vector<float> delta_prev(W.cols(), 0);
                for (size_t j = 0; j < W.cols(); ++j)
                {
                    // Softmax by default
                    for (size_t i = 0; i < W.rows(); ++i)
                        delta_prev[j] += W(i, j) * delta[i];

                    // Apply derivative of activation function (if ReLU)
                    if (model.layers[layer - 1].activation == "relu")
                        delta_prev[j] *= d_relu(A[layer][j]);
                }

                delta = delta_prev; //< Set delta for next iteration
            }
        }
        return grads;
    }


    // ================================================================
    //                      WEIGHT AND BIAS UPDATE
    // ================================================================

    void sgd_update(Model& model, const Gradients& grads, float lr)
    {
        for (size_t l = 0; l < model.layers.size(); ++l)
        {
            Layer& layer = model.layers[l];
            const Matrix<float>& dW = grads.dW[l];
            const std::vector<float>& db = grads.db[l];

            for (size_t i = 0; i < layer.out; ++i)
            {
                // Update each weight for neuron i
                for (size_t j = 0; j < layer.in; ++j)
                    layer.W(i, j) -= lr * dW(i, j);

                // Update bias for neuron i
                layer.b[i] -= lr * db[i];
            }
        }
    }
}