

// Copyright (c) 2025 Vicente Brisa Saez
// GitHub: Vicen-te
// License: MIT


#pragma once


// ===================== STANDARD LIBRARIES =====================
#include <vector>


// ===================== CUSTOM HEADERS =====================
#include "model.hpp"
#include "matrix.hpp"


namespace cpu
{

    /**
     * @brief Struct to hold gradients for a neural network model.
     *
     * Contains both weight and bias gradients for all layers, computed during backpropagation.
     */
    struct Gradients
    {
        std::vector<Matrix<float>> dW;       ///< Weight gradients for each layer
        std::vector<std::vector<float>> db;  ///< Bias gradients for each layer
    };


    // ===================== ACTIVATION FUNCTIONS =====================

    /**
     * @brief Rectified Linear Unit (ReLU) activation function.
     * 
     * Computes:
     *     ReLU(x) = max(0, x)
     *
     * It introduces non-linearity while avoiding the vanishing gradient problem
     * for positive values of x.
     * 
     * @param x Input value
     * @return Activated output
     */
    inline float relu(float x) { return x > 0 ? x : 0; }

    /**
     * @brief Derivative of the ReLU activation function.
     *
     * dReLU(x)/dx = 1 if x > 0, else 0
     *
     * This simplification comes from the fact that ReLU is piecewise linear.
     * 
     * @param x Input value
     * @return Derivative at x
     */
    inline float d_relu(float x)
    {
        // Using implicit conversion from bool to float (1.0f if true, 0.0f if false)
        return x > 0 ? 1.0f : 0.0f;
    }

    /**
     * @brief Compute the softmax activation for a vector of inputs.
     *
     * Converts logits (raw scores) into a probability distribution.
     * Each output value is between 0 and 1, and the sum equals 1.
     *
     * Formula:
     *     softmax(z_i) = exp(z_i - max(z)) / sum_j exp(z_j - max(z))
     *
     * Subtracting max(z) improves numerical stability and prevents overflow.
     *
     * @param z Input vector (logits)
     * @return Probability distribution vector of same length as input
     */
    std::vector<float> softmax(const std::vector<float>& z);


    // ===================== LOSS FUNCTIONS =====================

    /**
     * @brief Compute the categorical cross-entropy loss.
     *
     * Used to measure the difference between predicted probabilities and true one-hot labels.
     * 
     * Formula:
     *     L = -sum_i [y_true_i * log(y_pred_i)]
     * 
     * Uses an epsilon to prevent log(0) and numerical instability.
     * 
     * @param y_true Ground truth one-hot vector
     * @param y_pred Predicted probability vector
     * @return Scalar loss value
     */
    float cross_entropy(const std::vector<float>& y_true, const std::vector<float>& y_pred);


    // ===================== FULLY CONNECTED LAYER OPERATIONS =====================

    /**
     * @brief Sequential CPU forward pass for a fully-connected (dense) layer.
     *
     * Computes:
     *     z_i = b_i + sum_j(W_ij * x_j)
     * 
     * Optionally, an activation function can be applied outside this function.
     *
     * @param W Weight matrix of size [out × in]
     * @param b Bias vector of length [out]
     * @param x Input vector of length [in]
     * @return Output vector of length [out]
     */
    std::vector<float> forward_seq(
        const Matrix<float>& W,
        const std::vector<float>& b,
        const std::vector<float>& x
    );

    /**
     * @brief Parallel CPU forward pass for a fully-connected (dense) layer.
     *
     * Utilizes multithreading (e.g., std::execution::par or std::async) to accelerate computation:
     * y = W * x + b
     *
     * @param W Weight matrix of size [out × in]
     * @param b Bias vector of length [out]
     * @param x Input vector of length [in]
     * @return Output vector of length [out]
     */
    std::vector<float> forward_par(
        const Matrix<float>& W,
        const std::vector<float>& b,
        const std::vector<float>& x
    );

    // ===================== FORWARD AND BACKWARD PROPAGATION =====================

    /**
     * @brief Performs the complete forward propagation of the model on CPU.
     *
     * Sequentially computes activations layer by layer.
     * Applies ReLU or Softmax depending on layer configuration.
     *
     * @param model Neural network model containing layers and weights
     * @param x Input vector for the network
     * @return Vector of activations for all layers including input layer
     */
    std::vector<std::vector<float>> forward_cpu(const Model& model, const std::vector<float>& x);

    /**
     * @brief Perform backpropagation through the model on CPU.
     *
     * Computes gradients of weights and biases for each layer using the chain rule.
     *
     * @param model Neural network model to update
     * @param A Vector of layer activations from forward pass
     * @param y_true Ground truth one-hot vector
     * @return Gradients object containing dW and db for all layers
     */
    Gradients backward_cpu(Model& model, const std::vector<std::vector<float>>& A,
        const std::vector<float>& y_true);


    // ===================== WEIGHT AND BIAS UPDATE =====================

    /**
     * @brief Update model weights and biases using stochastic gradient descent (SGD).
     *
     * Applies gradient descent step: 
     *      W = W - lr * dW
     *      b = b - lr * db
     *
     * @param model Neural network model to update
     * @param grads Gradients for weights and biases
     * @param lr Learning rate
     */
    void sgd_update(Model& model, const Gradients& grads, float lr);

}