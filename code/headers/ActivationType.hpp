

// Copyright (c) 2025 Vicente Brisa Saez
// GitHub: Vicen-te
// License: MIT


#pragma once


// -------------------------------------------------
// Macro for CUDA / CPU compatibility
// -------------------------------------------------
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __device__ ///< Marks a function as CUDA device function
#else
#define CUDA_HOST_DEVICE
// ===================== STANDARD HEADERS =====================
#include <vector>
#include <cmath>
#endif


namespace activation
{

    // ===================== ENUMERATIONS =====================
    /**
     * @brief Supported activation types for neural networks.
     */
    enum class ActivationType : int
    {
        NONE,
        RELU,
        SIGMOID,
        TANH,
        SOFTMAX
    };


    // ===================== ACTIVATION FUNCTIONS =====================
    /**
     * @brief Numerically stable sigmoid function.
     *
     * This implementation avoids overflow for large positive or negative x.
     * Works on both CPU and CUDA.
     * 
     * @param x Input value
     * @return Sigmoid output
     * 
     */
    CUDA_HOST_DEVICE inline float sigmoid_stable(float x)
    {
        if (x >= 0.0f)
        {
            float z = expf(-x); // expf funciona en CUDA y CPU
            return 1.0f / (1.0f + z);
        }
        else
        {
            float z = expf(x);
            return z / (1.0f + z);
        }
    }

    /**
     * @brief Apply the selected activation function.
     * @param x Input value
     * @param type Activation type
     * @return Activated value
     */
    CUDA_HOST_DEVICE inline float activate(float x, ActivationType type)
    {
        switch (type)
        {
        case ActivationType::RELU:     return x > 0.0f ? x : 0.0f;
        case ActivationType::SIGMOID:  return sigmoid_stable(x);
        case ActivationType::TANH:     return tanhf(x); // tanhf funciona en CPU y CUDA
        default:                       return x;
        }
    }


    // ===================== DERIVATIVES =====================
    /**
     * @brief Compute derivative of activation function w.r.t output.
     *
     * The derivative is used in backpropagation for neural networks.
     * 
     * @param y Output of the activation function
     * @param type Activation type
     * @return Derivative value
     */
    CUDA_HOST_DEVICE inline float activate_derivative(float y, ActivationType type)
    {
        switch (type)
        {
        case ActivationType::RELU:     return y > 0.0f ? 1.0f : 0.0f;
        case ActivationType::SIGMOID:  return y * (1.0f - y);
        case ActivationType::TANH:     return 1.0f - y * y;
        default:                       return 1.0f;
        }
    }


    // ===================== SOFTMAX (CPU ONLY) =====================
#ifndef __CUDACC__
    /**
     * @brief Compute softmax for a vector (CPU only).
     * 
     * The implementation subtracts the maximum value from all elements
     * to improve numerical stability.
     * 
     * @param z Input vector
     * @return Softmax-normalized vector
     *
     */
    std::vector<float> softmax(const std::vector<float>& z);
#endif

};

