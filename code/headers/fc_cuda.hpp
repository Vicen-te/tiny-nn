


// Copyright (c) 2025 Vicente Brisa Saez
// Github: Vicen-te
// License: MIT


#pragma once


extern "C" 
{

    /**
     * @brief Forward pass of a fully-connected (dense) layer on the GPU using pre-allocated buffers.
     *
     * This function reuses device memory for weights, biases, input, and output,
     * avoiding repeated allocations for multiple calls.
     *
     * @param d_W Pointer to device weights (size: out × in)
     * @param d_b Pointer to device biases (size: out)
     * @param d_x Pointer to device input vector (size: in)
     * @param d_y Pointer to device output vector (size: out)
     * @param in  Number of input neurons
     * @param out Number of output neurons
     *
     * @note This function is implemented in CUDA and should be linked separately.
     */
    void fc_cuda_forward_reuse(
        const float* d_W,
        const float* d_b,
        const float* d_x,
        float* d_y,
        int in,
        int out
    );
} 
