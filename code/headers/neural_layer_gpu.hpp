

// Copyright (c) 2025 Vicente Brisa Saez
// GitHub: Vicen-te
// License: MIT


#pragma once


// ===================== PROJECT HEADERS =====================
#include "ActivationType.hpp"
#include "Tensor.hpp"


// Forward declarations for CUDA/cuBLAS types
using ActivationType = activation::ActivationType;

struct cublasContext;
typedef struct cublasContext* cublasHandle_t;

struct CUstream_st;
typedef struct CUstream_st* cudaStream_t;


namespace gpu
{

    namespace layer
    {

        // ===================== PROPAGATION =====================
        /**
         * @brief Perform a forward pass of a fully connected layer.
         *
         * Computes Y = activation(X * W^T + b) using cuBLAS for matrix multiplication
         * and CUDA kernels for bias addition and activation functions.
         *
         * @param handle cuBLAS handle for GPU operations.
         * @param stream CUDA stream for asynchronous execution.
         * @param W Weights tensor 
         * @param b Bias tensor 
         * @param X Input activations 
         * @param Y Output activations 
         * @param in Number of input neurons.
         * @param out Number of output neurons.
         * @param batch Batch size.
         * @param act Activation function to apply.
         */
        void forward
        (
            cublasHandle_t handle,
            cudaStream_t stream,
            const std::unique_ptr<float, CudaDeleter>& W,
            const std::unique_ptr<float, CudaDeleter>& b,
            const std::unique_ptr<float, CudaDeleter>& X,
            std::unique_ptr<float, CudaDeleter>& Y,
            const size_t& in,
            const size_t& out,
            const size_t& batch,
            ActivationType act
        );

        /**
          * @brief Backward pass for a hidden layer.
          *
          * Computes gradients w.r.t weights (dW), biases (dB), and input (dX) for a fully connected layer.
          * Uses the output gradient `delta` from the next layer.
          *
          * @param handle cuBLAS handle.
          * @param stream CUDA stream.
          * @param W Layer weights (out x in).
          * @param A_prev Input activations from previous layer.
          * @param delta Gradient of this layer's output.
          * @param dX Gradient to propagate backward to previous layer.
          * @param dW Gradient w.r.t weights (accumulated).
          * @param dB Gradient w.r.t biases (accumulated).
          * @param in Number of inputs.
          * @param out Number of outputs.
          * @param batch Batch size.
          */
        void backward
        (
            cublasHandle_t handle,
            cudaStream_t stream,
            std::unique_ptr<float, CudaDeleter>& W,           
            const std::unique_ptr<float, CudaDeleter>& A_prev,
            std::unique_ptr<float, CudaDeleter>& delta,
            std::unique_ptr<float, CudaDeleter>& dX,          
            std::unique_ptr<float, CudaDeleter>& dW,       
            std::unique_ptr<float, CudaDeleter>& dB,          
            const size_t& in,              
            const size_t& out,             
            const size_t& batch            
        );

        /**
         * @brief Backward pass for the output layer.
         *
         * Computes gradients w.r.t weights (dW) and biases (dB) for the last layer.
         * Does not compute dX because it is not propagated further.
         *
         * @param handle cuBLAS handle.
         * @param stream CUDA stream.
         * @param A_prev Input activations from previous layer.
         * @param delta Gradient of the loss w.r.t layer output.
         * @param dW Gradient w.r.t weights.
         * @param dB Gradient w.r.t biases.
         * @param in Number of inputs.
         * @param out Number of outputs.
         * @param batch Batch size.
         */
        void backward_output
        (
            cublasHandle_t handle,
            cudaStream_t stream,
            const std::unique_ptr<float, CudaDeleter>& A_prev,
            std::unique_ptr<float, CudaDeleter>& delta,
            std::unique_ptr<float, CudaDeleter>& dW,
            std::unique_ptr<float, CudaDeleter>& dB,
            const size_t& in,
            const size_t& out,
            const size_t& batch
        );


        // ===================== ACTIVATION UTILITIES =====================
        /**
         * @brief Apply the derivative of the activation function element-wise.
         *
         * Used during backpropagation to modify delta.
         *
         * @param stream CUDA stream.
         * @param delta Gradient to update.
         * @param Y Activated output of the layer.
         * @param batch Batch size.
         * @param out Number of neurons.
         * @param act Activation function type.
         */
        void activation_backward
        (
            cudaStream_t stream,
            std::unique_ptr<float, CudaDeleter>& delta,
            const std::unique_ptr<float, CudaDeleter>& Y,
            const size_t& batch,
            const size_t& out,
            ActivationType act
        );


        // ===================== PARAMETER UPDATES =====================
        /**
         * @brief Update layer parameters using SGD.
         *
         * Performs: param -= lr * grad
         *
         * @param stream CUDA stream.
         * @param param Parameter tensor to update (weights or biases).
         * @param grad Gradient tensor.
         * @param lr Learning rate.
         * @param size Number of elements in param.
         */
        void sgd_update
        (
            cudaStream_t stream,
            std::unique_ptr<float, CudaDeleter>& param,
            std::unique_ptr<float, CudaDeleter>& grad,
            const float& lr,
            const size_t& size
        );

    }

    namespace loss
    {

        // ===================== LOSS FUNCTIONS =====================
        /**
         * @brief Compute the total cross-entropy loss on the GPU.
         *
         * Launches a CUDA kernel to compute per-block losses and then reduces them on the host.
         *
         * @param y_pred Predicted probabilities (device tensor).
         * @param y_true True labels (device tensor, one-hot).
         * @return Total cross-entropy loss for the batch.
         */
        [[nodiscard]] float cross_entropy(const Tensor& y_pred, const Tensor& y_true);

    }

    // ===================== DELTA INITIALIZATION =====================
    /**
     * @brief Compute the initial delta for the output layer (Softmax + CrossEntropy).
     *
     * Computes delta = y_pred - y_true.
     *
     * @param delta Output gradient tensor to store delta.
     * @param Y Activated output of the layer.
     * @param y_true Ground-truth labels.
     * @param batch Batch size.
     * @param out Number of outputs.
     */
    void compute_delta_initial
    (
        std::unique_ptr<float, CudaDeleter>& delta,
        const std::unique_ptr<float, CudaDeleter>& Y,
        const std::unique_ptr<float, CudaDeleter>& y_true,
        const size_t& batch,
        const size_t& out
    );

}