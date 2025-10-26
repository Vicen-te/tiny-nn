

// Copyright (c) 2025 Vicente Brisa Saez
// GitHub: Vicen-te
// License: MIT


#pragma once


// ===================== PROJECT HEADERS =====================
#include "Tensor.hpp"
#include "LayerBase.hpp"

using namespace activation;


// Forward declarations for CUDA/cuBLAS types
struct cublasContext;
typedef struct cublasContext* cublasHandle_t;

struct CUevent_st;
typedef struct CUevent_st* cudaEvent_t;

struct CUstream_st;
typedef struct CUstream_st* cudaStream_t;


namespace gpu
{

    /**
     * @brief GPU-based fully connected neural network layer with host-device synchronization.
     *
     * This class represents a single dense neural network layer optimized for GPU computation.
     * It executes forward and backward propagation using cuBLAS and CUDA streams, while
     * maintaining synchronized copies of data on the CPU for easy inspection or serialization.
     *
     * It abstracts low-level GPU operations, focusing on providing a high-level interface
     * for neural network training and inference on the GPU.
     */
    class Layer : public LayerBase
    {

    private:
    

        // ===================== MEMBERS =====================
        Tensor T_W; /// Weights 
        Tensor T_b; /// Biases

        Tensor X;   /// Input activations
        Tensor Y;   /// Output activations

        Tensor dW;  /// Weight gradients
        Tensor db;  /// Bias gradients

        cudaStream_t stream = nullptr; /// Stream for async execution


        // ===================== INTERNAL METHODS =====================
        /**
         * @brief Upload initial parameters to GPU and create CUDA stream.
         *
         * Allocates GPU memory for weights and biases, uploads the CPU-side data to the GPU,
         * and initializes a CUDA stream to manage asynchronous kernel execution.
         */
        void init();

        /**
         * @brief Synchronize updated parameters from GPU to CPU.
         *
         * Transfers the latest GPU-side weights and biases back to the host memory.
         * This ensures that both CPU and GPU remain aligned after training updates.
         */
        void update_W_b();



    public:


        // ===================== CONSTRUCTORS / DESTRUCTOR =====================
        Layer() = default;

        /**
         * @brief Construct a fully connected GPU layer.
         *
         * Initializes dimensions, activation function, and allocates GPU memory.
         *
         * @param input_dim         Number of input neurons.
         * @param output_dim        Number of output neurons.
         * @param activation_type   Type of activation function.
         * @param layer_name        Optional name for the layer.
         */
        Layer
        (
            size_t input_dim,
            size_t output_dim,
            ActivationType activation_type,
            std::string_view layer_name
        );

        /**
         * @brief Destructor.
         *
         * Releases GPU memory and destroys the CUDA stream.
         */
        ~Layer();


        // ===================== SERIALIZATION =====================
        /**
         * @brief Load layer from JSON and initialize GPU memory.
         *
         * Loads weights, biases, and configuration, then uploads them
         * to the GPU and creates a CUDA stream.
         *
         * @param layer JSON object containing serialized layer parameters.
         */
        void from_json(const json& layer) override;


        // ===================== PROPAGATION =====================

        /**
         * @brief Perform the forward pass.
         *
         * Executes the forward propagation (matrix multiplication followed by activation).
         * Can optionally record CUDA events for performance measurement.
         *
         * @param handle    cuBLAS handle for matrix operations.
         * @param input     Input tensor from the previous layer.
         * @param start     Optional CUDA event marking start of execution.
         * @param stop      Optional CUDA event marking end of execution.
         * @return          Reference to the output tensor Y.
         */
        [[nodiscard]] const Tensor& forward
        (
            cublasHandle_t handle, 
            const Tensor& input, 
            cudaEvent_t start = nullptr, 
            cudaEvent_t stop = nullptr
        );

        /**
         * @brief  Perform the backward pass for a hidden layer.
         *
         * Computes activation derivatives and parameter gradients
         * for hidden layers, propagating error to previous layers.
         *
         * @param handle    cuBLAS handle for matrix operations.
         * @param delta     Gradient of this layer's output (current layer).
         * @param start     Optional CUDA event marking the start of execution.
         * @param stop      Optional CUDA event marking the end of execution.
         * @return          Tensor containing the propagated gradient (input delta).
         */
        [[nodiscard]] const Tensor backward
        (
            cublasHandle_t handle, 
            Tensor delta, 
            cudaEvent_t start = nullptr, 
            cudaEvent_t stop = nullptr
        );

        /**
         * @brief Perform backward pass for the output layer.
         *
         * Used for the final layer of the network, where the loss derivative is applied.
         * Computes parameter gradients without generating propagated deltas.
         *
         * @param handle    cuBLAS handle for matrix operations.
         * @param delta     Gradient of this layer's output (current layer).
         * @param start     Optional CUDA event marking the start of execution.
         * @param stop      Optional CUDA event marking the end of execution.
         */
        void backward_output
        (
            cublasHandle_t handle,
            Tensor delta,
            cudaEvent_t start = nullptr,
            cudaEvent_t stop = nullptr
        );


        // ===================== UPDATE =====================
        /**
         * @brief Update weights and biases using SGD.
         *
         * Applies the computed gradients to weights and biases,
         * synchronizes updates to CPU memory, and frees gradient buffers.
         *
         * @param lr     Learning rate.
         * @param start  Optional CUDA event marking the start of execution.
         * @param stop   Optional CUDA event marking the end of execution.
         */
        void update
        (
            const float& lr, 
            cudaEvent_t start = nullptr,
            cudaEvent_t stop = nullptr
        );

    };

}
