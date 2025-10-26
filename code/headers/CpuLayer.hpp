

// Copyright (c) 2025 Vicente Brisa Saez
// Github: Vicen-te
// License: MIT


#pragma once


// ===================== STANDARD HEADERS =====================
#include <vector>
#include <string_view>
#include <utility>
#include <span>


// ===================== PROJECT HEADERS =====================
#include "ActivationType.hpp"
#include "ParallelExecutor.hpp"
#include "Matrix.hpp"
#include "LayerBase.hpp"
#include "json.hpp"

using json = nlohmann::json;
using ActivationType = activation::ActivationType;


namespace cpu
{

    /**
     * @class Layer
     * @brief Fully connected neural network layer.
     *
     * This layer supports forward and backward propagation, gradient accumulation,
     * and stochastic gradient descent updates. Parallel execution is optional.
     */
    class Layer : public LayerBase, ParallelExecutor
    {

    private:


        // ===================== INTERNAL METHODS =====================
        /**
         * @brief  Helper template to loop over indices either sequentially or in parallel.
         *
         * @tparam Func Callable to execute for each index.
         * @param count Number of iterations.
         * @param func Function to execute per iteration.
         * @param parallel If true, uses parallel execution; otherwise sequential.
         */
        template <typename Func>
        inline void run_for(size_t count, Func&& func, bool parallel)
        {
            if (parallel) run(count, std::forward<Func>(func));
            else for (size_t i = 0; i < count; ++i) func(i);
        }

        /**
         * @brief Compute the pre-activation (weighted sum + bias) of a single neuron.
         *
         * @param i Neuron index.
         * @param X Input activations from previous layer.
         * @return Weighted sum for neuron i before activation.
         */
        [[nodiscard]] float forward_neuron(size_t i, std::span<const float> X);

        /**
         * @brief Compute gradient for one neuron.
         *
         * Updates bias gradient and weight gradients using the neuron’s delta
         * and activations from the previous layer.
         * 
         * @param i Neuron index.
         * @param delta Gradient of current neuron.
         * @param a_prev Activations from previous layer.
         */
        void grad_neuron
        (
            size_t i,
            std::span<const float> delta,
            std::span<const float> a_prev
        );

        /**
         * @brief Compute gradient (delta) for a neuron in the previous layer.
         *
         * Used during backpropagation to propagate error backwards.
         *
         * @param i Neuron index in previous layer.
         * @param delta Gradients from current layer.
         * @param a_prev Activations from previous layer.
         * @param act_prev Activation type of previous layer.
         * @return Gradient for neuron i in previous layer.
         */
        [[nodiscard]] float delta_prev_neuron
        (
            size_t i,
            std::span<const float> delta,
            std::span<const float> a_prev,
            ActivationType act_prev
        ) const;

        /**
         * @brief Average gradients for a single neuron over the batch.
         *
         * @param i Neuron index.
         * @param batch_size Number of examples in batch.
         */
        void average_neuron(size_t i, size_t batch_size);

        /**
         * @brief Apply SGD update for one neuron.
         *
         * Updates weights and bias using the computed gradients.
         *
         * @param i Neuron index.
         * @param lr Learning rate.
         */
        void sgd_update(size_t i, float lr);

        /**
         * @brief Allocate and initialize gradient containers.
         */
        void init();

        /**
         * @brief Reset all gradients to zero.
         */
        void reset_gradients();



    public:


        // ===================== MEMBERS =====================
        std::vector<float> db;  /// Bias gradients
        Matrix<float> dW;       /// Weight gradients


        // ===================== CONSTRUCTORS =====================
        Layer() = default;

        /**
         * @brief Construct a layer with specified dimensions and activation.
         *
         * @param input_dim Number of input neurons.
         * @param output_dim Number of output neurons.
         * @param activation_type Activation function.
         * @param layer_name Optional layer name.
         */
        Layer
        (
            size_t input_dim,
            size_t output_dim,
            ActivationType activation_type,
            std::string_view layer_name
        );


        // ===================== SERIALIZATION =====================
        /**
         * @brief Load layer from JSON and initialize gradients.
         *
         * @param layer JSON object containing serialized layer data.
         */
        void from_json(const json& layer) override;


        // ===================== PROPAGATION =====================
        /**
         * @brief Forward pass for the layer.
         *
         * Computes pre-activations (Z) for each neuron, applies activation,
         * and returns output activations.
         *
         * @param X Input activations from previous layer.
         * @param parallel If true, execute in parallel for each neuron.
         * @return Output activations of the layer.
         */
        [[nodiscard]] const std::vector<float> forward
        (
            std::span<const float> X,
            bool parallel = false
        );

        /**
         * @brief Backward pass for the layer.
         *
         * Computes gradients of weights, biases, and optionally propagates delta
         * to the previous layer.
         *
         * @param delta_next Gradient from next layer.
         * @param a_prev Activations from previous layer.
         * @param delta_prev_needed If true, compute and return delta for previous layer.
         * @param parallel If true, execute in parallel for each neuron.
         * @return Propagated delta for previous layer, or empty vector if not needed.
         */
        [[nodiscard]] const std::vector<float> backward
        (
            std::span<const float> delta_next,
            std::span<const float> a_prev,
            bool delta_prev_needed = true,
            bool parallel = false
        );


        // ===================== GRADIENTS / UPDATE =====================
        /**
         * @brief Average gradients over the batch.
         *
         * @param batch_size Number of examples in batch.
         * @param parallel If true, execute in parallel per neuron.
         */
        void average_gradients(size_t batch_size, bool parallel = false);
   
        /**
         * @brief Apply stochastic gradient descent update.
         *
         * Updates weights and biases for all neurons using computed gradients,
         * then resets gradients to zero.
         *
         * @param lr Learning rate.
         * @param parallel If true, update neurons in parallel.
         */
        void update(float lr, bool parallel = false);

    };

}