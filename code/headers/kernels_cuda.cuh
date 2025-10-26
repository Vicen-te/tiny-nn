

// Copyright (c) 2025 Vicente Brisa Saez
// GitHub: Vicen-te
// License: MIT


#pragma once


// ===================== PROJECT HEADERS =====================
#include "ActivationType.hpp"

using ActivationType = activation::ActivationType;


namespace gpu
{

	namespace layer::kernels 
	{

		// ===================== SOFTMAX =====================
		/**
		 * @brief Compute softmax per row (for a batch of outputs).
		 *
		 * Each CUDA block processes one row, threads compute exponentials in parallel.
		 * Uses shared memory for reduction (max and sum for numerical stability).
		 *
		 * @param Y Input/output matrix (batch x out) in row-major order.
		 * @param batch Number of rows (batch size).
		 * @param out Number of columns (neurons in layer).
		 */
		__global__ void softmax_per_row(float* Y, int batch, int out);


		// ===================== BIAS =====================
		/**
		 * @brief Add bias vector to each row of Y.
		 *
		 * @param Y Input/output matrix (batch x out).
		 * @param b Bias vector of length out.
		 * @param batch Batch size.
		 * @param out Number of outputs.
		 */
		__global__ void add_bias(float* Y, const float* b, int batch, int out);

		/**
		 * @brief Accumulate gradient w.r.t bias over a batch.
		 *
		 * Each thread computes sum for one output neuron.
		 *
		 * @param dY Gradient of layer outputs (batch x out).
		 * @param dB Output bias gradients (size out).
		 * @param batch Batch size.
		 * @param out Number of outputs.
		 */
		__global__ void accumulate_bias_gradient
		(
			const float* dY,
			float* db,
			int batch,
			int out
		);


		// ===================== ACTIVATION =====================
		/**
		 * @brief Apply an activation function element-wise.
		 *
		 * @param Y Input/output array.
		 * @param num_elements Total elements in Y.
		 * @param act_type Type of activation to apply.
		 */
		__global__ void apply_activation
		(
			float* Y,
			int num_elements,
			ActivationType act_type
		);

		/**
		 * @brief Multiply delta by derivative of activation function (element-wise).
		 *
		 * @param delta Gradient to update.
		 * @param Y Activated outputs.
		 * @param num_elements Total elements.
		 * @param act_type Activation type.
		 */
		__global__ void apply_activation_derivative
		(
			float* delta,
			const float* Y,
			int num_elements,
			ActivationType act_type
		);


		// ===================== PARAMETER UPDATES =====================
		/**
		 * @brief Perform SGD update: param -= lr * grad.
		 *
		 * @param param Parameters to update.
		 * @param grad Gradients.
		 * @param learning_rate Learning rate.
		 * @param num_elements Number of elements in param.
		 */
		__global__ void update_parameters
		(
			float* __restrict__ param,
			const float* __restrict__ grad,
			float learning_rate,
			int num_elements
		);

	}

	namespace loss::kernels
	{

		// ===================== LOSS =====================
		/**
		 * @brief Compute cross-entropy loss per block.
		 *
		 * Uses shared memory to reduce per-thread losses to one value per block.
		 *
		 * @param y_pred Predictions.
		 * @param y_true True labels.
		 * @param block_losses Output array to store block sums.
		 * @param total Total number of elements.
		 */
		__global__ void cross_entropy_per_block
		(
			const float* y_pred,
			const float* y_true,
			float* block_losses,
			int total
		);

	}

	namespace kernels
	{

		// ===================== DELTA INITIALIZATION =====================
		/**
		 * @brief Compute initial delta for output layer with Softmax + CrossEntropy.
		 *
		 * delta = y_pred - y_true
		 *
		 * @param y_pred Predictions.
		 * @param y_true True labels (one-hot).
		 * @param delta Output gradient (device memory).
		 * @param total Total number of elements.
		 */
		__global__ void delta_softmax_crossentropy
		(
			const float* y_pred,
			const float* y_true,
			float* delta,
			int total
		);

	}

}