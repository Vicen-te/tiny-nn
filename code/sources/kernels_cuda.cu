

// Copyright (c) 2025 Vicente Brisa Saez
// GitHub: Vicen-te
// License: MIT


// ===================== MAIN HEADER =====================
#include "../headers/kernels_cuda.cuh"


// ===================== CUDA HEADERS =====================
#include <cublas_v2.h>


namespace gpu
{

    namespace layer::kernels
    {

        __global__ void softmax_per_row(float* Y, int batch, int out)
        {
            int tid = threadIdx.x;
            int row = blockIdx.x;
            if (row >= batch) return;

            extern __shared__ float buffer[];

            // Compute max value in the row for numerical stability
            float max_val = -1e30f;
            for (int j = tid; j < out; j += blockDim.x)
                max_val = fmaxf(max_val, Y[row * out + j]);

            buffer[tid] = max_val;
            __syncthreads();

            // Parallel reduction to find row max
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
            {
                if (tid < stride) buffer[tid] = fmaxf(buffer[tid], buffer[tid + stride]);
                __syncthreads();
            }
            max_val = buffer[0];
            __syncthreads();

            // Compute sum of exponentials
            float sum_exp = 0.f;
            for (int j = tid; j < out; j += blockDim.x)
                sum_exp += expf(Y[row * out + j] - max_val);

            buffer[tid] = sum_exp;
            __syncthreads();

            // Parallel reduction to sum exponentials
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
            {
                if (tid < stride) buffer[tid] += buffer[tid + stride];
                __syncthreads();
            }
            sum_exp = buffer[0];
            __syncthreads();

            // Normalize each element to compute softmax
            for (int j = tid; j < out; j += blockDim.x)
                Y[row * out + j] = expf(Y[row * out + j] - max_val) / sum_exp;

        }

        __global__ void add_bias(float* Y, const float* b, int batch, int out)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch * out) return;
            int col = idx % out;

            // Add bias to each element if bias pointer is valid
            Y[idx] += (b ? b[col] : 0.0f);
        }

        __global__ void accumulate_bias_gradient(const float* dY, float* dB, int batch, int out)
        {
            int j = blockIdx.x * blockDim.x + threadIdx.x;
            if (j >= out) return;

            float sum = 0.0f;
            // Accumulate gradients over batch
            for (int i = 0; i < batch; i++)
                sum += dY[i * out + j];

            // Average gradients across batch
            dB[j] = sum / batch;
        }

        __global__ void apply_activation(float* Y, int N, ActivationType act)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= N) return;

            // Apply activation function element-wise
            Y[idx] = activate(Y[idx], act);
        }

        __global__ void apply_activation_derivative(float* delta, const float* Y, int size, ActivationType act)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size) return;

            // Multiply delta by derivative of activation function
            delta[idx] *= activate_derivative(Y[idx], act);
        }

        __global__ void update_parameters(float* __restrict__ param, const float* __restrict__ grad, float lr, int N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            // Loop over all elements assigned to this thread
            for (int i = idx; i < N; i += blockDim.x * gridDim.x)
                param[i] -= lr * grad[i]; //< SGD parameter update
        }

    }


    namespace loss::kernels
    {

        __global__ void cross_entropy_per_block
        (
            const float* y_pred, const float* y_true,
            float* block_losses, int total)
        {
            extern __shared__ float sdata[];
            int tid = threadIdx.x;
            int idx = blockIdx.x * blockDim.x + tid;

            float loss = 0.0f;
            constexpr float eps = 1e-7f;

            // Compute cross-entropy loss for each element, clamping predictions
            if (idx < total)
            {
                float yp = y_pred[idx];
                float yt = y_true[idx];
                float yp_clamped = fminf(fmaxf(yp, eps), 1.0f);
                loss = -yt * logf(yp_clamped); // __logf
            }

            // Store per-thread loss in shared memory
            sdata[tid] = loss;
            __syncthreads();

            // Reduce per-thread losses to block sum
            for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
            {
                if (tid < s)
                    sdata[tid] += sdata[tid + s];
                __syncthreads();
            }

            // Write block loss to global memory
            if (tid == 0)
                block_losses[blockIdx.x] = sdata[0];
        }

    }


    namespace kernels
    {

        __global__ void delta_softmax_crossentropy
        (
            const float* y_pred, const float* y_true, float* delta, int total
        )
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < total)
                // Compute initial delta for softmax + cross-entropy
                delta[idx] = y_pred[idx] - y_true[idx];
        }

    }

}