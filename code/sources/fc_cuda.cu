
// Copyright (c) 2025 Vicente Brisa Saez
// Github: Vicen-te
// License: MIT

#include <cuda_runtime.h>
#include <cstdio>
#include <cassert>
#include <stdexcept>
#include "../headers/fc_cuda.hpp"

// Undefine DEBUG to disable debug prints in this file
#ifdef DEBUG
#undef DEBUG
#endif


// Kernel: each thread computes one output neuron
__global__ void fc_kernel(
    const float* __restrict__ d_W,
    const float* __restrict__ d_b,
    const float* __restrict__ d_x, 
    float* d_y, 
    int in, 
    int out
) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out) return;

    const float* w_row = d_W + (size_t)idx * in;
    float sum = d_b[idx];
    // simple dot (no unroll)
    for (int j = 0; j < in; ++j) sum += w_row[j] * d_x[j];
    d_y[idx] = (sum > 0.0f) ? sum : 0.0f;
}

#if DEBUG
__global__ void fc_kernel_debug(
    const float* __restrict__ d_W,
    const float* __restrict__ d_b,
    const float* __restrict__ d_x,
    float* d_y,
    int in,
    int out
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out) return;

    const float* w_row = d_W + (size_t)idx * in;
    float sum = d_b[idx];

    if (idx < 5) {
        printf("DEBUG INPUTS: idx=%d, d_W=%f, d_b=%f, d_x=%f, w_row=%f\n",
            idx, d_W[0], d_b[0], d_x[0], w_row[0]);
    }

    for (int j = 0; j < in; ++j) {
        sum += w_row[j] * d_x[j];
    }

    if (idx < 5) { 
        printf("Neuron %d: sum_before_ReLU = %f\n", idx, sum);
    }

    // ReLU activation
    d_y[idx] = fmaxf(0.0f, sum);
}
#endif

// Wrapper: expects device pointers already allocated for W,b,x,y
extern "C" void fc_cuda_forward_reuse(
    const float* d_W, 
    const float* d_b, 
    const float* d_x, 
    float* d_y, 
    int in, 
    int out
) 
{
    if (!d_W || !d_b || !d_x || !d_y)
    {
        fprintf(stderr, "Null device pointer in fc_cuda_forward_reuse\n");
        return;
    }

    int block = 256;
    int grid = (out + block - 1) / block;

#if DEBUG
    printf("Launching FC kernel: in=%d out=%d\n", in, out);
#endif

    fc_kernel <<<grid, block>>> (d_W, d_b, d_x, d_y, in, out);

    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) 
    {
        fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(e));
        return;
    }
}
