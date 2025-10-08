
// Copyright (c) 2025 Vicente Brisa Saez
// GitHub: Vicen-te
// License: MIT

#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdio>

/**
 * @brief A lightweight host–device tensor wrapper for float data.
 *
 * Manages a contiguous buffer both on host (CPU) and device (GPU),
 * providing convenient methods for allocation, upload, and download.
 */
struct Tensor
{
    std::vector<float> host;  ///< Host (CPU) buffer
    float* dev = nullptr;     ///< Device (GPU) buffer pointer
    size_t size = 0;          ///< Number of elements in the tensor

    Tensor() = default;

    /// Construct tensor and allocate memory for `n` elements
    explicit inline Tensor(size_t n) { allocate(n); }

    /**
     * @brief Allocate host and device memory for `n` elements.
     *        If device memory already exists, it is freed and reallocated.
     */
    void allocate(size_t n)
    {
        if (n == 0) return;

        size = n;
        host.assign(n, 0.0f);

        // Reallocate GPU memory if needed
        if (dev)
        {
            cudaFree(dev);
            dev = nullptr;
        }

        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&dev), size * sizeof(float));
        if (err != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
            throw std::runtime_error("Tensor::allocate() -> cudaMalloc failed");
        }
    }

    /**
     * @brief Ensure that the tensor has at least `n` elements.
     *        Reallocates memory if the size differs.
     */
    void ensure_size(size_t n)
    {
        if (n != size) allocate(n);
    }

    /**
     * @brief Upload host data to device memory.
     */
    void upload() const
    {
        if (size == 0 || !dev) return;

        cudaError_t err = cudaMemcpy(dev, host.data(), size * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "cudaMemcpy host->device failed: %s\n", cudaGetErrorString(err));
            throw std::runtime_error("Tensor::upload() -> cudaMemcpy failed");
        }
    }

    /**
     * @brief Download device data back to host memory.
     */
    void download() const
    {
        if (size == 0 || !dev) return;

        cudaError_t err = cudaMemcpy(const_cast<float*>(host.data()), dev, size * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "cudaMemcpy device->host failed: %s\n", cudaGetErrorString(err));
            throw std::runtime_error("Tensor::download() -> cudaMemcpy failed");
        }
    }

    /**
     * @brief Release GPU and host resources.
     */
    void free()
    {
        if (dev)
        {
            cudaFree(dev);
            dev = nullptr;
        }

        host.clear();
        size = 0;
    }

    /// Destructor — automatically frees GPU memory
    ~Tensor() { free(); }
};
